from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from typing import Tuple

from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.validation_tests.validation_utils import evaluator_toggled
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline

def summarize_fdr_results(
    df: pd.DataFrame,
    pval_col: str = 'pval',
    fdr_sig_col: str = 'fdr_significant',
    correction_alpha: float = 0.05,
    groupby_col: str = None,
    as_markdown: bool = False
):
    """
    Summarizes FDR results for logging or reporting.

    Args:
        df: DataFrame with p-value and FDR significance columns.
        pval_col: Name of the p-value column.
        fdr_sig_col: Name of the boolean FDR significance column.
        correction_alpha: Alpha level used for FDR.
        groupby_col: Optionally, a column (e.g., rule size) to break down results.
        as_markdown: If True, returns markdown-formatted string; else, returns summary DataFrame.

    Returns:
        Markdown string or summary DataFrame.
    """
    def _summarize(sub_df):
        total = len(sub_df)
        sig_count = sub_df[fdr_sig_col].sum()
        non_sig_count = total - sig_count
        summary = {
            "Total Tested": total,
            "Significant (FDR)": sig_count,
            "Proportion Significant": sig_count / total if total else float('nan'),
            f'P<{correction_alpha}': (sub_df[pval_col] < correction_alpha).sum(),
            'P<0.01': (sub_df[pval_col] < 0.01).sum(),
            'P<0.05': (sub_df[pval_col] < 0.05).sum(),
            'Min P-value': sub_df[pval_col].min(),
            'Median P-value': sub_df[pval_col].median(),
            'Max P-value': sub_df[pval_col].max(),
        }
        return pd.Series(summary)

    if groupby_col:
        summary_df = df.groupby(groupby_col).apply(_summarize).reset_index()
    else:
        summary_df = _summarize(df).to_frame().T

    if not as_markdown:
        return summary_df

    # Markdown formatting
    def _df_to_md(sub_df):
        md = f"### FDR Correction Summary\n\n"
        for col, val in sub_df.iloc[0].items():
            md += f"- **{col}:** {val}\n"
        return md

    if groupby_col:
        md_blocks = []
        for _, row in summary_df.iterrows():
            md = f"#### Group: {row[groupby_col]}\n"
            for col, val in row.drop(groupby_col).items():
                md += f"- **{col}:** {val}\n"
            md_blocks.append(md)
        return "\n".join(md_blocks)
    else:
        return _df_to_md(summary_df)

def compute_empirical_pvals(actual_stats: np.ndarray, null_stats: np.ndarray, greater_is_better: bool = True) -> np.ndarray:
    """
    Computes empirical p-values by comparing actual statistics to a null distribution.
    """
    if greater_is_better:
        return np.array([(null_stats >= val).mean() for val in actual_stats])
    else:
        return np.array([(null_stats <= val).mean() for val in actual_stats])

def run_fdr(
    df: pd.DataFrame,
    null_data: pd.DataFrame,
    cfg: Config,
    early_stop_metric: str,
    correction_metric: str,
    correction_alpha: float,
    logger: MarkdownLogger,
    mine_rules: bool,
    greater_is_better: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Full pipeline for FDR correction after rule mining and empirical null testing.
    
    Args:
        df: Cleaned/engineered input DataFrame.
        null_data: DataFrame with null distribution statistics (same structure as df).
        cfg: Configuration object for edge discovery pipeline.
        early_stop_metric: Name of column with the test statistic.
        correction_metric: Multiple testing correction method ('fdr_bh', 'fdr_by', etc.).
        correction_alpha: FDR significance threshold (e.g., 0.05).
        logger: Logging object
        mine_rules: Whether to mine rules in the pipeline.
        greater_is_better: If True, higher statistic is better (right tail).
    
    Returns:
        combined_results: DataFrame with results and FDR fields.
        log_summary: Markdown summary for logging/reporting.
    """
    # --- Run rule mining / edge discovery
    combined_results, rules_log_df, combined_summary = run_edge_discovery_pipeline(
        prepared_df=df, cfg=cfg, logger=logger, mine_rules=mine_rules
    )

    # --- Check data validity
    if combined_results.empty:
        raise ValueError("No results to evaluate after rule mining.")
    if null_data.empty:
        raise ValueError("Null data is empty – cannot compute empirical p-values.")
    if early_stop_metric not in combined_results.columns or early_stop_metric not in null_data.columns:
        raise KeyError(f"Metric '{early_stop_metric}' must be present in both results and null data.")

    # --- Compute empirical p-values
    actual_stats = combined_results[early_stop_metric].values
    null_stats = null_data[early_stop_metric].values
    pvals = compute_empirical_pvals(actual_stats, null_stats, greater_is_better=greater_is_better)
    combined_results = combined_results.copy()
    combined_results['pval'] = pvals

    # --- FDR correction
    try:
        rej, pvals_corr, _, _ = multipletests(pvals, alpha=correction_alpha, method=correction_metric)
    except Exception as e:
        raise

    pval_col = f'pval_{correction_metric}'
    sig_col = f'{correction_metric}_significant'
    combined_results[pval_col] = pvals_corr
    combined_results[sig_col] = rej

    # --- Summarize for logging
    log_summary = summarize_fdr_results(
        df=combined_results,
        pval_col='pval',
        fdr_sig_col=sig_col,
        correction_alpha=correction_alpha,
        as_markdown=False
    )
    logger.log_step(
        step_name="FDR / Multiple Correction Test",
        info={
            "early_stop_metric": early_stop_metric,
            "correction_metric": correction_metric,
            "correction_alpha": correction_alpha,
            "mine_rules": mine_rules,
        },
        df=log_summary
    )

    return combined_results, log_summary

def fdr_main(
    prepared_df: pd.DataFrame, 
    null_data: pd.DataFrame,
    cfg: Config,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    func = run_fdr
    func_args = {
        "cfg": cfg, 
        "null_data": null_data, 
        "early_stop_metric": cfg.early_stop_metric, 
        "correction_metric": cfg.correction_metric, 
        "correction_alpha": cfg.correction_alpha, 
        "logger": logger
    }
    
    fdr_res, fdr_log = evaluator_toggled(
        prepared_df, 
        cfg, 
        func, 
        func_args, 
        "discover_per_split", # For single run no reason to discover first
        logger, 
    )
    return fdr_res, fdr_log