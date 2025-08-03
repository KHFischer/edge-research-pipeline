import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm 

from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.validation_tests.validation_utils import evaluator_toggled
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline

def resample_dataframe(
    df: pd.DataFrame,
    mode: str,
    block_size: int = 5,
    date_col: str = "date",
    id_cols: List[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Resamples a dataframe in one of three modes: traditional bootstrap, block bootstrap, or block bootstrap per group.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        mode (str): One of 'traditional', 'block', 'block_ids'.
        block_size (int): Block size for block bootstrap modes.
        date_col (str): Name of the date column.
        id_cols (List[str]): List of columns to group by for block_ids mode.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Resampled dataframe.
    """
    if mode not in {"traditional", "block", "block_ids"}:
        raise ValueError("mode must be one of 'traditional', 'block', 'block_ids'.")
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")
    if block_size < 1:
        raise ValueError("block_size must be >=1.")
    if mode == "block_ids" and (id_cols is None or len(id_cols) == 0):
        raise ValueError("id_cols must be provided for mode='block_ids'.")

    rng = np.random.default_rng(random_state)
    n_rows = len(df)

    # Traditional i.i.d bootstrap
    if mode == "traditional":
        indices = rng.integers(low=0, high=n_rows, size=n_rows)
        return df.iloc[indices].reset_index(drop=True)

    # Block bootstrap helpers
    def block_sample(single_df: pd.DataFrame) -> pd.DataFrame:
        single_df = single_df.sort_values(date_col).reset_index(drop=True)
        n = len(single_df)
        n_blocks = int(np.ceil(n / block_size))
        max_start = n - block_size
        if max_start < 0:
            raise ValueError("block_size larger than length of the group or dataframe.")
        start_indices = rng.integers(low=0, high=max_start + 1, size=n_blocks)

        blocks = [single_df.iloc[start:start + block_size] for start in start_indices]
        result = pd.concat(blocks, ignore_index=True)
        return result.iloc[:n].reset_index(drop=True)

    if mode == "block":
        return block_sample(df)

    if mode == "block_ids":
        grouped = df.groupby(id_cols, group_keys=False, sort=False)
        resampled_groups = [block_sample(g) for _, g in grouped]
        return pd.concat(resampled_groups, ignore_index=True)

    # Should never reach here
    raise RuntimeError("Unhandled resampling mode.")

def summarize_rule_metrics(
    df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    agg_funcs = {
        col: [
            "mean",
            "std",
            "min",
            "max",
            lambda x: x.quantile(0.05),
            lambda x: x.quantile(0.95)
        ]
        for col in metrics
    }
    agg_funcs["selected"] = ["sum", "count"]

    summary = (
        df
        .groupby(["antecedents", "consequents"])
        .agg(agg_funcs)
        .reset_index()
    )

    custom_names = {
        "<lambda_0>": "q05",
        "<lambda_1>": "q95",
        "sum": "selected_count",
        "count": "test_count"
    }
    summary.columns = [
        f"{c[0]}_{custom_names.get(c[1], c[1])}" if c[1] else c[0]
        for c in summary.columns
    ]

    summary["selected_fraction"] = (
        summary["selected_selected_count"] / summary["selected_test_count"]
    )
    return summary

def create_validation_summary_log(
    summary_df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Creates a summary dataframe with aggregate statistics about the validation run.
    
    Parameters:
        summary_df (pd.DataFrame): The dataframe returned by summarize_rule_metrics().
        metrics (List[str]): List of metric base names (e.g., ['support', 'confidence', 'lift']).
    
    Returns:
        pd.DataFrame: Single-row dataframe with summary statistics.
    """
    n_rules = len(summary_df)
    # Assuming all rules have same test count, take max
    total_tests = summary_df["selected_test_count"].max()
    avg_selection_rate = summary_df["selected_fraction"].mean()

    # Aggregate stats for each metric
    stat_records = {}
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"

        stat_records[f"{metric}_mean_mean"] = summary_df[mean_col].mean()
        stat_records[f"{metric}_mean_std"] = summary_df[mean_col].std()
        stat_records[f"{metric}_mean_min"] = summary_df[mean_col].min()
        stat_records[f"{metric}_mean_max"] = summary_df[mean_col].max()

    # Compose all stats into one record
    summary_record = {
        "total_rules": n_rules,
        "total_tests": total_tests,
        "avg_selection_rate": avg_selection_rate,
        **stat_records
    }

    return pd.DataFrame([summary_record])

def maybe_tqdm(iterable, use_tqdm: bool, **tqdm_kwargs):
    if use_tqdm:
        from tqdm.auto import tqdm
        return tqdm(iterable, **tqdm_kwargs)
    return iterable

def run_bootstrap(
    df: pd.DataFrame,
    cfg: Config,
    resample_method: str,
    block_size: int,
    date_col: str,
    id_cols: List[str],
    n_bootstrap: int,
    use_metrics: List[str],
    logger: MarkdownLogger,
    verbose: bool = True,
    mine_rules: bool = True,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs bootstrap validation by repeatedly resampling the dataset and evaluating rules.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        cfg: Configuration object for the edge discovery pipeline.
        resample_method (str): One of 'traditional', 'block', 'block_ids'.
        block_size (int): Size of blocks for block bootstrap modes.
        date_col (str): Name of the date column.
        id_cols (List[str]): Columns to group by for 'block_ids' mode.
        n_bootstrap (int): Number of bootstrap iterations.
        use_metrics (List[str]): Metrics to include in summarization.
        logger: Logger instance.
        mine_rules (bool): Whether to perform rule mining during bootstrap.
        random_state (int): Optional random seed for reproducibility.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Detailed bootstrap results per rule, summary log dataframe)
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    if not use_metrics:
        raise ValueError("use_metrics must be a non-empty list.")
    

    all_results = []
    rng = np.random.default_rng(random_state)

    for i in maybe_tqdm(range(n_bootstrap), verbose, total=n_bootstrap, desc="Bootstrapping"):
        
        # Resample the dataframe
        resampled_df = resample_dataframe(
            df=df,
            mode=resample_method,
            block_size=block_size,
            date_col=date_col,
            id_cols=id_cols,
            random_state=rng.integers(0, 1e9)  # different seed per iteration
        )
        
        # Run edge discovery
        res, _, _ = run_edge_discovery_pipeline(
            prepared_df=resampled_df,
            cfg=cfg,
            logger=logger,
            mine_rules=mine_rules
        )
        
        res["test_num"] = i
        all_results.append(res)

    final_result = pd.concat(all_results, ignore_index=True)
    
    metrics_for_summary = ["obs"] + use_metrics
    bootstrap_results = summarize_rule_metrics(
        final_result,
        metrics=metrics_for_summary
    )

    bootstrap_log = create_validation_summary_log(
        bootstrap_results,
        metrics=metrics_for_summary
    )

    logger.log_step(
        step_name="Bootstrap Test",
        info={
            "resample_method": resample_method,
            "block_size": block_size,
            "date_col": date_col,
            "id_cols": id_cols,
            "n_bootstrap": n_bootstrap,
            "mine_rules": mine_rules,
            "random_state": random_state,
        },
        df=bootstrap_log
    )

    return bootstrap_results, bootstrap_log

def bootstrap_main(
    prepared_df: pd.DataFrame, 
    cfg: Config,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    func = run_bootstrap
    func_args = {
        "cfg": cfg, 
        "resample_method": cfg.resample_method, 
        "block_size": cfg.block_size, 
        "date_col": cfg.date_col, 
        "id_cols": cfg.id_cols, 
        "n_bootstrap": cfg.n_bootstrap, 
        "use_metrics": cfg.use_metrics,
        "verbose": cfg.bootstrap_verbose,
        "logger": logger
    }
    
    bootstrap_results, bootstrap_log = evaluator_toggled(
        prepared_df, 
        cfg, 
        func, 
        func_args, 
        cfg.discovery_mode, 
        logger, 
    )
    return bootstrap_results, bootstrap_log