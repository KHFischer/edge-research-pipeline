import pandas as pd
from typing import List, Union, Callable, Tuple
from functools import reduce

from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.validation_tests.validation_utils import evaluator_toggled
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline

def summarize_splits(
    splits: List[pd.DataFrame],
    date_column: str
) -> pd.DataFrame:
    """
    Summarizes metadata for each split (start/end date, row count).
    
    Parameters:
        splits (List[pd.DataFrame]): List of dataframes (splits).
        date_column (str): Name of the date column.
        
    Returns:
        pd.DataFrame: Summary dataframe with per-split metadata.
    """
    summary_data = []
    
    for i, split_df in enumerate(splits, start=1):
        if date_column not in split_df.columns:
            raise ValueError(f"'{date_column}' not found in split {i}")
        
        # Handle empty splits gracefully
        if split_df.empty:
            start_date = pd.NaT
            end_date = pd.NaT
            row_count = 0
        else:
            # Ensure date column is datetime
            dates = pd.to_datetime(split_df[date_column])
            start_date = dates.min()
            end_date = dates.max()
            row_count = len(split_df)
        
        summary_data.append({
            "split_number": i,
            "start_date": start_date,
            "end_date": end_date,
            "row_count": row_count
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df
    
def split_dataframe_by_date(
    df: pd.DataFrame,
    date_column: str,
    n_splits: int,
    logger: MarkdownLogger
) -> List[pd.DataFrame]:
    """
    Splits a dataframe into N chronological, non-overlapping splits by date.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        date_column (str): The name of the date column.
        n_splits (int): The number of splits to create.
        
    Returns:
        List[pd.DataFrame]: A list of dataframes, each containing one split.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    
    if date_column not in df.columns:
        raise ValueError(f"'{date_column}' column not found in dataframe.")
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # Determine split indices
    split_indices = [
        int(i * len(df) / n_splits)
        for i in range(n_splits + 1)
    ]
    
    # Slice dataframe into splits
    splits = [
        df.iloc[split_indices[i]:split_indices[i + 1]]
        for i in range(n_splits)
    ]
    # Log for auditability
    split_log = summarize_splits(splits, date_column)
    logger.log_step(
        step_name="WFA Split Log",
        info={
            "date_col": date_column,
            "n_splits": n_splits,
        },
        df=split_log
    )
    return splits, split_log

def summarize_wfa_results(
    merged_df: pd.DataFrame,
    merge_keys: List[str]
) -> pd.DataFrame:
    """
    Summarizes WFA merged results, computing descriptive stats per split metric.
    
    Parameters:
        merged_df (pd.DataFrame): The merged WFA results dataframe.
        merge_keys (List[str]): The columns used for merging splits (e.g., ['antecedents', 'consequents', 'depth']).
    
    Returns:
        pd.DataFrame: Summary dataframe with per-metric descriptive statistics.
    """
    # Identify metric columns (anything not a merge key)
    metric_columns = [col for col in merged_df.columns if col not in merge_keys]

    summary_records = []

    for col in metric_columns:
        series = merged_df[col]
        summary_records.append({
            "metric_column": col,
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "non_null_count": series.notnull().sum()
        })

    summary_df = pd.DataFrame(summary_records)

    return summary_df
    
def run_wfa(
    df: pd.DataFrame,
    cfg: Config,
    use_metrics: Union[List[str], str],
    date_col: str,
    wfa_n_splits: int,
    logger: MarkdownLogger,
    mine_rules: bool = True
) -> pd.DataFrame:
    """
    Runs Walk-Forward Analysis over pre-split datasets, applies edge discovery,
    selects and renames metric columns per split, and merges results.
    
    Parameters:
        splits (List[pd.DataFrame]): List of data splits to process.
        cfg: Configuration object for the pipeline.
        use_metrics (Union[List[str], str]): Metrics to keep ('all' or list of names).
        logger: Logger instance.
        mine_rules (bool): Whether to perform rule mining.
        
    Returns:
        pd.DataFrame: Merged dataframe with per-split metrics.
    """
    # Split dataset into N splits
    splits, split_log = split_dataframe_by_date(df, date_col, wfa_n_splits, logger)

    if not splits:
        raise ValueError("Splits list is empty.")
    
    merge_keys = ["antecedents", "consequents", "depth"]
    dfs = []

    for idx, split_df in enumerate(splits, start=1):        
        combined_results, rules_log_df, combined_summary = run_edge_discovery_pipeline(
            prepared_df=split_df,
            cfg=cfg,
            logger=logger,
            mine_rules=mine_rules
        )
        
        # Determine columns to keep
        if use_metrics == "all":
            selected_cols = combined_results.columns.tolist()
        else:
            selected_cols = merge_keys + ['obs', 'selected'] + use_metrics
        
        # Select columns
        filtered_df = combined_results[selected_cols].copy()
        
        # Rename metric columns to reflect split
        renamed_cols = {
            col: f"split_{idx}_{col}"
            for col in filtered_df.columns
            if col not in merge_keys
        }
        filtered_df.rename(columns=renamed_cols, inplace=True)
        
        dfs.append(filtered_df)
    
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=merge_keys, how="outer"),
        dfs
    )    
    # Log for auditability
    wfa_log = summarize_wfa_results(merged_df, merge_keys)
    logger.log_step(
        step_name="WFA Results Log",
        info={
            "use_metrics": use_metrics,
            "mine_rules": mine_rules,
        },
        df=wfa_log
    )
    return merged_df, wfa_log

def wfa_main(
    prepared_df: pd.DataFrame, 
    cfg: Config,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    func = run_wfa
    func_args = {
        "cfg": cfg, 
        "use_metrics": cfg.use_metrics, 
        "date_col": cfg.date_col, 
        "wfa_n_splits": cfg.wfa_n_splits,
        "logger": logger
    }
    
    wfa_res, wfa_log = evaluator_toggled(
        prepared_df, 
        cfg, 
        func, 
        func_args, 
        cfg.discovery_mode, 
        logger, 
    )
    return wfa_res, wfa_log