from sklearn.model_selection import train_test_split
from typing import List, Union, Callable, Tuple
import pandas as pd

from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline
from scripts.validation_tests.validation_utils import evaluator_toggled

def create_split_summary_df(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary DataFrame with key statistics for train and test splits.
    """
    summary = []

    for name, df in [("train", train), ("test", test)]:
        dates = pd.to_datetime(df["date"])
        summary.append({
            "split": name,
            "n_rows": len(df),
            "n_unique_dates": df["date"].nunique(),
            "earliest_date": dates.min(),
            "latest_date": dates.max()
        })

    return pd.DataFrame(summary)

def train_test_split(
    prepared_df: pd.DataFrame,
    split_method: str,
    split_date: str,
    split_pct: float,
    date_col: str,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe into train and test sets by date or percentage of unique dates.

    Parameters
    ----------
    prepared_df : pd.DataFrame
        The input dataframe.
    split_method : str
        'date' or 'percentage'.
    split_date : str
        If split_method == 'date', the cutoff date (train < date, test >= date).
    split_pct : float
        If split_method == 'percentage', fraction of dates in train (0-1).
    date_col : str
        Name of the date column.
    logger : MarkdownLogger
        Logger instance for logging split info.

    Returns
    -------
    Tuple of:
        - train dataframe
        - test dataframe
        - split summary dataframe
    """
    if split_method not in {"date", "percentage"}:
        raise ValueError("split_method must be 'date' or 'percentage'")

    # Convert dates to Timestamps for robust sorting/comparison
    prepared_df[date_col] = pd.to_datetime(prepared_df[date_col])

    if split_method == "percentage":
        unique_dates = sorted(prepared_df[date_col].unique())
        if not 0 < split_pct < 1:
            raise ValueError("split_pct must be between 0 and 1.")
        cutoff_idx = int(len(unique_dates) * split_pct)
        train_dates = unique_dates[:cutoff_idx]
        test_dates = unique_dates[cutoff_idx:]
        train = prepared_df[prepared_df[date_col].isin(train_dates)]
        test = prepared_df[prepared_df[date_col].isin(test_dates)]
    else:
        split_ts = pd.to_datetime(split_date)
        train = prepared_df[prepared_df[date_col] < split_ts]
        test = prepared_df[prepared_df[date_col] >= split_ts]

    # Check for overlapping dates
    train_dates_set = set(train[date_col].unique())
    test_dates_set = set(test[date_col].unique())
    overlap = train_dates_set & test_dates_set
    if overlap:
        raise ValueError(
            f"Train/test date overlap detected: {sorted(overlap)}"
        )

    # Create summary dataframe
    split_log = create_split_summary_df(train, test)

    logger.log_step(
        step_name="Train/Test Split Log",
        info={
            "date_col": date_col,
            "split_method": split_method,
            "split_date": split_date,
            "split_pct": split_pct,
        },
        df=split_log
    )

    return train, test, split_log

def create_result_summary_df(res: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary DataFrame with high-level statistics about the merged results.

    Parameters
    ----------
    res : pd.DataFrame
        The merged results DataFrame containing selected flags and lift metrics.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame summarizing key metrics.
    """
    # Safe counts
    n_total = len(res)
    n_selected_train = (res["selected_train"] == True).sum()
    n_selected_test = (res["selected_test"] == True).sum()
    n_both_selected = ((res["selected_train"] == True) & (res["selected_test"] == True)).sum()
    n_train_only = ((res["selected_train"] == True) & (res["selected_test"] != True)).sum()
    n_test_only = ((res["selected_train"] != True) & (res["selected_test"] == True)).sum()
    n_neither_selected = ((res["selected_train"] != True) & (res["selected_test"] != True)).sum()

    # Lift stats
    lift_train_mean = res["lift_train"].mean()
    lift_train_std = res["lift_train"].std()
    lift_test_mean = res["lift_test"].mean()
    lift_test_std = res["lift_test"].std()

    # Merge type counts
    merge_counts = res["_merge"].value_counts().to_dict()
    n_left_only = merge_counts.get("left_only", 0)
    n_right_only = merge_counts.get("right_only", 0)
    n_both = merge_counts.get("both", 0)

    # Build the summary DataFrame
    summary = pd.DataFrame([{
        "n_rules_total": n_total,
        "n_rules_selected_train": n_selected_train,
        "n_rules_selected_test": n_selected_test,
        "n_rules_both_selected": n_both_selected,
        "n_rules_train_only": n_train_only,
        "n_rules_test_only": n_test_only,
        "n_rules_neither_selected": n_neither_selected,
        "n_left_only": n_left_only,
        "n_right_only": n_right_only,
        "n_both": n_both,
        "lift_train_mean": lift_train_mean,
        "lift_train_std": lift_train_std,
        "lift_test_mean": lift_test_mean,
        "lift_test_std": lift_test_std
    }])

    return summary
    
def run_train_test(
    df: pd.DataFrame,
    cfg: Config,
    split_method: str,
    split_date: str,
    split_pct: float,
    date_col: str,
    use_metrics: list,
    logger: MarkdownLogger,
    mine_rules: bool = True
):
    standard_cols = ['antecedents', 'consequents', 'obs', 'selected', 'depth']
    
    # Create train/test/split
    train, test, split_log = train_test_split(
        df,
        split_method,
        split_date,
        split_pct,
        date_col,
        logger
    )
    
    # Train split
    train_combined_results, train_rules_log_df, train_combined_summary = run_edge_discovery_pipeline(
        prepared_df=train, cfg=cfg, logger=logger, mine_rules=mine_rules
    )
    
    if not use_metrics == 'all':
        select_cols = standard_cols + use_metrics
        train_combined_results = train_combined_results[select_cols].copy()
    
    # Test split
    test_combined_results, test_rules_log_df, test_combined_summary = run_edge_discovery_pipeline(
        prepared_df=test, cfg=cfg, logger=logger, mine_rules=mine_rules
    )
    
    if not use_metrics == 'all':
        select_cols = standard_cols + use_metrics
        test_combined_results = test_combined_results[select_cols].copy()
    
    # Merge results: outer join on keys to preserve unmatched
    merge_keys = ["antecedents", "consequents", "depth"]
    
    merged_results = train_combined_results.merge(
        test_combined_results,
        on=merge_keys,
        suffixes=("_train", "_test"),
        how="outer",
        indicator=True
    )
    
    # Clear fill of selected columns: label missing selection status explicitly
    merged_results["selected_train"] = merged_results["selected_train"].fillna("not_performed")
    merged_results["selected_test"] = merged_results["selected_test"].fillna("not_performed")
    
    # Optional: fill numeric columns only
    numeric_cols = [col for col in merged_results.columns if merged_results[col].dtype.kind in "biufc"]
    merged_results[numeric_cols] = merged_results[numeric_cols].fillna(0)
    
    # Optional: sort for reproducibility
    merged_results = merged_results.sort_values(by=merge_keys).reset_index(drop=True)

    # Log results
    train_test_log = create_result_summary_df(merged_results)
    logger.log_step(
        step_name="train/test split",
        info={
            "split_method": split_method,
            "split_date": split_date,
            "split_pct": split_pct,
            "date_col": date_col,
            "use_metrics": use_metrics,
            "mine_rules": mine_rules,
        },
        df=train_test_log
    )
    return merged_results, train_test_log

def train_test_main(
    prepared_df: pd.DataFrame, 
    cfg: Config,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    func = run_train_test
    func_args = {
        "cfg": cfg, 
        "split_method": cfg.split_method, 
        "split_date": cfg.split_date, 
        "split_pct": cfg.split_pct, 
        "date_col": cfg.date_col, 
        "use_metrics": cfg.use_metrics, 
        "logger": logger
    }
    
    train_test_res, train_test_log = evaluator_toggled(
        prepared_df, 
        cfg, 
        func, 
        func_args, 
        cfg.discovery_mode, 
        logger, 
    )

    return train_test_res, train_test_log