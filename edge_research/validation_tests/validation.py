from typing import List, Optional, Tuple, Set, Sequence,Dict, Any, Optional, Union, Literal
import pandas as pd
import numpy as np
import warnings
import re
from statsmodels.stats.multitest import multipletests

from scripts.rules_mining.mining import (
    validate_parsed_rules, 
    generate_rule_activation_dataframe, 
    normalize_and_dedup_rules,
    compute_rule_depth, 
    merge_multivar_map_into_stats
)
from scripts.preprocessing.cleaning import clean_pipeline
from scripts.preprocessing.engineering import engineer_pipeline
from scripts.preprocessing.target import target_pipeline
from scripts.rules_mining.mining import (
    data_prep_pipeline, 
    mining_pipeline
)
from scripts.statistics.calculator import generate_statistics

# --- Train test / WFA --- #
def parse_date_ranges(date_ranges):
    """Convert list of lists (or list of tuples) of date strings to list of (pd.Timestamp, pd.Timestamp) tuples."""
    if not isinstance(date_ranges, list):
        raise ValueError("date_ranges must be a list")
    result = []
    for rng in date_ranges:
        # Accept tuple or list of two elements
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise ValueError(f"Each date range must be a list or tuple of two date strings: got {rng}")
        start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
        result.append((start, end))
    return result

def generate_time_splits(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int,
    date_ranges: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Slice a DataFrame into N chronological splits or user-specified date windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a datetime column.
    date_col : str
        Name of the datetime column in df.
    n_splits : int
        Number of splits to generate (if date_ranges not provided).
    date_ranges : list of (start, end) pd.Timestamp tuples, optional
        Explicit date windows to use for splitting. If provided, splits will be created
        based on these windows and may overlap or leave gaps.

    Returns
    -------
    splits : List[pd.DataFrame]
        List of splits (ordered earliest to latest).
    log_df : pd.DataFrame
        DataFrame logging metadata for each split, including configured and actual
        start/end dates, number of rows, and relative size.

    Raises
    ------
    ValueError
        If n_splits < 1, or if any date_range is malformed (start >= end).
    KeyError
        If date_col is not found in the dataframe.
    """
    # --- Constants ---
    EMPTY_NOTE = "empty split"

    # --- Ensure correct format of date ranges ---
    date_ranges = parse_date_ranges(date_ranges)
    
    # --- Input validation ---
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in dataframe columns.")

    # --- Normalize and sort datetime column ---
    _df = df.copy()
    _df[date_col] = pd.to_datetime(_df[date_col], utc=True).dt.tz_localize(None)
    _df = _df.sort_values(date_col)
    total_rows = len(_df)

    splits: List[pd.DataFrame] = []
    log_records: List[dict] = []

    if date_ranges is not None:
        if len(date_ranges) != n_splits:
            warnings.warn(
                f"Number of date_ranges ({len(date_ranges)}) != n_splits ({n_splits}). Proceeding anyway."
            )
        split_items = []
        for i, (start, end) in enumerate(date_ranges):
            if start >= end:
                raise ValueError(f"Malformed date_range: {start} >= {end}.")
            mask = (_df[date_col] >= start) & (_df[date_col] < end)
            split_df = _df.loc[mask].copy()
            split_items.append((start, split_df, {
                "split_index": i,
                "split_type": "range",
                "configured_start": start,
                "configured_end": end,
                "start_date": split_df[date_col].min() if not split_df.empty else pd.NaT,
                "end_date": split_df[date_col].max() if not split_df.empty else pd.NaT,
                "n_rows": len(split_df),
                "fraction_of_total": len(split_df) / total_rows if total_rows > 0 else 0.0,
                "note": EMPTY_NOTE if split_df.empty else "",
            }))
        # Sort splits and logs by window start time for reproducibility
        split_items.sort(key=lambda x: x[0])
        splits = [item[1] for item in split_items]
        log_records = [item[2] for item in split_items]
    else:
        min_date = _df[date_col].min()
        max_date = _df[date_col].max()
        date_bins = pd.date_range(start=min_date, end=max_date, periods=n_splits + 1)
        for i in range(n_splits):
            start = date_bins[i]
            end = date_bins[i + 1]
            is_last = (i == n_splits - 1)
            mask = (_df[date_col] >= start) & (
                _df[date_col] < end if not is_last else _df[date_col] <= end
            )
            split_df = _df.loc[mask].copy()
            splits.append(split_df)
            log_records.append({
                "split_index": i,
                "split_type": "equal",
                "configured_start": start,
                "configured_end": end,
                "start_date": split_df[date_col].min() if not split_df.empty else pd.NaT,
                "end_date": split_df[date_col].max() if not split_df.empty else pd.NaT,
                "n_rows": len(split_df),
                "fraction_of_total": len(split_df) / total_rows if total_rows > 0 else 0.0,
                "note": EMPTY_NOTE if split_df.empty else "",
            })

    log_df = pd.DataFrame(log_records)
    return splits, log_df

def generate_fraction_splits(
    df: pd.DataFrame,
    date_col: str,
    n_splits: Optional[int] = None,
    window_frac: Optional[float] = None,
    step_frac: Optional[float] = None,
    fractions: Optional[List[float]] = None,
    overlap: bool = False,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Partition a DataFrame into one or more chronologically ordered windows,
    each defined by a fraction of the dataset's total date span, with optional overlap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime column.
    date_col : str
        Name of the datetime column to use for splitting.
    n_splits : int, optional
        Number of equal-length, non-overlapping windows (implies window_frac=1/n_splits).
    window_frac : float, optional
        Length of each window as a fraction of the date span (used with step_frac).
    step_frac : float, optional
        Step size between window starts, as a fraction of the date span.
    fractions : List[float], optional
        Explicit list of window lengths (fractions of total date span).
    overlap : bool, default=False
        If True, allows overlapping windows.

    Returns
    -------
    splits : List[pd.DataFrame]
        List of split DataFrames, each covering the specified window.
    log_df : pd.DataFrame
        DataFrame logging configuration and content of each split.

    Raises
    ------
    ValueError
        If configuration is invalid or unsupported.
    KeyError
        If date_col not present in df.

    Notes
    -----
    - Windows are defined by their (start, end) time and may overlap if requested.
    - Returned splits are always in ascending order of window start.
    - Does not mutate the input DataFrame.
    """
    # --- Constants ---
    FLOAT_TOL = 1e-8
    EMPTY_NOTE = "empty split"

    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in dataframe columns.")

    _df = df.copy()
    _df[date_col] = pd.to_datetime(_df[date_col], utc=True).dt.tz_localize(None)
    _df = _df.sort_values(date_col)
    total_rows = len(_df)

    min_date = _df[date_col].min()
    max_date = _df[date_col].max()
    if pd.isnull(min_date) or pd.isnull(max_date):
        raise ValueError("Date column is empty or not parseable.")
    span_seconds = (max_date - min_date).total_seconds()

    splits: List[pd.DataFrame] = []
    log_records: List[dict] = []

    # --- Window configuration resolution ---
    if fractions is not None:
        if not isinstance(fractions, list) or not all(isinstance(f, float) for f in fractions):
            raise ValueError("fractions must be a list of floats.")
        if any(f <= 0 or f > 1 for f in fractions):
            raise ValueError("All fractions must be in (0, 1].")
        if not overlap:
            total_frac = sum(fractions)
            if total_frac > 1.0 + FLOAT_TOL:
                raise ValueError(f"Sum of fractions ({total_frac}) > 1 with overlap=False.")
        n = len(fractions)
        starts, ends = [], []
        # Non-overlapping: next window starts where previous ends
        if not overlap:
            start_pct = 0.0
            for f in fractions:
                starts.append(start_pct)
                ends.append(start_pct + f)
                start_pct += f
        else:
            # Overlapping: each window starts after previous (can overlap)
            start_pct = 0.0
            for f in fractions:
                starts.append(start_pct)
                ends.append(start_pct + f)
                start_pct += f
    elif n_splits is not None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        window_frac = 1.0 / n_splits
        step_frac = window_frac if not overlap else window_frac / 2
        starts = [i * step_frac for i in range(n_splits)]
        ends = [s + window_frac for s in starts]
        n = n_splits
    elif window_frac is not None and step_frac is not None:
        if not (0 < window_frac <= 1):
            raise ValueError("window_frac must be in (0, 1].")
        if not (0 < step_frac <= 1):
            raise ValueError("step_frac must be in (0, 1].")
        starts, ends = [], []
        curr = 0.0
        n = 0
        while curr + window_frac <= 1.0 + FLOAT_TOL:
            starts.append(curr)
            ends.append(curr + window_frac)
            n += 1
            curr += step_frac
        if n == 0:
            raise ValueError("No splits produced. Check window_frac and step_frac.")
    else:
        raise ValueError("Must provide one of: fractions, n_splits, or (window_frac and step_frac).")

    # --- Create splits and logs ---
    for i in range(n):
        s_frac = starts[i]
        e_frac = min(ends[i], 1.0)
        s_time = min_date + pd.Timedelta(seconds=s_frac * span_seconds)
        e_time = min_date + pd.Timedelta(seconds=e_frac * span_seconds)
        mask = (_df[date_col] >= s_time) & (_df[date_col] < e_time if i < n - 1 else _df[date_col] <= e_time)
        split_df = _df.loc[mask].copy()
        splits.append(split_df)
        log_records.append({
            "split_index": i,
            "configured_start": s_time,
            "configured_end": e_time,
            "start_date": split_df[date_col].min() if not split_df.empty else pd.NaT,
            "end_date": split_df[date_col].max() if not split_df.empty else pd.NaT,
            "n_rows": len(split_df),
            "fraction_of_total": len(split_df) / total_rows if total_rows > 0 else 0.0,
            "note": EMPTY_NOTE if split_df.empty else "",
        })

    log_df = pd.DataFrame(log_records)
    return splits, log_df


def split_datasets(
    df: pd.DataFrame,
    hloc: pd.DataFrame,
    cfg: Any,
    date_col: str,
    logger: Optional[Any] = None,
    splits: Optional[int] = None,
    ranges: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    method: Optional[str] = None,
    window_frac: Optional[float] = None,
    step_frac: Optional[float] = None,
    fractions: Optional[List[float]] = None,
    overlap: Optional[bool] = None,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Split a DataFrame into train/test sets using either temporal or fractional logic.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    date_col : str
        Name of the datetime column used for ordering and filtering.
    logger : Optional[Any], default=None
        Optional logger instance for structured logging.
    splits : int, optional
        Number of splits to generate.
    ranges : list of tuple, optional
        Explicit (start, end) timestamps for each split (used in temporal mode).
    method : {'temporal', 'fractional'}, optional
        Splitting strategy. 'temporal' uses fixed date windows, 'fractional' uses fractions of time span.
    window_frac : float, optional
        Length of each window as a fraction of the total time span (used in fractional mode).
    step_frac : float, optional
        Step size between window starts, as a fraction of the total time span (used in fractional mode).
    fractions : list of float, optional
        Explicit list of window lengths (fractions of total time span, used in fractional mode).
    overlap : bool, optional
        Whether to allow overlapping windows (fractional mode only).

    Returns
    -------
    splits : List[pd.DataFrame]
        List of DataFrames, one per split.
    split_log : pd.DataFrame
        Summary log with metadata about each split.

    Raises
    ------
    ValueError
        If method is invalid or required parameters are missing.
    """
    valid_methods = {"fractional", "temporal"}

    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}.")

    if method == "temporal":
        splits, split_log = generate_time_splits(
            df=df,
            date_col=date_col,
            n_splits=splits,
            date_ranges=ranges,
        )
    else:  # method == "fractional"
        splits, split_log = generate_fraction_splits(
            df=df,
            date_col=date_col,
            n_splits=splits,
            window_frac=window_frac,
            step_frac=step_frac,
            fractions=fractions,
            overlap=overlap or False,
        )

    cleaned_splits = []

    # Perform all data cleaning steps for each split after splitting
    for idx, split_df in enumerate(splits):
        
        df, cleaning_logs = clean_pipeline(split_df, cfg, logger)        
        df, engineering_logs = engineer_pipeline(df, cfg, logger)    
        df, target_logs = target_pipeline(df, cfg, hloc, logger)        
        prepped_df, prep_log = data_prep_pipeline(df, cfg, logger)    
        cleaned_splits.append(prepped_df)
    
    # Align all dataframes to have only the common columns
    common_cols = set.intersection(*(set(df.columns) for df in cleaned_splits))
    cleaned_splits = [df.loc[:, sorted(common_cols)] for df in cleaned_splits]

    return cleaned_splits, split_log

def rules_series_to_unique_rules(
    rule_series: pd.Series,
    provenance: bool = False
) -> List[Tuple[List[Tuple[str, int]], Set[str]]]:
    """
    Parse a Series of boolean rule strings into a structured format.

    Each rule string is expected to follow the format:
        "('feature1' == 0) AND ('feature2' == 1)"
    and will be parsed into a list of (feature, value) pairs.

    Parameters
    ----------
    rule_series : pd.Series
        Series of strings, each representing a rule as a conjunction of feature==0/1 clauses.
    provenance : bool, default=False
        If True, include the original rule string as a provenance set alongside each rule.

    Returns
    -------
    List[Tuple[List[Tuple[str, int]], Set[str]]]
        List of parsed rules. Each rule is a tuple of:
        - A list of (feature_name, value) conditions (in order),
        - A provenance set (empty if provenance=False).

    Raises
    ------
    ValueError
        If any rule clause does not match the expected format.
    """
    # Pattern matches clauses like: ('feature' == 0)
    CONDITION_PATTERN = re.compile(r"\(\s*'([^']+)'\s*==\s*([01])\s*\)")
    
    parsed_rules = []

    for idx, rule_str in rule_series.items():
        if not isinstance(rule_str, str) or not rule_str.strip():
            conditions = []
        else:
            parts = [part.strip() for part in rule_str.split("AND")]
            conditions = []
            for cond_str in parts:
                match = CONDITION_PATTERN.fullmatch(cond_str)
                if not match:
                    raise ValueError(
                        f"Malformed condition '{cond_str}' in rule {idx}: '{rule_str}'"
                    )
                feature, value = match.group(1), int(match.group(2))
                conditions.append((feature, value))
        
        provenance_set = {rule_str} if provenance else set()
        parsed_rules.append((conditions, provenance_set))
    
    return parsed_rules

def extract_rule_feature_names(series: Sequence[str]) -> List[str]:
    """
    Given a sequence (e.g., pandas Series) of single-order rule strings like
    'grossMargin_combined_trend=others == 1', return the feature name before ' == '.

    Args:
        series (Sequence[str]): Series or list of rule strings.

    Returns:
        List[str]: List of feature/column names with ' == ' and the trailing value removed.

    Raises:
        ValueError: If any string does not contain ' == '.
    """
    feature_names = []
    for idx, rule_str in enumerate(series):
        if not isinstance(rule_str, str) or '==' not in rule_str:
            raise ValueError(
                f"Rule at index {idx} is not a valid string with '==': {rule_str!r}"
            )
        # Split by '==', take the part before, strip any trailing spaces
        feature = rule_str.split('==')[0].rstrip()
        feature_names.append(feature)
    return feature_names


def make_combined_rule_feature_df(
    train_mining_res: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a combined rule-activation feature matrix for the test set, incorporating both
    single-variate and multivariate rules.

    Parameters
    ----------
    train_mining_res : pd.DataFrame
        Output of the mining pipeline. Must contain columns: 'antecedents', 'rule_depth', and 'selected'.
    test_df : pd.DataFrame
        Full test dataset on which rule activations are computed.
    target_col : str
        Name of the target column. Must be present in `test_df`.

    Returns
    -------
    combined_df : pd.DataFrame
        DataFrame where each column represents a rule (either original feature or generated
        rule mask), and includes the target column.
    multivar_map : pd.DataFrame
        Mapping of multivariate rule column names to their human-readable conditions.

    Raises
    ------
    KeyError
        If required columns are missing from inputs.
    ValueError
        If `target_col` is missing from test_df.
    """
    # --- Validate inputs ---
    required_cols = {"antecedents", "rule_depth", "selected"}
    if not required_cols.issubset(train_mining_res.columns):
        missing = required_cols - set(train_mining_res.columns)
        raise KeyError(f"train_mining_res is missing columns: {missing}")
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test_df.")

    # --- Extract selected multivariate rules ---
    multivar_rules_raw = train_mining_res.loc[
        (train_mining_res["rule_depth"] > 1) & (train_mining_res["selected"]),
        "antecedents"
    ].drop_duplicates()
    multivar_rules = rules_series_to_unique_rules(multivar_rules_raw) if not multivar_rules_raw.empty else []

    # --- Extract selected single-variate rule feature names ---
    singlevar_rules_raw = train_mining_res.loc[
        (train_mining_res["rule_depth"] == 1) & (train_mining_res["selected"]),
        "antecedents"
    ].drop_duplicates()
    singlevar_rules = extract_rule_feature_names(singlevar_rules_raw) if not singlevar_rules_raw.empty else []

    # --- Generate rule-activation DataFrames ---
    multivar_df, multivar_map = (
        generate_rule_activation_dataframe(test_df, multivar_rules, target_col)
        if multivar_rules else (pd.DataFrame(index=test_df.index), pd.DataFrame())
    )

    singlevar_df = (
        test_df[singlevar_rules] if singlevar_rules else pd.DataFrame(index=test_df.index)
    )

    # --- Combine rule features and handle target column ---
    combined_parts = []

    if not singlevar_df.empty:
        combined_parts.append(singlevar_df)

    if not multivar_df.empty:
        # Avoid target_col duplication
        if target_col in singlevar_df.columns and target_col in multivar_df.columns:
            multivar_df = multivar_df.drop(columns=[target_col])
        combined_parts.append(multivar_df)

    if not combined_parts or target_col not in pd.concat(combined_parts, axis=1).columns:
        # Add target column if not already present
        combined_parts.append(test_df[[target_col]])

    combined_df = pd.concat(combined_parts, axis=1)

    return combined_df, multivar_map


def test_mined_rules(
    train_mining_res: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate mined rules on a test dataset and return rule-level performance statistics.

    This function:
    - Applies mined rules (both single- and multivariate) to the test set
    - Computes statistical metrics for each rule-based feature
    - Maps internal rule IDs to human-readable expressions
    - Calculates rule depth (number of conditions per rule)
    - Normalizes and renames columns for consistent downstream consumption

    Parameters
    ----------
    train_mining_res : pd.DataFrame
        Mined rules with at least 'antecedents', 'rule_depth', and 'selected' columns.
    test_df : pd.DataFrame
        Full test dataset (must include target_col).
    cfg : "params.config_validator.Config"
        Configuration dictionary for `generate_statistics`.
    target_col : str
        Name of the target column in `test_df`.

    Returns
    -------
    test_stats : pd.DataFrame
        Rule-level statistics with readable antecedents and depth information.
    test_stats_log : pd.DataFrame
        Diagnostic log or metadata from the statistics computation step.

    Raises
    ------
    KeyError
        If required columns are missing in inputs.
    """
    # Apply rules to generate test features
    test_df_prepped, multivar_map = make_combined_rule_feature_df(
        train_mining_res, test_df, target_col
    )

    # Compute rule statistics
    test_stats, test_stats_log = generate_statistics(test_df_prepped, cfg)

    # Annotate rules with metadata
    multivar_map["rule_depth"] = multivar_map["human_readable_rule"].apply(compute_rule_depth)
    test_stats = merge_multivar_map_into_stats(test_stats, multivar_map)

    # Fill in missing metadata for single-variate rules
    test_stats["rule_depth"] = test_stats["rule_depth"].fillna(1)
    test_stats["human_readable_rule"] = test_stats["human_readable_rule"].fillna(test_stats["antecedents"])

    # Normalize output column names
    test_stats = (
        test_stats
        .drop(columns=["antecedents", "rule_column"], errors="ignore")
        .rename(columns={"human_readable_rule": "antecedents"})
    )

    return test_stats, test_stats_log


def split_mining_pipeline(
    splits: List[pd.DataFrame],
    cfg: "params.config_validator.Config",
    re_mine: bool,
    target_col: str,
    logger: Optional[Any] = None,
) -> Tuple[List[pd.DataFrame], List[int], List[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Run rule mining or testing across multiple dataset splits.

    This function supports both:
    - Train/test-style evaluation (mine once, test on future splits), or
    - Walk-forward evaluation (re-mine rules independently per split).

    Parameters
    ----------
    splits : List[pd.DataFrame]
        Chronologically ordered list of dataset splits. Must have length >= 2.
    cfg : params.config_validator.Config
        Configuration object passed to mining and testing pipelines.
    re_mine : bool
        If True, re-mine rules on every split. If False, mine on the first and test on the rest.
    target_col : str
        Name of the target column for use in statistics computation.
    logger : Optional[Any], default=None
        Optional logger instance supporting `.log_step(...)`. Passed into all pipeline steps.

    Returns
    -------
    results : List[pd.DataFrame]
        Per-split DataFrames of statistics or mined rules, depending on `re_mine`.
    rule_counts : List[int]
        Number of rules mined or tested per split.
    logs : List[Dict[str, Any]]
        Per-split logs including preprocessing, mining, or testing metadata.
    initial_rules : Optional[pd.DataFrame]
        Rules mined from the first split, if `re_mine` is False. Else, returns None.

    Raises
    ------
    ValueError
        If `splits` is not a list of at least 2 DataFrames.
    TypeError
        If any element of `splits` is not a DataFrame.
    """
    if not isinstance(splits, list) or len(splits) < 2:
        raise ValueError("`splits` must be a list with at least 2 pd.DataFrame elements.")
    if any(not isinstance(df, pd.DataFrame) for df in splits):
        raise TypeError("All elements in `splits` must be of type pd.DataFrame.")

    results: List[pd.DataFrame] = []
    rule_counts: List[int] = []
    logs: List[Dict[str, Any]] = []
    initial_rules: Optional[pd.DataFrame] = None

    for idx, split_df in enumerate(splits):
        split_log = {}
        
        # Mining or Testing
        if idx == 0:
            mined_rules, rule_count, mine_log = mining_pipeline(split_df, cfg, logger)
            results.append(mined_rules)
            rule_counts.append(rule_count)
            split_log["mine_log"] = mine_log
            if not re_mine:
                initial_rules = mined_rules
        else:
            if re_mine:
                mined_rules, rule_count, mine_log = mining_pipeline(split_df, cfg, logger)
                results.append(mined_rules)
                rule_counts.append(rule_count)
                split_log["mine_log"] = mine_log
            else:
                if initial_rules is None:
                    raise RuntimeError("Expected initial_rules to be set when re_mine is False.")
                test_stats, test_log = test_mined_rules(initial_rules, split_df, cfg, target_col)
                results.append(test_stats)
                rule_counts.append(len(initial_rules))
                split_log["test_log"] = test_log

        logs.append(split_log)

    return results, rule_counts, logs, None if re_mine else initial_rules

def combine_split_results(
    results: List[pd.DataFrame],
    split_prefixes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merge rule-level statistics from multiple split evaluation results into a single aligned DataFrame.

    Each input DataFrame must contain at minimum: 'antecedents' and 'consequents'.
    Optionally, a 'rule_depth' column will also be included in the index if present in any split.

    Parameters
    ----------
    results : List[pd.DataFrame]
        List of rule statistics DataFrames (≥2), one per split. Each must contain rule identifiers and metrics.
    split_prefixes : Optional[List[str]], default=None
        List of prefixes for each split (e.g., ["train", "val"]). If not provided, defaults to ["split_0", ..., "split_N"].

    Returns
    -------
    combined_df : pd.DataFrame
        Merged DataFrame indexed by rule identifier columns (antecedents, consequents, [rule_depth]),
        with split-specific metric columns prefixed accordingly.

    Raises
    ------
    ValueError
        If input list is too short or lacks required columns.
    TypeError
        If input is not a list of DataFrames.
    """
    # Validate inputs
    if not isinstance(results, list) or len(results) < 2:
        raise ValueError("`results` must be a list of at least 2 DataFrames.")
    if any(not isinstance(df, pd.DataFrame) for df in results):
        raise TypeError("All elements in `results` must be pandas DataFrames.")

    if split_prefixes is not None:
        if len(split_prefixes) != len(results):
            raise ValueError("Length of `split_prefixes` must match number of splits.")
    else:
        split_prefixes = [f"split_{i}" for i in range(len(results))]

    # Identify identifier columns
    id_cols = ["antecedents", "consequents"]
    if any("rule_depth" in df.columns for df in results):
        id_cols.append("rule_depth")

    # Confirm all required ID columns are present
    for df in results:
        if not {"antecedents", "consequents"}.issubset(df.columns):
            raise ValueError("Each result must include 'antecedents' and 'consequents' columns.")

    # Gather all possible metrics across splits (excluding identifier columns)
    all_metric_cols = set()
    for df in results:
        all_metric_cols.update(set(df.columns) - set(id_cols))

    # Build prefix-annotated splits
    indexed_splits = []
    for df, prefix in zip(results, split_prefixes):
        id_cols_present = [col for col in id_cols if col in df.columns]
        metric_cols_present = [col for col in all_metric_cols if col in df.columns]
        slim_df = df[id_cols_present + metric_cols_present].copy()
        slim_df = slim_df.set_index(id_cols_present)
        slim_df.columns = [f"{prefix}_{col}" for col in slim_df.columns]
        indexed_splits.append(slim_df)

    # Outer join on all rule identifiers
    combined_df = indexed_splits[0]
    for split_df in indexed_splits[1:]:
        combined_df = combined_df.join(split_df, how="outer")

    return combined_df.reset_index()


def ensure_bool(series: pd.Series) -> pd.Series:
    """
    Normalize a Series to clean boolean values, avoiding fillna warnings.

    Parameters
    ----------
    series : pd.Series
        Input Series containing boolean-like data.

    Returns
    -------
    pd.Series
        Boolean Series with missing and ambiguous values coerced safely.
    """
    s = series.fillna(False).infer_objects(copy=False)

    if s.dtype != bool:
        truth_map = {"true": True, "1": True, "false": False, "0": False}
        s = (
            s.astype(str)
            .str.strip()
            .str.lower()
            .map(truth_map)
            .fillna(False)
        )

    return s.astype(bool)


def create_validation_log_df(
    train_test_results: pd.DataFrame,
    splits: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a one-row summary DataFrame of rule-level validation metrics across splits.

    Parameters
    ----------
    train_test_results : pd.DataFrame
        Combined DataFrame from multiple validation splits (from `combine_split_results`),
        with columns like 'split_0_lift', 'split_1_selected', etc.
    splits : list of str, optional
        List of split name prefixes. If None, inferred from column names like 'split_0_lift'.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame containing counts, mean/median metrics, and overlap statistics.

    Raises
    ------
    ValueError
        If input DataFrame is empty.
    """
    if train_test_results.empty:
        raise ValueError("Input train_test_results is empty.")

    # Set option to avoid silent downcasting warnings
    pd.set_option('future.no_silent_downcasting', True)

    # --- 1. Infer split prefixes if not provided ---
    if splits is None:
        splits = sorted({
            col.split("_")[0]
            for col in train_test_results.columns
            if col.startswith("split_")
        })

    log: dict[str, Union[int, float]] = {}

    # --- 2. Per-split metrics ---
    for split in splits:
        lift_col = f"{split}_lift"
        selected_col = f"{split}_selected"

        split_mask = (
            train_test_results[lift_col].notna()
            if lift_col in train_test_results.columns
            else pd.Series(False, index=train_test_results.index)
        )
        log[f"n_rules_{split}"] = split_mask.sum()

        if selected_col in train_test_results.columns:
            selected = ensure_bool(train_test_results[selected_col])
            log[f"n_selected_{split}"] = (split_mask & selected).sum()
        else:
            log[f"n_selected_{split}"] = np.nan

        for metric in ("lift", "confidence", "observations"):
            col = f"{split}_{metric}"
            if col in train_test_results.columns:
                values = train_test_results.loc[split_mask, col]
                log[f"mean_{metric}_{split}"] = values.mean()
                log[f"median_{metric}_{split}"] = values.median()
            else:
                log[f"mean_{metric}_{split}"] = np.nan
                log[f"median_{metric}_{split}"] = np.nan

    # --- 3. Overlap metrics between first two splits ---
    if len(splits) >= 2:
        s0, s1 = splits[0], splits[1]

        mask0 = (
            train_test_results[f"{s0}_lift"].notna()
            if f"{s0}_lift" in train_test_results.columns
            else pd.Series(False, index=train_test_results.index)
        )
        mask1 = (
            train_test_results[f"{s1}_lift"].notna()
            if f"{s1}_lift" in train_test_results.columns
            else pd.Series(False, index=train_test_results.index)
        )
        overlap_mask = mask0 & mask1
        log["n_overlap_rules"] = overlap_mask.sum()

        sel0 = ensure_bool(train_test_results.get(f"{s0}_selected", pd.Series(False, index=train_test_results.index)))
        sel1 = ensure_bool(train_test_results.get(f"{s1}_selected", pd.Series(False, index=train_test_results.index)))
        log["n_overlap_selected"] = (overlap_mask & sel0 & sel1).sum()

        for metric in ("lift", "confidence"):
            c0 = train_test_results.get(f"{s0}_{metric}")
            c1 = train_test_results.get(f"{s1}_{metric}")
            if c0 is not None and c1 is not None:
                v0 = c0[overlap_mask]
                v1 = c1[overlap_mask]
                log[f"mean_{metric}_overlap_{s0}"] = v0.mean()
                log[f"mean_{metric}_overlap_{s1}"] = v1.mean()
                log[f"mean_delta_{metric}_overlap"] = (v1 - v0).mean()
            else:
                log[f"mean_{metric}_overlap_{s0}"] = np.nan
                log[f"mean_{metric}_overlap_{s1}"] = np.nan
                log[f"mean_delta_{metric}_overlap"] = np.nan

    return pd.DataFrame([log])


def validate_train_test(
    df: pd.DataFrame,
    hloc: pd.DataFrame,
    cfg: Any,
    target_col: str,
    date_col: str,
    train_test_splits: int,
    train_test_ranges: Union[List[List[str]], List[Tuple[str, str]]],
    train_test_split_method: str,
    train_test_window_frac: float,
    train_test_step_frac: float,
    train_test_fractions: List[float],
    train_test_overlap: bool,
    train_test_re_mine: bool,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a complete train/test rule mining validation procedure over a DataFrame.

    This function:
    - Splits the dataset into multiple chronological windows.
    - Applies the mining and validation pipeline on each split.
    - Merges per-split results into a combined evaluation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The full input dataset to split and validate on.
    cfg : Any
        Configuration object passed to the mining and statistic pipelines.
    target_col : str
        Column name of the binary/multiclass prediction target.
    date_col : str
        Column name representing chronological order (must be sortable).
    train_test_splits : int
        Number of train/test splits to create.
    train_test_ranges : list of [str, str] or list of tuple
        Optional manual overrides for start/end of each split.
    train_test_split_method : str
        Splitting strategy. 'temporal' uses fixed date windows, 'fractional' uses fractions of time span.
    train_test_window_frac : float
        Fraction of data used per training window.
    train_test_step_frac : float
        Step size as a fraction of total data when moving to the next window.
    train_test_fractions : list of float
        Relative sizes for train/test allocation within each split.
    train_test_overlap : bool
        Whether train/test windows can overlap chronologically.
    train_test_re_mine : bool
        If True, re-mine rules on each split; if False, mine once and test forward.
    logger : optional
        Optional logger object that supports `.log_step(...)`.

    Returns
    -------
    combined_results : pd.DataFrame
        Merged rule performance metrics across all splits.
    metadata : dict
        Contains:
            - "train_test_rule_counts": List of rule counts per split
            - "train_test_logs": List of per-split logs
            - "train_test_initial_rules": Rules mined on first split (if `re_mine` is False)

    Raises
    ------
    ValueError
        If input validation fails in any pipeline step.
    """
    
    # --- 1. Split the dataset ---
    splits, split_log = split_datasets(
        df=df,
        hloc=hloc,
        cfg=cfg,
        logger=logger,
        date_col=date_col,
        splits=train_test_splits,
        ranges=train_test_ranges,
        method=train_test_split_method,
        window_frac=train_test_window_frac,
        step_frac=train_test_step_frac,
        fractions=train_test_fractions,
        overlap=train_test_overlap,
    )

    # --- 2. Apply mining or test pipeline to each split ---
    train_test_results, rule_counts, logs, initial_rules = split_mining_pipeline(
        splits=splits,
        cfg=cfg,
        re_mine=train_test_re_mine,
        target_col=target_col,
        logger=logger,
    )

    # --- 3. Combine all per-split results into one matrix ---
    combined_results = combine_split_results(train_test_results)

    return combined_results, {
        "train_test_rule_counts": rule_counts,
        "train_test_logs": logs,
        "train_test_initial_rules": initial_rules,
    }


def validate_wfa(
    df: pd.DataFrame,
    hloc: pd.DataFrame,
    cfg: Any,
    target_col: str,
    date_col: str,
    wfa_splits: int,
    wfa_ranges: Union[List[List[str]], List[Tuple[str, str]]],
    wfa_split_method: str,
    wfa_window_frac: float,
    wfa_step_frac: float,
    wfa_fractions: List[float],
    wfa_overlap: bool,
    wfa_re_mine: bool,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a complete Walk Forward Analysis (WFA) validation using a rule mining pipeline.

    This function:
    - Splits the dataset into chronological walk-forward windows
    - Applies mining and testing per window
    - Aggregates per-window statistics into a combined results matrix

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to evaluate over WFA windows.
    cfg : Any
        Configuration object for pipeline steps (passed into mining, testing, etc.).
    target_col : str
        Column name for target variable used in rule evaluation.
    date_col : str
        Column representing temporal ordering (must be sortable).
    wfa_splits : int
        Number of WFA windows/splits to generate.
    wfa_ranges : List[List[str]] or List[Tuple[str, str]]
        Optional manual split boundaries (start/end per window).
    wfa_split_method : str
        Splitting strategy. 'temporal' uses fixed date windows, 'fractional' uses fractions of time span.
    wfa_window_frac : float
        Fraction of data to include in each training window.
    wfa_step_frac : float
        Step size (fraction of dataset) between WFA splits.
    wfa_fractions : List[float]
        Relative train/test sizes per split, e.g., [0.7, 0.3].
    wfa_overlap : bool
        Whether training/test windows may overlap chronologically.
    wfa_re_mine : bool
        If True, re-mine rules on each WFA window; otherwise, reuse first split’s rules.
    logger : Optional[Any], default=None
        Optional logger instance supporting `.log_step(...)`.

    Returns
    -------
    combined_results : pd.DataFrame
        Aggregated rule-level statistics across all WFA splits.
    metadata : Dict[str, Any]
        Dictionary containing:
            - "wfa_rule_counts": Number of rules per split
            - "wfa_logs": Per-split processing logs
            - "wfa_initial_rules": Rules from the first split (if not re-mining)

    Raises
    ------
    ValueError
        If input validation fails in underlying pipeline components.
    """
    splits, split_log = split_datasets(
        df=df,
        hloc=hloc,
        cfg=cfg,
        logger=logger,
        date_col=date_col,
        splits=wfa_splits,
        ranges=wfa_ranges,
        method=wfa_split_method,
        window_frac=wfa_window_frac,
        step_frac=wfa_step_frac,
        fractions=wfa_fractions,
        overlap=wfa_overlap,
    )

    wfa_results, wfa_rule_counts, wfa_logs, wfa_initial_rules = split_mining_pipeline(
        splits=splits,
        cfg=cfg,
        re_mine=wfa_re_mine,
        target_col=target_col,
        logger=logger,
    )

    combined_results = combine_split_results(wfa_results)

    return combined_results, {
        "wfa_rule_counts": wfa_rule_counts,
        "wfa_logs": wfa_logs,
        "wfa_initial_rules": wfa_initial_rules,
    }

def train_test_pipeline(
    df: pd.DataFrame,
    hloc: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[pd.DataFrame]]]:
    """
    Execute the full train/test validation workflow with optional config overrides.

    This function:
    - Extracts train/test split parameters from a config object and override dict
    - Runs the train/test validation via `validate_train_test`
    - Computes a summary log DataFrame using `create_validation_log_df`
    - Optionally logs the step using a provided logger

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset for rule mining validation.
    cfg : Any
        Configuration object from which parameters are resolved.
    logger : Optional[Any], default=None
        Optional logger instance supporting `.log_step(...)`.
    **overrides : dict
        Optional parameter overrides that supersede config defaults.

    Returns
    -------
    train_test_results : pd.DataFrame
        Combined per-rule performance statistics from all splits.
    train_test_log : pd.DataFrame
        One-row summary log of split-level metrics and overlap stats.
    pipeline_logs : Dict[str, List[pd.DataFrame]]
        Metadata including rule counts, logs, and initial rules.

    Raises
    ------
    ValueError
        If required config values are missing or validation fails.
    """

    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    train_test_kwargs = {
        "date_col": param("date_col"),
        "target_col": param("target_col"),
        "train_test_splits": param("train_test_splits"),
        "train_test_ranges": param("train_test_ranges"),
        "train_test_split_method": param("train_test_split_method"),
        "train_test_window_frac": param("train_test_window_frac"),
        "train_test_step_frac": param("train_test_step_frac"),
        "train_test_fractions": param("train_test_fractions"),
        "train_test_overlap": param("train_test_overlap"),
        "train_test_re_mine": param("train_test_re_mine"),
    }

    train_test_results, pipeline_logs = validate_train_test(
        df=df,
        hloc=hloc,
        cfg=cfg,
        logger=logger,
        **train_test_kwargs,
    )

    train_test_log = create_validation_log_df(
        train_test_results, splits=["split_0", "split_1"]
    )

    if logger:
        logger.log_step(
            step_name="Train / Test split validation test",
            info=train_test_kwargs,
            df=train_test_log,
            max_rows=param("log_max_rows"),
        )

    return train_test_results, train_test_log, pipeline_logs

def wfa_pipeline(
    df: pd.DataFrame,
    hloc: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[pd.DataFrame]]]:
    """
    Execute a full Walk Forward Analysis (WFA) validation pipeline with optional config overrides.

    This function:
    - Extracts WFA parameters from a config object and override dictionary
    - Runs the WFA rule mining and evaluation via `validate_wfa`
    - Computes a summary log DataFrame using `create_validation_log_df`
    - Optionally logs the step using the provided logger

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to be evaluated using WFA.
    cfg : Any
        Configuration object containing WFA parameters.
    logger : Optional[Any], default=None
        Logger instance supporting `.log_step(...)`.
    **overrides : dict
        Keyword arguments that override config parameters.

    Returns
    -------
    wfa_results : pd.DataFrame
        Combined rule-level performance metrics across all WFA splits.
    wfa_log : pd.DataFrame
        One-row summary log of split metrics and overlap analysis.
    pipeline_logs : Dict[str, List[pd.DataFrame]]
        Contains:
            - "wfa_rule_counts": Number of rules per split
            - "wfa_logs": Logs per split
            - "wfa_initial_rules": Rules from first split (if not re-mining)

    Raises
    ------
    ValueError
        If required config values are missing or validation fails.
    """

    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    wfa_kwargs = {
        "date_col": param("date_col"),
        "target_col": param("target_col"),
        "wfa_split_method": param("wfa_split_method"),
        "wfa_splits": param("wfa_splits"),
        "wfa_ranges": param("wfa_ranges"),
        "wfa_window_frac": param("wfa_window_frac"),
        "wfa_step_frac": param("wfa_step_frac"),
        "wfa_fractions": param("wfa_fractions"),
        "wfa_overlap": param("wfa_overlap"),
        "wfa_re_mine": param("wfa_re_mine"),
    }

    log_max_rows = param("log_max_rows")

    wfa_results, pipeline_logs = validate_wfa(df, hloc, cfg, logger=logger, **wfa_kwargs)

    # Extract unique split prefixes like "split_0", "split_1", ...
    split_prefixes = sorted(
        {
            "_".join(col.split("_")[:2])
            for col in wfa_results.columns
            if col.startswith("split_")
        }
    )

    wfa_log = create_validation_log_df(wfa_results, splits=split_prefixes)

    if logger:
        logger.log_step(
            step_name="Walk Forward validation test",
            info=wfa_kwargs,
            df=wfa_log,
            max_rows=log_max_rows,
        )

    return wfa_results, wfa_log, pipeline_logs

# --- Bootstrap --- #
def resample_dataframe(
    df: pd.DataFrame,
    mode: str,
    block_size: int = 5,
    date_col: str = "date",
    id_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample a DataFrame using traditional or block bootstrap methods.

    Supports three modes:
    - 'traditional': i.i.d. row sampling with replacement.
    - 'block': block bootstrap over the full dataframe (ordered by date_col).
    - 'block_ids': block bootstrap applied separately per group defined by id_cols.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to be resampled.
    mode : str
        Resampling mode. One of {'traditional', 'block', 'block_ids'}.
    block_size : int, optional
        Number of consecutive rows per block for block bootstrap. Default is 5.
    date_col : str, optional
        Column name to use for sorting before block sampling. Default is 'date'.
    id_cols : list of str, optional
        List of columns to group by when mode is 'block_ids'. Required in that mode.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        Resampled dataframe, same approximate size as input.
    
    Raises
    ------
    ValueError
        If inputs are invalid or incompatible with the selected mode.
    """
    VALID_MODES = {"traditional", "block", "block_ids"}

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if mode == "block_ids" and (not id_cols):
        raise ValueError("id_cols must be provided for mode='block_ids'.")

    rng = np.random.default_rng(random_state)

    def block_sample(data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(date_col).reset_index(drop=True)
        n = len(data)
        if n < block_size:
            raise ValueError(f"block_size={block_size} exceeds data length {n}.")

        n_blocks = int(np.ceil(n / block_size))
        max_start = n - block_size
        start_indices = rng.integers(0, max_start + 1, size=n_blocks)

        blocks = [data.iloc[start : start + block_size] for start in start_indices]
        return pd.concat(blocks, ignore_index=True).iloc[:n].reset_index(drop=True)

    if mode == "traditional":
        indices = rng.integers(0, len(df), size=len(df))
        return df.iloc[indices].reset_index(drop=True)

    if mode == "block":
        return block_sample(df)

    # mode == 'block_ids'
    grouped = df.groupby(id_cols, group_keys=False, sort=False)
    return pd.concat([block_sample(group) for _, group in grouped], ignore_index=True)

def maybe_tqdm(iterable, use_tqdm: bool, **tqdm_kwargs):
    if use_tqdm:
        from tqdm.auto import tqdm
        return tqdm(iterable, **tqdm_kwargs)
    return iterable

def summarize_rule_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Aggregate performance metrics for each unique (antecedents, consequents) rule pair.

    For each specified metric, computes:
    - mean, std, min, max
    - 5th and 95th percentiles

    Also computes:
    - total number of selections (`selected_count`)
    - total number of test runs (`test_count`)
    - fraction of times the rule was selected (`selected_fraction`)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing rule evaluation results. Must include:
        - 'antecedents', 'consequents' columns
        - a 'selected' boolean column
        - all specified metric columns
    metrics : list of str
        List of metric column names to aggregate (e.g., ["lift", "confidence"]).

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with one row per rule (antecedent/consequent pair),
        including summary statistics for each metric and selection counts.
    
    Raises
    ------
    ValueError
        If required columns are missing from the input dataframe.
    """
    required_cols = {"antecedents", "consequents", "selected"} | set(metrics)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    QUANTILE_LOW = 0.05
    QUANTILE_HIGH = 0.95

    # Define aggregation functions
    metric_aggs = {
        col: ["mean", "std", "min", "max",
              lambda x: x.quantile(QUANTILE_LOW),
              lambda x: x.quantile(QUANTILE_HIGH)]
        for col in metrics
    }
    selection_aggs = {"selected": ["sum", "count"]}
    agg_funcs = {**metric_aggs, **selection_aggs}

    grouped = df.groupby(["antecedents", "consequents"]).agg(agg_funcs).reset_index()

    # Rename lambda outputs and flatten columns
    custom_names = {
        "<lambda_0>": "q05",
        "<lambda_1>": "q95",
        "sum": "selected_count",
        "count": "test_count"
    }
    grouped.columns = [
        f"{col}_{custom_names.get(func, func)}" if func else col
        for col, func in grouped.columns
    ]

    # Compute selected fraction
    grouped["selected_fraction"] = (
        grouped["selected_selected_count"] / grouped["selected_test_count"]
    )

    return grouped


def create_validation_summary_log(
    summary_df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Create a single-row summary log of validation performance across all rules.

    For each metric (e.g., "lift"), computes aggregate statistics over its mean values:
    - Mean of means
    - Std of means
    - Min and max of means

    Also includes:
    - Total number of unique rules evaluated
    - Maximum number of tests any rule was evaluated on
    - Average fraction of times rules were selected

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from `summarize_rule_metrics()`. Must include:
        - 'selected_test_count', 'selected_fraction'
        - For each metric in `metrics`, a `<metric>_mean` column.
    metrics : list of str
        Base metric names to summarize (e.g., ["lift", "confidence"]).

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing aggregate summary statistics.
    
    Raises
    ------
    ValueError
        If any required columns are missing from `summary_df`.
    """
    REQUIRED_COLS = {"selected_test_count", "selected_fraction"}
    missing = REQUIRED_COLS - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary_df: {missing}")

    for metric in metrics:
        col = f"{metric}_mean"
        if col not in summary_df.columns:
            raise ValueError(f"Missing column '{col}' in summary_df.")

    n_rules = len(summary_df)
    total_tests = summary_df["selected_test_count"].max()
    avg_selection_rate = summary_df["selected_fraction"].mean()

    stat_summary = {}
    for metric in metrics:
        mean_values = summary_df[f"{metric}_mean"]
        stat_summary.update({
            f"{metric}_mean_mean": mean_values.mean(),
            f"{metric}_mean_std": mean_values.std(),
            f"{metric}_mean_min": mean_values.min(),
            f"{metric}_mean_max": mean_values.max(),
        })

    result = {
        "total_rules": n_rules,
        "total_tests": total_tests,
        "avg_selection_rate": avg_selection_rate,
        **stat_summary
    }

    return pd.DataFrame([result])


def validate_bootstrap(
    df: pd.DataFrame,
    cfg: Any,
    date_col: str,
    id_cols: List[str],
    target_col: str,
    logger: Optional[Any] = None,
    random_state: int = 42,
    n_bootstrap: int = 10,
    verbose: bool = True,
    resample_method: str = "traditional",
    block_size: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform bootstrap-based validation of mined rules.

    Steps:
    - Preprocess original dataframe
    - Mine rules from the prepared dataset
    - For each bootstrap iteration:
        - Resample the original dataframe
        - Preprocess the resampled data (without re-cleaning)
        - Re-test the originally mined rules on the resample
    - Concatenate all test results
    - Aggregate and summarize rule-level metrics
    - Return rule-wise statistics and a one-row validation summary

    Parameters
    ----------
    df : pd.DataFrame
        Original input dataframe (uncleaned, raw).
    cfg : Any
        Configuration object used across pipeline components.
    logger : optional
        Optional logger for debug or progress tracking.
    random_state : int, default=42
        Seed for reproducibility.
    n_bootstrap : int, default=10
        Number of bootstrap samples to test on.
    verbose : bool, default=True
        If True, shows progress bar.
    resample_method : str, default='traditional'
        Resampling strategy to use — one of {'traditional', 'block', 'block_ids'}.
    block_size : int, default=5
        Size of blocks for block-bootstrap resampling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `bootstrap_results`: Rule-level summary statistics after all resamples.
        - `bootstrap_log`: Single-row summary log of the bootstrap run.

    Raises
    ------
    ValueError
        If required components fail or unexpected structure is returned.
    """
    rng = np.random.default_rng(random_state)
    all_results = []

    # === Step 1: Initial mining on preprocessed data ===
    prepped_df, _ = data_prep_pipeline(df, cfg, logger)
    mining_res, _, _ = mining_pipeline(prepped_df, cfg, logger)

    for i in maybe_tqdm(range(n_bootstrap), verbose, total=n_bootstrap, desc="Bootstrapping"):
        # === Step 2: Resample raw input ===
        resampled_df = resample_dataframe(
            df=df,
            mode=resample_method,
            block_size=block_size,
            date_col=date_col,
            id_cols=id_cols,
            random_state=rng.integers(0, 1_000_000_000)
        )

        # === Step 3: Prepare resample ===
        prepped_df, _ = data_prep_pipeline(resampled_df, cfg)

        # === Step 4: Re-evaluate original rules ===
        test_stats, _ = test_mined_rules(mining_res, prepped_df, cfg, target_col)
        test_stats["test_num"] = i
        all_results.append(test_stats)

    # === Step 5: Aggregate across bootstraps ===
    final_result = pd.concat(all_results, ignore_index=True)

    non_metric_cols = {"antecedents", "consequents", "selected", "test_num"}
    metrics = [col for col in final_result.columns if col not in non_metric_cols]

    bootstrap_results = summarize_rule_metrics(final_result, metrics)
    bootstrap_log = create_validation_summary_log(bootstrap_results, metrics)

    return bootstrap_results, bootstrap_log


def bootstrap_pipeline(
    df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the bootstrap resampling validation test with configuration and logging support.

    This function wraps `validate_bootstrap()` to make it compatible with the modular pipeline
    architecture. It extracts relevant parameters from the config object or override dictionary,
    runs the bootstrap test, and optionally logs the summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw data.
    cfg : Any
        Configuration object with bootstrap-related parameters. Must support attribute access (dot notation).
    logger : Optional[Any], default=None
        Optional logger object with `.log_step(...)` method for structured logging.
    **overrides : dict
        Optional keyword overrides for any bootstrap config values.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `bootstrap_results`: Rule-level summary metrics after bootstrapping.
        - `bootstrap_log`: Single-row summary of the overall bootstrap run.

    Raises
    ------
    ValueError
        If required config values are missing or the bootstrap process fails.
    """

    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    # Extract config parameters
    log_max_rows = param("log_max_rows")
    bootstrap_kwargs = {
        "date_col": param("date_col"),
        "id_cols": param("id_cols"),
        "target_col": param("target_col"),
        "n_bootstrap": param("n_bootstrap"),
        "verbose": param("bootstrap_verbose"),
        "resample_method": param("resample_method"),
        "block_size": param("block_size"),
    }

    # Run validation
    bootstrap_results, bootstrap_log = validate_bootstrap(
        df=df,
        cfg=cfg,
        logger=logger,
        **bootstrap_kwargs,
    )

    # Log output if logger is provided
    if logger:
        logger.log_step(
            step_name="Bootstrap Resampling validation test",
            info=bootstrap_kwargs,
            df=bootstrap_log,
            max_rows=log_max_rows,
        )

    return bootstrap_results, bootstrap_log

# --- Null Dist --- #
def shuffle_dataframe(
    df: pd.DataFrame,
    mode: str = "target",
    target_col: Optional[str] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Shuffle a dataframe in one of three modes: by target column, rows, or column-wise.

    This function is used in null hypothesis testing to break data dependencies
    while preserving schema and marginal distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to shuffle. This is not modified in-place.
    mode : str, default="target"
        One of:
        - "target": shuffles only the target column (requires `target_col`)
        - "rows": shuffles all rows (row-wise permutation)
        - "columns": shuffles each column independently (column-wise permutation)
    target_col : str, optional
        Required when mode="target". Specifies the column to shuffle.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled dataframe with the same structure as input.

    Raises
    ------
    ValueError
        If `mode` is invalid or if `target_col` is missing when required.
    """
    if mode not in {"target", "rows", "columns"}:
        raise ValueError("mode must be one of 'target', 'rows', or 'columns'.")

    if mode == "target" and not target_col:
        raise ValueError("target_col must be specified when mode='target'.")

    rng = np.random.default_rng(random_state)
    df_copy = df.copy()

    if mode == "target":
        shuffled = df_copy[target_col].sample(frac=1, random_state=random_state).reset_index(drop=True)
        df_copy[target_col] = shuffled
        return df_copy

    if mode == "rows":
        return df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # mode == "columns"
    for col in df_copy.columns:
        df_copy[col] = rng.permutation(df_copy[col].values)
    return df_copy

def compute_relative_error(
    df: pd.DataFrame,
    metric_col: str,
    iteration_col: str,
    m_recent: int
) -> Union[float, np.nan]:
    """
    Compute the relative error (standard deviation / mean) over the last M metric estimates.

    Typically used to assess convergence of a bootstrapped or permuted distribution,
    and inform early stopping or uncertainty diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric values and iteration identifiers.
    metric_col : str
        Column name containing the numeric metric values to evaluate (e.g., "lift_q95").
    iteration_col : str
        Column name representing iteration ordering (e.g., "perm_num" or "step").
    m_recent : int
        Number of most recent entries to include in the relative error calculation.

    Returns
    -------
    float
        Relative error, defined as std / mean over the last M values.
        Returns np.nan if the mean is zero (to avoid division by zero).

    Raises
    ------
    ValueError
        If required columns are missing or not enough data points are available.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Missing required column: '{metric_col}'")
    if iteration_col not in df.columns:
        raise ValueError(f"Missing required column: '{iteration_col}'")
    if m_recent < 1:
        raise ValueError("m_recent must be at least 1")

    df_sorted = df.sort_values(by=iteration_col).reset_index(drop=True)

    if len(df_sorted) < m_recent:
        raise ValueError(f"Not enough data points: required {m_recent}, got {len(df_sorted)}")

    recent_values = df_sorted[metric_col].iloc[-m_recent:]

    mean_val = recent_values.mean()
    std_val = recent_values.std()

    return np.nan if mean_val == 0 else std_val / mean_val

def summarize_null_distribution(
    null_df: pd.DataFrame,
    metric_col: str,
    iteration_col: str
) -> pd.DataFrame:
    """
    Generate a one-row summary of null distribution statistics for logging or diagnostics.

    Parameters
    ----------
    null_df : pd.DataFrame
        DataFrame containing the results of null/permutation runs.
    metric_col : str
        Name of the column containing the metric values (e.g., "lift_q95").
    iteration_col : str
        Name of the column used to track iteration or permutation number.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with summary statistics including mean, std, min, max,
        quantiles, and counts of permutations and total observations.

    Raises
    ------
    ValueError
        If `metric_col` or `iteration_col` is missing from `null_df`.
    """
    if metric_col not in null_df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataframe.")
    if iteration_col not in null_df.columns:
        raise ValueError(f"Iteration column '{iteration_col}' not found in dataframe.")

    metric = null_df[metric_col]
    summary = {
        "metric_mean": metric.mean(),
        "metric_std": metric.std(),
        "metric_min": metric.min(),
        "metric_max": metric.max(),
        "metric_q05": metric.quantile(0.05),
        "metric_q50": metric.median(),
        "metric_q95": metric.quantile(0.95),
        "n_permutations": null_df[iteration_col].nunique(),
        "n_observations": len(null_df)
    }

    return pd.DataFrame([summary])


def validate_null(
    df: pd.DataFrame,
    cfg: Any,
    target_col: str,
    logger: Optional[Any] = None,
    n_null: int = 1000,
    shuffle_mode: str = "target",
    early_stop_metric: str = "lift",
    es_m_permutations: int = 50,
    rel_error_threshold: float = 0.01,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a null distribution test by shuffling the target and evaluating previously mined rules.

    The function:
    - Mines rules once on the original dataset.
    - Iteratively shuffles the dataset to create null samples.
    - Re-evaluates the mined rules on each shuffled dataset.
    - Optionally applies early stopping based on the stability of a selected metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target column.
    cfg : Any
        Configuration object passed to downstream pipeline components.
    target_col : str
        Name of the column to shuffle in null mode.
    logger : Optional[Any], default=None
        Logger object for tracking steps (optional).
    n_null : int, default=1000
        Maximum number of null permutations to run.
    shuffle_mode : str, default="target"
        Shuffle mode passed to `shuffle_dataframe`. Usually "target".
    early_stop_metric : str, default="lift"
        Metric used to compute relative error for early stopping.
    es_m_permutations : int, default=50
        Number of recent iterations used to compute relative error.
    rel_error_threshold : float, default=0.01
        If relative error of metric falls below this threshold, stop early.
    verbose : bool, default=True
        If True, shows a progress bar.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Null distribution results as a dataframe of rule evaluations.
        - A single-row summary log including summary statistics and final relative error.

    Raises
    ------
    ValueError
        If `n_null < es_m_permutations`, or required columns are missing.
    """
    if n_null < es_m_permutations:
        raise ValueError("n_null must be >= es_m_permutations for early stopping to be meaningful.")

    mining_res, _, _ = mining_pipeline(df, cfg, logger)
    all_results = []

    iterator = maybe_tqdm(
        range(n_null),
        verbose,
        total=n_null,
        desc="Creating Null Distribution"
    )

    for i in iterator:
        shuffled = shuffle_dataframe(
            df=df,
            mode=shuffle_mode,
            target_col=target_col
        )

        prepped_df, _ = data_prep_pipeline(shuffled, cfg)

        result_df, _ = test_mined_rules(mining_res, prepped_df, cfg, target_col)
        result_df["test_num"] = i
        all_results.append(result_df)

        should_check = (i + 1) % es_m_permutations == 0 and (i + 1) >= es_m_permutations
        if should_check:
            interim_df = pd.concat(all_results, ignore_index=True)
            rel_error = compute_relative_error(
                df=interim_df,
                metric_col=early_stop_metric,
                iteration_col="test_num",
                m_recent=es_m_permutations
            )
            if rel_error < rel_error_threshold:
                null_log = summarize_null_distribution(interim_df, early_stop_metric, "test_num")
                null_log["final_rel_error"] = rel_error
                return interim_df, null_log

    final_df = pd.concat(all_results, ignore_index=True)
    rel_error = compute_relative_error(
        df=final_df,
        metric_col=early_stop_metric,
        iteration_col="test_num",
        m_recent=es_m_permutations
    )
    null_log = summarize_null_distribution(final_df, early_stop_metric, "test_num")
    null_log["final_rel_error"] = rel_error

    return final_df, null_log

def null_pipeline(
    df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the null distribution validation test using config and optional overrides.

    This function wraps `validate_null()` to align with the modular pipeline pattern.
    It extracts the required parameters from either the `cfg` object or from `overrides`,
    runs the null validation process, and returns the null distribution and summary log.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw feature and target data.
    cfg : Any
        Configuration object with null test parameters. Must support attribute-style access.
    logger : Optional[Any], default=None
        Optional logger instance for structured logging.
    **overrides : Any
        Optional keyword overrides for config parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `null_df`: DataFrame of test statistics from the null distribution.
        - `null_log`: One-row summary of null test metrics and run statistics.

    Raises
    ------
    ValueError
        If required configuration values are missing or invalid.
    """

    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    log_max_rows = param("log_max_rows")

    null_kwargs = {
        "target_col": param("target_col"),
        "n_null": param("n_null"),
        "shuffle_mode": param("shuffle_mode"),
        "early_stop_metric": param("early_stop_metric"),
        "es_m_permutations": param("es_m_permutations"),
        "rel_error_threshold": param("rel_error_threshold"),
        "verbose": param("null_verbose"),
    }

    null_df, null_log = validate_null(
        df=df,
        cfg=cfg,
        logger=logger,
        **null_kwargs,
    )

    if logger:
        logger.log_step(
            step_name="Null Distribution validation test",
            info=null_kwargs,
            df=null_log,
            max_rows=log_max_rows,
        )

    return null_df, null_log

# --- Multiple Testing Correction --- #
def summarize_fdr_results(
    df: pd.DataFrame,
    pval_col: str = "pval",
    fdr_sig_col: str = "fdr_significant",
    correction_alpha: float = 0.05,
    groupby_col: Optional[str] = None,
    as_markdown: bool = False
) -> Union[pd.DataFrame, str]:
    """
    Summarize False Discovery Rate (FDR) test results for reporting or logging.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing p-values and FDR significance flags.
    pval_col : str, default="pval"
        Column name containing p-values.
    fdr_sig_col : str, default="fdr_significant"
        Column name containing boolean FDR significance flags.
    correction_alpha : float, default=0.05
        Alpha threshold used in FDR correction (e.g., 0.05).
    groupby_col : str or None, optional
        If provided, summarizes results per group (e.g., by rule depth or algorithm).
    as_markdown : bool, default=False
        If True, returns a markdown-formatted summary string; otherwise, returns a summary DataFrame.

    Returns
    -------
    Union[pd.DataFrame, str]
        Summary DataFrame or markdown-formatted summary string.
    """
    REQUIRED_COLS = [pval_col, fdr_sig_col]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    def _summarize(sub_df: pd.DataFrame) -> pd.Series:
        total = len(sub_df)
        sig_count = sub_df[fdr_sig_col].sum()
        summary = {
            "Total Tested": total,
            "Significant (FDR)": sig_count,
            "Proportion Significant": sig_count / total if total else float("nan"),
            f"P < {correction_alpha}": (sub_df[pval_col] < correction_alpha).sum(),
            "P < 0.01": (sub_df[pval_col] < 0.01).sum(),
            "P < 0.05": (sub_df[pval_col] < 0.05).sum(),
            "Min P-value": sub_df[pval_col].min(),
            "Median P-value": sub_df[pval_col].median(),
            "Max P-value": sub_df[pval_col].max(),
        }
        return pd.Series(summary)

    if groupby_col:
        summary_df = df.groupby(groupby_col, dropna=False).apply(_summarize).reset_index()
    else:
        summary_df = _summarize(df).to_frame().T

    if not as_markdown:
        return summary_df

    def _df_to_markdown(sub_df: pd.DataFrame, group_label: Optional[str] = None) -> str:
        lines = []
        if group_label is not None:
            lines.append(f"#### Group: {group_label}")
        for col, val in sub_df.items():
            lines.append(f"- **{col}:** {val}")
        return "\n".join(lines)

    if groupby_col:
        md_blocks = [
            _df_to_markdown(row.drop(labels=groupby_col), group_label=row[groupby_col])
            for _, row in summary_df.iterrows()
        ]
        return "\n\n".join(md_blocks)
    else:
        return _df_to_markdown(summary_df.iloc[0])

def compute_empirical_pvals(
    actual_stats: np.ndarray,
    null_stats: np.ndarray,
    mode: Literal["greater", "less", "two-sided"] = "greater",
    center: Optional[float] = None
) -> np.ndarray:
    """
    Computes empirical p-values by comparing actual statistics to a null distribution.

    Parameters
    ----------
    actual_stats : np.ndarray
        Array of test statistics from the actual dataset.
    null_stats : np.ndarray
        Array of test statistics from the null distribution.
    mode : {"greater", "less", "two-sided"}, default="greater"
        - "greater": One-tailed test for values significantly higher than null.
        - "less": One-tailed test for values significantly lower than null.
        - "two-sided": Two-tailed test for values significantly different from center of null.
    center : float or None, optional
        Reference center for "two-sided" test. If None, uses median of null distribution.

    Returns
    -------
    np.ndarray
        Empirical p-values for each actual statistic.
    """
    null_stats = np.asarray(null_stats)
    actual_stats = np.asarray(actual_stats)

    if mode == "greater":
        return np.array([(null_stats >= val).mean() for val in actual_stats])
    elif mode == "less":
        return np.array([(null_stats <= val).mean() for val in actual_stats])
    elif mode == "two-sided":
        if center is None:
            center = np.median(null_stats)
        null_deltas = np.abs(null_stats - center)
        actual_deltas = np.abs(actual_stats - center)
        return np.array([(null_deltas >= delta).mean() for delta in actual_deltas])
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Choose from 'greater', 'less', 'two-sided'.")
        
def validate_multiple_tests(
    mining_res: pd.DataFrame,
    null_df: pd.DataFrame,
    early_stop_metric: str = "lift",
    mode: Literal["greater", "less", "two-sided"] = "greater",
    correction_alpha: float = 0.05,
    correction_metric: str = "fdr_bh",
    center: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate rule mining results by computing empirical p-values and applying FDR correction.

    Parameters
    ----------
    mining_res : pd.DataFrame
        DataFrame containing the actual rule mining results, including the test statistic.
    null_df : pd.DataFrame
        DataFrame containing the null distribution for the same test statistic.
    early_stop_metric : str, default="lift"
        The column name of the test statistic to compare between actual and null.
    mode : {"greater", "less", "two-sided"}, default="greater"
        Defines the tail for p-value calculation:
        - "greater": right-tailed test (higher is better)
        - "less": left-tailed test (lower is better)
        - "two-sided": extreme values in both directions are significant
    correction_alpha : float, default=0.05
        Significance threshold for FDR correction.
    correction_metric : str, default="fdr_bh"
        Method for multiple testing correction. Compatible with `statsmodels.stats.multitest.multipletests`.
    center : float or None, optional
        If using "two-sided" mode, this is the reference center for deviation (defaults to median of null distribution).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Updated mining results with raw and corrected p-values, and boolean significance flag.
        - Summary DataFrame suitable for logging/reporting.

    Raises
    ------
    ValueError
        If required columns are missing or FDR correction fails.
    """
    # --- Input validation
    for df, label in [(mining_res, "mining_res"), (null_df, "null_df")]:
        if early_stop_metric not in df.columns:
            raise ValueError(f"Column '{early_stop_metric}' not found in {label} DataFrame.")

    actual_stats = mining_res[early_stop_metric].values
    null_stats = null_df[early_stop_metric].values

    # --- Compute empirical p-values (three-mode logic)
    pvals = compute_empirical_pvals(
        actual_stats=actual_stats,
        null_stats=null_stats,
        mode=mode,
        center=center
    )
    mining_res = mining_res.copy()
    mining_res["pval"] = pvals

    # --- Apply FDR correction
    try:
        rejected, pvals_corrected, _, _ = multipletests(
            pvals, alpha=correction_alpha, method=correction_metric
        )
    except Exception as e:
        raise ValueError(f"FDR correction failed with method '{correction_metric}': {str(e)}")

    # --- Annotate results
    pval_col = f"pval_{correction_metric}"
    sig_col = f"{correction_metric}_significant"
    mining_res[pval_col] = pvals_corrected
    mining_res[sig_col] = rejected

    # --- Generate logging summary
    log_summary = summarize_fdr_results(
        df=mining_res,
        pval_col="pval",
        fdr_sig_col=sig_col,
        correction_alpha=correction_alpha,
        as_markdown=False
    )

    return mining_res, log_summary


def fdr_pipeline(
    mining_res: pd.DataFrame,
    null_df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the FDR multiple testing correction validation using configuration and overrides.

    This function wraps `validate_multiple_tests` in a modular pipeline format. It extracts
    required parameters from either the `cfg` object or from `overrides`, performs empirical
    p-value computation and FDR correction, and optionally logs a summary.

    Parameters
    ----------
    mining_res : pd.DataFrame
        DataFrame of actual rule mining results to evaluate.
    null_df : pd.DataFrame
        DataFrame of null distribution values for the same test statistic.
    cfg : Any
        Configuration object with FDR test parameters. Must support attribute-style access.
    logger : Optional[Any], default=None
        Optional logger instance for structured logging.
    **overrides : Any
        Optional keyword overrides for config parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `fdr_res`: Rule mining results annotated with p-values and FDR significance flags.
        - `fdr_log`: One-row summary of FDR validation statistics.

    Raises
    ------
    ValueError
        If required configuration values are missing or invalid.
    """
    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    log_max_rows = param("log_max_rows")

    fdr_kwargs = {
        "early_stop_metric": param("early_stop_metric"),
        "mode": param("fdr_mode"),
        "correction_alpha": param("correction_alpha"),
        "correction_metric": param("correction_metric"),
    }

    fdr_res, fdr_log = validate_multiple_tests(
        mining_res=mining_res,
        null_df=null_df,
        **fdr_kwargs,
    )

    if logger:
        logger.log_step(
            step_name="FDR Multiple Correction validation test",
            info=fdr_kwargs,
            df=fdr_log,
            max_rows=log_max_rows,
        )

    return fdr_res, fdr_log
