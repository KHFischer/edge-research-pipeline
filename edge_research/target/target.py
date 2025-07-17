import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Literal, Tuple, Dict, Any

def compute_forward_return(
    price_df: pd.DataFrame,
    n_periods: int = 60,
    id_cols: List[str] = ["ticker"],
    date_col: str = "date",
    target_col: str = "forward_return",
    price_col: str = "adj_close",
    return_mode: Literal["pct_change", "log_return", "vol_adjusted", "smoothed"] = "pct_change",
    vol_window: Optional[int] = None,
    smoothing_method: Literal["median", "mean", "max", "min"] = "median"
) -> pd.DataFrame:
    """
    Calculate forward returns over a specified horizon for each instrument group, supporting multiple return modes.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data with ID, date, and price columns.
    n_periods : int
        Forward horizon (must be > 0).
    id_cols : List[str]
        Columns identifying instruments/groups.
    date_col : str
        Name of the datetime column.
    target_col : str
        Output column name for the computed return.
    price_col : str
        Name of the price column to base returns on.
    return_mode : str
        Return calculation mode. Options:
            - 'pct_change' : Simple forward % change.
            - 'log_return' : Log(future_price / current_price).
            - 'vol_adjusted' : % change divided by realized volatility.
            - 'smoothed' : % change from current price to median/max/min future price.
    vol_window : Optional[int]
        Window (in periods) for realized volatility calculation (used in 'vol_adjusted' mode).
    smoothing_method : str
        Smoothing method ('median', 'mean', 'max', 'min') for 'smoothed' mode.

    Returns
    -------
    pd.DataFrame
        DataFrame with ID columns, date_col, and target_col.

    Raises
    ------
    ValueError
        If required columns are missing, if n_periods <= 0, or if invalid mode/options are provided.
    """
    required_cols = set(id_cols + [date_col, price_col])
    missing_cols = required_cols - set(price_df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    if n_periods <= 0:
        raise ValueError(f"n_periods must be positive, got {n_periods}")

    df = price_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.sort_values(by=id_cols + [date_col])

    # Compute future price or series depending on mode
    if return_mode == "smoothed":
        future_window = df.groupby(id_cols, observed=True)[price_col].rolling(window=n_periods, min_periods=1).apply({
            "median": np.median,
            "mean": np.mean,
            "max": np.max,
            "min": np.min
        }[smoothing_method], raw=True).reset_index(level=0, drop=True).shift(-n_periods + 1)
        df["_future_value"] = future_window

    else:
        df["_future_value"] = df.groupby(id_cols, observed=True)[price_col].shift(-n_periods)

    # Compute returns based on mode
    if return_mode == "pct_change":
        df[target_col] = (df["_future_value"] - df[price_col]) / df[price_col]

    elif return_mode == "log_return":
        df[target_col] = np.log(df["_future_value"] / df[price_col])

    elif return_mode == "vol_adjusted":
        if vol_window is None:
            raise ValueError("vol_window must be provided for vol_adjusted mode.")
        returns = df.groupby(id_cols, observed=True)[price_col].apply(lambda x: x.pct_change())
        realized_vol = returns.groupby(df[id_cols].apply(tuple, axis=1), observed=True).rolling(vol_window, min_periods=1).std().reset_index(level=0, drop=True)
        df["_realized_vol"] = realized_vol.shift(-n_periods + 1)
        df[target_col] = ((df["_future_value"] - df[price_col]) / df[price_col]) / df["_realized_vol"]

    elif return_mode == "smoothed":
        df[target_col] = (df["_future_value"] - df[price_col]) / df[price_col]

    else:
        raise ValueError(f"Unsupported return_mode: {return_mode}")

    # Drop rows where future value is missing
    df = df.dropna(subset=["_future_value", target_col])

    df[id_cols] = df[id_cols].astype(str)
    df = df.drop(columns=["_future_value"])
    if "_realized_vol" in df.columns:
        df = df.drop(columns=["_realized_vol"])

    return df[id_cols + [date_col, target_col]]

def merge_features_with_returns(
    feature_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    id_cols: List[str],
    feature_date_col: str,
    returns_date_col: str,
    direction: str = "forward",
    tolerance: Optional[pd.Timedelta] = None,
    id_case: str = "upper"
) -> pd.DataFrame:
    """
    Merge a feature dataframe with a returns dataframe using a case-agnostic,
    timezone-normalized, forward-looking as-of merge (by default), preserving NaNs.

    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame containing feature rows, indexed by ID columns and feature_date_col.
    returns_df : pd.DataFrame
        DataFrame containing returns or targets to merge, indexed similarly.
    id_cols : List[str]
        List of columns identifying the instrument or entity (e.g. ticker).
    feature_date_col : str
        Name of the datetime column in feature_df.
    returns_date_col : str
        Name of the datetime column in returns_df.
    direction : str, default="forward"
        Merge direction: "forward", "backward", or "nearest".
        Use "forward" to prevent forward-looking bias (default).
    tolerance : Optional[pd.Timedelta], default=None
        Optional time window tolerance for merges (e.g. "7D").
    id_case : str, default="upper"
        Controls case-normalization of ID columns:
        - "upper": convert to uppercase (default)
        - "lower": convert to lowercase
        - "original": preserve case

    Returns
    -------
    pd.DataFrame
        Merged dataframe with feature_df columns plus returns columns merged as-of
        from returns_df. All NaNs preserved where no merge was possible.

    Raises
    ------
    ValueError
        If dates are unparsable, or direction/id_case arguments are invalid.

    Notes
    -----
    This function enforces:
    - Case normalization of ID columns
    - Timezone-naive UTC date columns
    - Sort order required for pandas merge_asof
    - No backward merging unless explicitly allowed via direction
    """
    VALID_DIRECTIONS = {"forward", "backward", "nearest"}
    VALID_ID_CASES = {"upper", "lower", "original"}

    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}")

    if id_case not in VALID_ID_CASES:
        raise ValueError(f"id_case must be one of {VALID_ID_CASES}")

    # --- Normalize IDs ---
    def _normalize_ids(df: pd.DataFrame) -> None:
        for col in id_cols:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                if id_case == "upper":
                    df[col] = df[col].astype(str).str.upper()
                elif id_case == "lower":
                    df[col] = df[col].astype(str).str.lower()
                else:  # "original"
                    df[col] = df[col].astype(str)

    f_df = feature_df.copy()
    r_df = returns_df.copy()

    _normalize_ids(f_df)
    _normalize_ids(r_df)

    # --- Normalize datetime columns ---
    def _normalize_datetime(df: pd.DataFrame, col: str) -> None:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        if df[col].isna().any():
            raise ValueError(f"{col} contains unparsable dates in dataframe.")

    _normalize_datetime(f_df, feature_date_col)
    _normalize_datetime(r_df, returns_date_col)

    # --- Sort for merge_asof ---
    f_df = f_df.sort_values([feature_date_col] + id_cols)
    r_df = r_df.sort_values([returns_date_col] + id_cols)

    # --- Sub-select RHS columns (targets) ---
    rhs_cols = [col for col in r_df.columns if col not in id_cols and col != returns_date_col]
    r_df_subset = r_df[id_cols + [returns_date_col] + rhs_cols]

    # --- Perform merge_asof ---
    merged_df = pd.merge_asof(
        left=f_df,
        right=r_df_subset,
        left_on=feature_date_col,
        right_on=returns_date_col,
        by=id_cols,
        direction=direction,
        tolerance=tolerance,
        allow_exact_matches=True
    )

    return merged_df
    
def summarize_merge(
    merged: pd.DataFrame,
    id_cols: List[str],
    returns_date_col: str,
    feature_date_col: str,
    target_col: str = "forward_return",
    sample_size: int = 10
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Summarize merge results by counting matched/unmatched rows and sampling unmatched cases.

    Parameters
    ----------
    merged : pd.DataFrame
        The merged dataframe containing features and targets.
    id_cols : List[str]
        List of column names identifying each instrument/entity (e.g., ticker).
    returns_date_col : str
        Name of the column holding target (returns) dates.
    feature_date_col : str
        Name of the feature date column.
    target_col : str, default="forward_return"
        Name of the target column indicating whether a match occurred.
    sample_size : int, default=10
        Number of unmatched samples to include in the output.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]
        - log_summary : dict
            Summary counts including total, matched, and unmatched rows.
        - counts_per_id : pd.DataFrame
            Per-ID breakdown of total, matched, and unmatched counts.
        - unmatched_sample : pd.DataFrame
            Sample of unmatched rows (ID columns + feature date).

    Raises
    ------
    ValueError
        If required columns are missing in the merged dataframe.

    Notes
    -----
    - Rows are considered matched if the target_col is not NaN.
    - The function is non-destructive and returns copies.
    """
    required_cols = set(id_cols + [target_col, feature_date_col])
    missing_cols = required_cols - set(merged.columns)
    if missing_cols:
        raise ValueError(f"Merged dataframe missing required columns: {missing_cols}")

    merged = merged.copy()
    merged["_matched"] = ~merged[target_col].isna()

    total_rows = len(merged)
    matched_rows = int(merged["_matched"].sum())
    unmatched_rows = total_rows - matched_rows

    counts_per_id = (
        merged.groupby(id_cols)["_matched"]
        .agg(total="count", matched="sum")
        .reset_index()
    )
    counts_per_id["unmatched"] = counts_per_id["total"] - counts_per_id["matched"]

    unmatched_sample = (
        merged.loc[~merged["_matched"], id_cols + [feature_date_col]]
        .head(sample_size)
        .copy()
    )

    log_summary = {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
    }

    return log_summary, counts_per_id, unmatched_sample

def bin_target_column(
    df: pd.DataFrame,
    binning_method: Literal["quantile", "custom", "binary"],
    bins: List[float],
    labels: List[str],
    target_col: str = "return",
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    grouping: str = "none",
    n_datetime_units: Optional[int] = None,
    nan_placeholder: str = "no_data"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin a target column using quantile, custom threshold, or binary encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    binning_method : {'quantile', 'custom', 'binary'}
        Binning strategy.
    bins : List[float]
        For 'quantile': quantile cutoffs (0–1).
        For 'custom': numeric thresholds.
        For 'binary': single threshold (optional; default=0 if empty).
    labels : List[str]
        Labels for bins or binary encoding.
    target_col : str, default='return'
        Column to bin.
    id_cols : Optional[List[str]]
        ID columns for grouping in quantile binning.
    date_col : Optional[str]
        Date column for quantile binning with datetime grouping.
    grouping : {'none', 'ids', 'datetime', 'datetime+ids'}
        Grouping method for quantile binning.
    n_datetime_units : Optional[int]
        Number of rows per time window (datetime grouping).
    nan_placeholder : str, default='no_data'
        Value to assign to NaNs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Binned dataframe and binning log dataframe.
    """
    if binning_method not in {"quantile", "custom", "binary"}:
        raise ValueError("binning_method must be 'quantile', 'custom', or 'binary'")

    if grouping not in {"none", "ids", "datetime", "datetime+ids"}:
        raise ValueError(f"Invalid grouping: {grouping}")

    df_new = df.copy()
    log_entries = []

    if binning_method == "binary":
        threshold = bins[0] if bins else 0
        pos_label, neg_label = labels if labels else ["positive", "negative"]

        df_new[target_col] = df_new[target_col].apply(
            lambda x: pos_label if pd.notna(x) and x > threshold else neg_label
        )

        log_entries.append(pd.DataFrame({
            "method": ["binary"],
            "threshold": [threshold],
            "labels": [(pos_label, neg_label)],
            "group": ["global"]
        }))

        log_df = pd.concat(log_entries, ignore_index=True)
        return df_new, log_df

    elif binning_method == "custom":
        try:
            bin_result = pd.cut(
                df_new[target_col],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
        except ValueError:
            bin_result = pd.Series([nan_placeholder] * len(df_new), index=df_new.index)

        bin_result_str = bin_result.astype(str).where(~bin_result.isna(), nan_placeholder)
        df_new[target_col] = bin_result_str.astype(object)

        log_entries.append(pd.DataFrame({
            "method": ["custom"],
            "group": ["global"],
            "bins": [bins],
            "labels": [labels]
        }))

        log_df = pd.concat(log_entries, ignore_index=True)
        return df_new, log_df

    # ---- Quantile binning ----
    if grouping in {"ids", "datetime+ids"} and not id_cols:
        raise ValueError("id_cols must be provided when grouping includes 'ids'.")
    if grouping in {"datetime", "datetime+ids"}:
        if date_col is None or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping includes 'datetime'.")

    if grouping == "none":
        df_new["_group"] = "all"
    elif grouping == "ids":
        df_new["_group"] = df_new[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_new.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_new = df_sorted.set_index("index").sort_index()
    elif grouping == "datetime+ids":
        df_new["_group"] = None
        chunks = []
        grouped = df_new.groupby(id_cols, group_keys=False)
        for name, group in grouped:
            group_sorted = group.sort_values(date_col).reset_index()
            group_sorted["_group"] = (group_sorted.index // n_datetime_units).astype(str)
            prefix = "_".join(str(v) for v in name) if isinstance(name, tuple) else str(name)
            group_sorted["_group"] = prefix + "_window_" + group_sorted["_group"]
            chunks.append(group_sorted.set_index("index"))
        df_new = pd.concat(chunks).sort_index()

    grouped = df_new.groupby("_group")

    for group_name, group_df in grouped:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bin_result = pd.qcut(
                    group_df[target_col],
                    q=bins,
                    labels=labels,
                    duplicates="drop"
                )
        except ValueError:
            bin_result = pd.Series([nan_placeholder] * len(group_df), index=group_df.index)

        bin_result_str = bin_result.astype(str).where(~bin_result.isna(), nan_placeholder)
        df_new[target_col] = df_new[target_col].astype(object)
        df_new.loc[group_df.index, target_col] = bin_result_str

        log_entries.append(pd.DataFrame({
            "method": ["quantile"],
            "group": [group_name],
            "quantiles": [bins],
            "labels": [labels]
        }))

    df_new = df_new.drop(columns="_group")
    df_new[target_col] = df_new[target_col].astype(str)
    log_df = pd.concat(log_entries, ignore_index=True)

    return df_new, log_df