import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
import warnings
from pandas.api.types import CategoricalDtype

MAX_REPLACEMENT_DEFAULT = 1e60
def generate_ratio_features(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = "all",
    suffix: str = "ratio",
    max_replacement: Optional[float] = MAX_REPLACEMENT_DEFAULT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate all unique pairwise ratio features between specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns.
    columns : Union[str, List[str]], optional
        Either:
        - "all": Automatically infer numeric columns.
        - List of specific column names to use for ratio generation.
        Default is "all".
    suffix : str, optional
        Suffix appended to generated column names. Default is "ratio".
    max_replacement : float, optional
        Value used to replace infinite division results. Defaults to 1e60.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with new ratio columns added.
        - DataFrame logging the generated features with columns:
            ['numerator', 'denominator', 'new_column']

    Raises
    ------
    ValueError
        If fewer than two valid columns are found for ratio generation.

    Notes
    -----
    - Only generates one direction per pair (A/B where A index < B index).
    - Handles divide-by-zero and infinite values gracefully.
    """
    if columns == "all":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, list):
        numeric_cols = [col for col in columns if col in df.columns]
    else:
        raise TypeError("columns must be 'all' or a list of column names.")

    if len(numeric_cols) < 2:
        raise ValueError(
            f"Need at least 2 numeric columns for ratio generation. Found: {numeric_cols}"
        )

    df_new = df.copy()
    created_features = []

    for i, numerator in enumerate(numeric_cols):
        for j, denominator in enumerate(numeric_cols):
            if i >= j:
                continue  # skip self and reciprocal pairs

            new_col = f"{numerator}_div_{denominator}_{suffix}"

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df_new[numerator] / df_new[denominator]

            ratio_clean = ratio.replace(
                {np.inf: max_replacement, -np.inf: -max_replacement}
            )

            df_new[new_col] = ratio_clean

            created_features.append({
                "numerator": numerator,
                "denominator": denominator,
                "new_column": new_col
            })

    log_df = pd.DataFrame(created_features, columns=["numerator", "denominator", "new_column"])
    return df_new, log_df

def generate_temporal_pct_change(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = "all",
    id_cols: List[str] = [],
    datetime_col: str = "",
    n_dt: int = 1,
    suffix: str = "pctchange"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate temporal percent change features over n_dt rows for specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Union[str, List[str]], optional
        Either:
        - "all": Use all numeric columns.
        - List of specific column names to compute percent change on.
        Default is "all".
    id_cols : List[str], optional
        Columns used to identify groups/entities for grouping.
    datetime_col : str, optional
        Name of the datetime column to order rows within each group.
    n_dt : int, optional
        Number of rows to lag for percent change computation. Default is 1.
    suffix : str, optional
        Suffix appended to generated feature column names. Default is "pctchange".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with new percent change feature columns added.
        - DataFrame logging generated columns with:
            ['original_column', 'new_column', 'n_lag']

    Raises
    ------
    ValueError
        If fewer than 2 valid columns are found for feature generation.

    Notes
    -----
    - Designed to work with row-indexed lagging (`n_dt`) rather than fixed time intervals.
    - Handles each group independently using the provided ID columns.
    - Does not modify the input dataframe in-place.
    """
    if not datetime_col:
        raise ValueError("datetime_col must be specified.")

    if columns == "all":
        selected_cols = df.select_dtypes(include=["number"]).columns.difference(id_cols + [datetime_col]).tolist()
    elif isinstance(columns, list):
        selected_cols = [col for col in columns if col in df.columns]
    else:
        raise TypeError("columns must be 'all' or a list of column names.")

    if len(selected_cols) < 1:
        raise ValueError(f"At least one valid numeric column is required for percent change. Found: {selected_cols}")

    df_new = df.copy()

    # Ensure sorting
    df_new = df_new.sort_values(by=id_cols + [datetime_col]).reset_index(drop=True)

    grouped = df_new.groupby(id_cols, group_keys=False, observed=False)

    created_features = []

    for col in selected_cols:
        new_col = f"{col}_{suffix}"

        lagged = grouped[col].shift(n_dt)
        pct_change = (df_new[col] - lagged) / lagged

        df_new[new_col] = pct_change

        created_features.append({
            "original_column": col,
            "new_column": new_col,
            "n_lag": n_dt
        })

    log_df = pd.DataFrame(created_features, columns=["original_column", "new_column", "n_lag"])

    return df_new, log_df

def extract_date_features(
    df: pd.DataFrame,
    date_col: str,
    prefix: str = "dt_"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract calendar-based datetime features from a specified date column.

    Features Added:
        - {prefix}year
        - {prefix}quarter
        - {prefix}month
        - {prefix}week
        - {prefix}weekday
        - {prefix}is_month_end
        - {prefix}is_month_start
        - {prefix}is_quarter_end
        - {prefix}is_quarter_start
        - {prefix}is_year_end
        - {prefix}is_year_start

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the datetime column to extract features from.
    prefix : str, optional
        Prefix for generated feature columns. Default is "dt_".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Dataframe with new datetime-derived feature columns appended.
        - Dataframe logging generated columns, with:
            ['source_column', 'new_column', 'feature_type']
    """
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in dataframe.")

    df_out = df.copy()

    # Ensure datetime type
    try:
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="raise")
    except Exception as e:
        raise ValueError(f"Cannot convert '{date_col}' to datetime: {e}")

    dt_series = df_out[date_col]

    features_added = []

    # Numeric calendar features - cast to categorical for OHE
    numeric_features = {
        "year": dt_series.dt.year,
        "quarter": dt_series.dt.quarter,
        "month": dt_series.dt.month,
        "week": dt_series.dt.isocalendar().week,
        "weekday": dt_series.dt.weekday
    }

    for suffix, series in numeric_features.items():
        col_name = f"{prefix}{suffix}"
        df_out[col_name] = series.astype("category")  # Ensure ready for get_dummies
        features_added.append({
            "source_column": date_col,
            "new_column": col_name,
            "feature_type": "calendar_numeric_categorical"
        })

    # Calendar boundary flags
    flag_features = {
        "is_month_end": dt_series.dt.is_month_end.astype(int),
        "is_month_start": dt_series.dt.is_month_start.astype(int),
        "is_quarter_end": dt_series.dt.is_quarter_end.astype(int),
        "is_quarter_start": dt_series.dt.is_quarter_start.astype(int),
        "is_year_end": dt_series.dt.is_year_end.astype(int),
        "is_year_start": dt_series.dt.is_year_start.astype(int)
    }

    for suffix, series in flag_features.items():
        col_name = f"{prefix}{suffix}"
        df_out[col_name] = series
        features_added.append({
            "source_column": date_col,
            "new_column": col_name,
            "feature_type": "calendar_flag"
        })

    feature_log = pd.DataFrame(features_added, columns=["source_column", "new_column", "feature_type"])

    return df_out, feature_log

def bin_columns_flexible(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = "all",
    quantiles: List[float] = [0, 0.25, 0.5, 0.75, 1.0],
    quantile_labels: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    grouping: str = "none",
    n_datetime_units: Optional[int] = None,
    nan_placeholder: str = "no_data"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin numeric columns into quantile-based categories with optional grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Union[str, List[str]], optional
        "all" to automatically infer numeric columns, or list of column names. Default is "all".
    quantiles : List[float], optional
        Quantile edges (from 0 to 1) to use as bin cutoffs.
    quantile_labels : List[str], optional
        Labels assigned to bins. If None, bins are labeled numerically.
    id_cols : List[str], optional
        Columns identifying entity groups (used if grouping involves 'ids').
    date_col : str, optional
        Datetime column to control temporal grouping.
    grouping : str, optional
        One of: "none", "ids", "datetime", "datetime+ids".
    n_datetime_units : int, optional
        Row count per time window (only required if grouping uses datetime).
    nan_placeholder : str, optional
        Label used for missing or unassigned bins.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with binned columns as categorical strings.
        - DataFrame logging binning parameters for each group and column.

    Raises
    ------
    ValueError
        For invalid grouping mode or missing required arguments.
    """

    VALID_GROUPINGS = {"none", "ids", "datetime", "datetime+ids"}

    if grouping not in VALID_GROUPINGS:
        raise ValueError(f"Invalid grouping: {grouping}. Must be one of {VALID_GROUPINGS}.")

    if grouping in {"ids", "datetime+ids"} and not id_cols:
        raise ValueError("id_cols must be provided when grouping includes 'ids'.")

    if grouping in {"datetime", "datetime+ids"}:
        if date_col is None or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units are required for datetime-based grouping.")

    # Determine target columns
    if columns == "all":
        target_cols = (
            df.select_dtypes(include="number")
            .columns.difference(id_cols or [])
            .difference([date_col or ""])
            .tolist()
        )
    
        # Exclude columns starting with "dt_"
        target_cols = [col for col in target_cols if not col.startswith("dt_")]
    
    elif isinstance(columns, list):
        target_cols = [col for col in columns if col in df.columns]
    
    else:
        raise TypeError("columns must be 'all' or a list of column names.")

    if len(target_cols) < 1:
        raise ValueError(f"At least one numeric column required for binning. Found: {target_cols}")

    df_new = df.copy()
    log_entries = []

    # Generate grouping keys
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
        for _, group in df_new.groupby(id_cols, group_keys=False):
            group_sorted = group.sort_values(date_col).reset_index()
            group_sorted["_group"] = (group_sorted.index // n_datetime_units).astype(str)
            prefix = "_".join(str(v) for v in group_sorted.loc[0, id_cols])
            group_sorted["_group"] = prefix + "_window_" + group_sorted["_group"]
            chunks.append(group_sorted.set_index("index"))
        df_new = pd.concat(chunks).sort_index()

    grouped = df_new.groupby("_group")

    for group_name, group_df in grouped:
        for col in target_cols:
            df_new[col] = df_new[col].astype(object)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bin_result = pd.qcut(
                        group_df[col],
                        q=quantiles,
                        labels=quantile_labels,
                        duplicates="drop"
                    )
            except ValueError:
                bin_result = pd.Series([nan_placeholder] * len(group_df), index=group_df.index)

            bin_result_str = bin_result.astype(str).where(~bin_result.isna(), nan_placeholder)
            df_new.loc[group_df.index, col] = bin_result_str

            log_entries.append({
                "group": group_name,
                "column": col,
                "quantiles": quantiles,
                "labels": quantile_labels if quantile_labels else list(range(len(quantiles) - 1))
            })

    df_new = df_new.drop(columns="_group")

    for col in target_cols:
        if not isinstance(df_new[col].dtype, CategoricalDtype):
            df_new[col] = df_new[col].astype(str)

    log_df = pd.DataFrame(log_entries)

    return df_new, log_df

def sweep_low_count_bins(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = "all",
    min_count: Optional[int] = None,
    min_fraction: Optional[float] = None,
    reserved_labels: Optional[List[str]] = None,
    sweep_label: str = "others"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep low-frequency categories in specified or all categorical columns into 'others',
    logging which categories were swept.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to process.
    columns : Union[str, List[str]], optional
        "all" to automatically detect suitable columns (categoricals or object dtype),
        or explicit list of column names. Default is "all".
    min_count : int, optional
        Absolute count threshold for sweeping.
    min_fraction : float, optional
        Fractional threshold (0-1) for sweeping.
    reserved_labels : List[str], optional
        Labels that should never be swept (e.g. ['no_data']).
    sweep_label : str, optional
        Label to assign to swept categories. Default is "others".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Modified DataFrame with sweeping applied.
        - Log DataFrame with columns: ['column', 'bin_swept', 'count_swept'].
    """
    if min_count is None and min_fraction is None:
        raise ValueError("At least one of min_count or min_fraction must be specified.")

    if reserved_labels is None:
        reserved_labels = []

    df_out = df.copy()
    log_entries = []

    # Infer columns if needed
    if columns == "all":
        target_cols = [
            col for col in df_out.columns
            if (
                (isinstance(df_out[col].dtype, CategoricalDtype) or df_out[col].dtype == object)
                and not col.startswith("dt_")  # <-- Exclude dt_* columns
            )
        ]

    elif isinstance(columns, list):
        target_cols = [col for col in columns if col in df_out.columns]
    else:
        raise TypeError("columns must be 'all' or a list of column names.")

    for col in target_cols:
        value_counts = df_out[col].value_counts(dropna=False)
        total = value_counts.sum()

        count_threshold = min_count if min_count is not None else 0
        fraction_threshold = int(min_fraction * total) if min_fraction is not None else 0
        threshold = max(count_threshold, fraction_threshold)

        sweep_categories = [
            category for category, count in value_counts.items()
            if (count < threshold) and (category not in reserved_labels)
        ]

        for category in sweep_categories:
            count = value_counts[category]
            log_entries.append({
                "column": col,
                "bin_swept": category,
                "count_swept": count
            })

        if sweep_categories:
            if isinstance(df_out[col].dtype, CategoricalDtype):
                df_out[col] = df_out[col].astype(object)
            df_out[col] = df_out[col].replace(sweep_categories, sweep_label)

    log_df = pd.DataFrame(log_entries, columns=["column", "bin_swept", "count_swept"])

    return df_out, log_df

def one_hot_encode_features(
    df: pd.DataFrame,
    id_cols: List[str],
    date_col: Optional[str],
    drop_cols: List[str],
    no_data_label: str = "no_data",
    drop_no_data_columns: bool = False
) -> pd.DataFrame:
    """
    One-hot encode dataframe while excluding id/date/drop columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical features.
    id_cols : List[str]
        Columns used as IDs. Retained but not encoded.
    date_col : str or None
        Datetime column. Retained but not encoded.
    drop_cols : List[str]
        Columns to exclude from encoding and retain as-is.
    no_data_label : str, optional
        Category label representing missing data. Default "no_data".
    drop_no_data_columns : bool, optional
        If True, drop any one-hot encoded columns representing "no_data".

    Returns
    -------
    pd.DataFrame
        Dataframe with one-hot encoded features + retained id/date/drop columns.
    """
    retain_cols = id_cols + ([date_col] if date_col else []) + drop_cols
    retain_cols = [col for col in retain_cols if col in df.columns]

    encode_cols = [col for col in df.columns if col not in retain_cols]

    df_encoded = pd.get_dummies(df[encode_cols], prefix_sep="=", dtype=bool)

    if drop_no_data_columns:
        no_data_columns = [col for col in df_encoded.columns if f"={no_data_label}" in col]
        df_encoded = df_encoded.drop(columns=no_data_columns)

    # Concatenate retained columns back with encoded columns
    df_final = pd.concat([df[retain_cols].reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    return df_final

# --- Premium Functions
def generate_and_encode_temporal_trends(
    df: pd.DataFrame,
    n_dt_list: List[int],
    columns: Union[str, List[str]] = "all",
    id_cols: List[str] = [],
    datetime_col: str = "",
    flat_threshold: List[float] = [-0.01, 0.01],
    return_mode: str = "combined_only"  # options: "combined_only", "encoded_and_combined", "raw_and_combined"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate temporal lag features, encode them as trends (up/down/flat), and produce combined pattern features.

    Parameters
    ----------
    Same as original.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with only the requested feature columns.
        - DataFrame logging generated features.
    """

    assert return_mode in ["combined_only", "encoded_and_combined", "raw_and_combined"], "Invalid return_mode."
    assert len(flat_threshold) == 2, "flat_threshold must be a list of two values [lower, upper]."

    df_out = df.copy()
    all_logs = []

    lower_thresh, upper_thresh = flat_threshold

    for col in columns if isinstance(columns, list) else (
        df.select_dtypes(include=["number"]).columns.difference(id_cols + [datetime_col]).tolist() if columns == "all" else []
    ):
        pattern_cols = []
        raw_cols = []

        for n_dt in sorted(n_dt_list, reverse=True):
            suffix = f"{n_dt}_pctchange"
            df_out, log_df = generate_temporal_pct_change(
                df=df_out,
                columns=[col],
                id_cols=id_cols,
                datetime_col=datetime_col,
                n_dt=n_dt,
                suffix=suffix
            )
            pct_col = f"{col}_{suffix}"
            raw_cols.append(pct_col)

            # Create encoded trend feature
            encoded_col = f"{col}_{n_dt}_trend"
            df_out[encoded_col] = pd.cut(
                df_out[pct_col],
                bins=[-float('inf'), lower_thresh, upper_thresh, float('inf')],
                labels=["down", "flat", "up"]
            )
            
            # Replace NaN (unassigned) bins with 'no_data' before casting to string
            df_out[encoded_col] = df_out[encoded_col].cat.add_categories(['no_data']).fillna('no_data').astype(str)

            pattern_cols.append(encoded_col)

            all_logs.append(log_df)

        # Create combined pattern column
        combined_col = f"{col}_combined_trend"
        df_out[combined_col] = df_out[pattern_cols].agg("_".join, axis=1)

        # Drop columns according to return_mode
        if return_mode == "combined_only":
            df_out.drop(columns=raw_cols + pattern_cols, inplace=True)
        elif return_mode == "raw_and_combined":
            df_out.drop(columns=pattern_cols, inplace=True)
        elif return_mode == "encoded_and_combined":
            df_out.drop(columns=raw_cols, inplace=True)

    combined_log = pd.concat(all_logs, ignore_index=True)

    return df_out, combined_log

def engineer_features(
    df: pd.DataFrame,
    date_col: str,
    id_cols: List[str],
    engineer_cols: str = 'base',
    to_engineer_dates: bool = False,
    to_engineer_ratios: bool = True,
    to_engineer_lags: bool = True,
    lag_mode: str = 'raw_only',
    n_dt_list: List[int] = [1, 2, 4],
    flat_threshold: List[float] = [-0.01, 0.01]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs feature engineering steps on the input dataframe, including optional
    date features, ratio features, and temporal lag features. Returns the augmented
    dataframe and a log dictionary tracking generated features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the datetime column.
    id_cols : List[str]
        Columns used for grouping in lag feature generation.
    engineer_cols : str
        Either 'base' to restrict to numeric columns, or 'all' to process all columns.
    to_engineer_dates : bool
        Whether to generate date-derived features.
    to_engineer_ratios : bool
        Whether to generate ratio features.
    to_engineer_lags : bool
        Whether to generate lag-related features.
    lag_mode : str
        Controls lag feature output:
        - "raw_only"
        - "combined_only"
        - "encoded_and_combined"
        - "raw_and_combined"
    n_dt_list : List[int]
        List of lag intervals (only first element used if lag_mode == "raw_only").
    flat_threshold : List[float]
        Threshold range to classify flat trends for lag encoding.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        - Dataframe with engineered features appended.
        - Dictionary logging each feature generation step:
            keys = ['date_log', 'ratio_log', 'lag_log'] (depending on execution)
    """

    # Validate inputs
    if engineer_cols not in {"base", "all"}:
        raise ValueError(f"Invalid engineer_cols: {engineer_cols}. Must be 'base' or 'all'.")
    if lag_mode not in {"raw_only", "combined_only", "encoded_and_combined", "raw_and_combined"}:
        raise ValueError(f"Invalid lag_mode: {lag_mode}")

    logs: Dict[str, pd.DataFrame] = {}

    # Select columns for ratio or lag engineering
    if engineer_cols == 'base':
        cols = df.select_dtypes(include=["number"]).columns.difference([date_col] + id_cols).tolist()
    else:
        cols = 'all'

    if to_engineer_dates:
        df, date_log = extract_date_features(df, date_col)
        logs['date_log'] = date_log

    if to_engineer_ratios:
        df, ratio_log = generate_ratio_features(df, cols)
        logs['ratio_log'] = ratio_log

    if to_engineer_lags:
        if lag_mode == 'raw_only':
            n_lag = n_dt_list[0]
            df, lag_log = generate_temporal_pct_change(
                df=df,
                columns=cols,
                id_cols=id_cols,
                datetime_col=date_col,
                n_dt=n_lag,
                suffix=f"{n_lag}_pctchange"
            )
            print('raw_only')
        else:
            df, lag_log = generate_and_encode_temporal_trends(
                df=df,
                n_dt_list=n_dt_list,
                columns=cols,
                id_cols=id_cols,
                datetime_col=date_col,
                flat_threshold=flat_threshold,
                return_mode=lag_mode
            )
        logs['lag_log'] = lag_log

    return df, logs

def drop_no_data_patterns(
    df: pd.DataFrame,
    acceptable_no_data: int = 0
) -> pd.DataFrame:
    """
    Drop one-hot encoded pattern columns that contain too many 'no_data' tokens.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (after one-hot encoding), containing pattern columns
        of form '*_trend=<pattern>'.
    acceptable_no_data : int
        Maximum allowed count of 'no_data' tokens in a pattern before
        its corresponding one-hot column is dropped.
        - 0 = strict: drop any pattern column containing 'no_data'.
        - 1 = allow patterns with up to 1 'no_data'.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected columns dropped.
    """

    df_out = df.copy()

    # Identify candidate columns (one-hot encoded pattern columns)
    candidate_cols = [
        col for col in df_out.columns
        if "_trend=" in col
    ]

    cols_to_drop = []

    for col in candidate_cols:
        # Extract the pattern part from column name (after '=')
        pattern = col.split("=", maxsplit=1)[-1]

        # Count how many times 'no_data' appears in the pattern
        no_data_count = pattern.split("_").count("no_data")

        if no_data_count > acceptable_no_data:
            cols_to_drop.append(col)

    # Drop the identified columns
    df_out.drop(columns=cols_to_drop, inplace=True)

    return df_out

def encode_data(
    df: pd.DataFrame,
    date_col: str,
    id_cols: List[str],
    drop_cols: List[str] = [],
    bin_cols: Union[str, List[str]] = 'all',
    bin_quantiles: List[float] = [0, 0.25, 0.5, 0.75, 1.0],
    bin_quantile_labels: Union[List[str], None] = None,
    bin_grouping: str = "none",
    bin_dt_units: int = 20,
    to_sweep: bool = False,
    to_drop_no_data: bool = False,
    min_bin_obs: int = 20,
    min_bin_fraction: float = 0.01,
    lag_num_missing: int = 0
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Perform binning, sweeping, one-hot encoding, and optional dropping of incomplete patterns
    on the input dataframe. Logs feature engineering steps for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to encode.
    bin_cols : Union[str, List[str]]
        Columns to bin, or "all" to auto-select.
    bin_quantiles : List[float]
        Quantile thresholds for binning (e.g., [0, 0.25, 0.5, 0.75, 1.0]).
    bin_quantile_labels : Union[List[str], None]
        Optional custom labels for binned categories.
    id_cols : List[str]
        Identifier columns (e.g., ['ticker']).
    date_col : str
        Name of the datetime column.
    bin_grouping : str
        Method for bin grouping (e.g., "none", "by_id", etc.).
    bin_dt_units : int
        Time units for binning, if applicable.
    to_sweep : bool
        Whether to sweep low-count bins into a reserved label.
    to_drop_no_data : bool
        Whether to drop no_data pattern columns after encoding.
    min_bin_obs : int
        Minimum count threshold for bin sweeping.
    min_bin_fraction : float
        Minimum fraction threshold for bin sweeping.
    lag_num_missing : int
        Number of acceptable 'no_data' tokens in pattern columns before dropping.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        - Encoded dataframe ready for modeling.
        - Dictionary of logs tracking binning and sweeping steps.
    """

    logs: Dict[str, pd.DataFrame] = {}

    df, bin_log = bin_columns_flexible(
        df=df,
        columns=bin_cols,
        quantiles=bin_quantiles,
        quantile_labels=bin_quantile_labels,
        id_cols=id_cols,
        date_col=date_col,
        grouping=bin_grouping,
        n_datetime_units=bin_dt_units
    )
    logs['bin_log'] = bin_log

    if to_sweep:
        df, sweep_log = sweep_low_count_bins(
            df=df,
            columns=bin_cols,
            min_count=min_bin_obs,
            min_fraction=min_bin_fraction,
            reserved_labels=['no_data'],
            sweep_label='others'
        )
        logs['sweep_log'] = sweep_log

    ohe_df = one_hot_encode_features(
        df=df,
        id_cols=id_cols,
        date_col=date_col,
        drop_cols=drop_cols,
        no_data_label='no_data',
        drop_no_data_columns=to_drop_no_data
    )

    ohe_df = drop_no_data_patterns(
        df=ohe_df,
        acceptable_no_data=lag_num_missing
    )

    return ohe_df, logs

def engineer_pipeline(
    df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full feature engineering pipeline, including feature creation and encoding,
    with optional logging and configuration overrides.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to process.
    cfg : Any
        Configuration object containing default pipeline parameters as attributes.
    logger : Optional[Any], optional
        Logger object supporting `.log_step()`. If None, no logging is performed.
    overrides : dict, optional
        Runtime overrides for configuration parameters. Keys must match config attribute names.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Dataframe with engineered and encoded features.
        - Dictionary containing logs of both feature engineering and encoding stages.

    Notes
    -----
    Steps applied:
    1. Feature engineering (dates, ratios, lags).
    2. Feature encoding (binning, one-hot encoding, pattern pruning).

    Each step is optionally logged via logger with configuration metadata and feature logs.
    """

    def param(name: str):
        """Helper to resolve parameter from overrides or fallback to cfg."""
        return overrides.get(name, getattr(cfg, name))

    # Extract Parameters
    log_max_rows = param('log_max_rows')

    engineer_kwargs = {
        "date_col": param('date_col'),
        "id_cols": param('id_cols'),
        "engineer_cols": param('engineer_cols'),
        "to_engineer_dates": param('to_engineer_dates'),
        "to_engineer_ratios": param('to_engineer_ratios'),
        "to_engineer_lags": param('to_engineer_lags'),
        "lag_mode": param('lag_mode'),
        "n_dt_list": param('n_dt_list'),
        "flat_threshold": param('flat_threshold')
    }

    encode_kwargs = {
        "bin_cols": param('bin_cols'),
        "bin_quantiles": param('bin_quantiles'),
        "bin_quantile_labels": param('bin_quantile_labels'),
        "id_cols": param('id_cols'),
        "date_col": param('date_col'),
        "drop_cols": param('drop_cols'),
        "bin_grouping": param('bin_grouping'),
        "bin_dt_units": param('bin_dt_units'),
        "to_sweep": param('to_sweep'),
        "to_drop_no_data": param('to_drop_no_data'),
        "min_bin_obs": param('min_bin_obs'),
        "min_bin_fraction": param('min_bin_fraction'),
        "lag_num_missing": param('lag_num_missing')
    }

    # Step 1: Feature Engineering
    df, engineer_features_logs = engineer_features(df=df, **engineer_kwargs)

    if logger:
        logger.log_step(
            step_name="Feature Engineering",
            info=engineer_kwargs,
            df=list(engineer_features_logs.values()),
            max_rows=log_max_rows
        )

    # Step 2: Feature Encoding
    df, encode_data_logs = encode_data(df=df, **encode_kwargs)

    if logger:
        logger.log_step(
            step_name="Feature Encoding",
            info=encode_kwargs,
            df=list(encode_data_logs.values()),
            max_rows=log_max_rows
        )

    return df, {
        "engineer_features_logs": engineer_features_logs,
        "encode_data_logs": encode_data_logs
    }

def validate_pipeline_input(
    df: pd.DataFrame,
    id_cols: List[str],
    date_col: str,
    drop_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate and inspect whether a dataframe is ready for a binary-only pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_cols : List[str]
        Columns representing unique entity IDs.
    date_col : str
        Column containing datetime values.
    drop_cols : Optional[List[str]]
        Columns to ignore from checks (will not raise warnings).

    Returns
    -------
    Dict[str, Any]
        Structured report containing:
            - 'column_summary'
            - 'warnings'
            - 'pipeline_ready' (True/False)
            - 'report_text'
    """

    drop_cols = drop_cols or []

    report = {
        'column_summary': {},
        'warnings': [],
        'pipeline_ready': True,
        'report_text': ""
    }

    critical_columns = id_cols + [date_col]
    feature_cols = [
        col for col in df.columns
        if col not in critical_columns and col not in drop_cols
    ]

    # Check ID columns
    missing_ids = [col for col in id_cols if col not in df.columns]
    if missing_ids:
        report['warnings'].append(f"❌ Missing ID columns: {missing_ids}")
        report['pipeline_ready'] = False

    # Check date column
    if date_col not in df.columns:
        report['warnings'].append(f"❌ Date column '{date_col}' missing.")
        report['pipeline_ready'] = False
    else:
        try:
            pd.to_datetime(df[date_col])
        except Exception:
            report['warnings'].append(f"❌ Date column '{date_col}' cannot be parsed as datetime.")
            report['pipeline_ready'] = False

    # Uniqueness check
    if report['pipeline_ready']:
        duplicate_rows = df.duplicated(subset=id_cols + [date_col]).sum()
        if duplicate_rows > 0:
            report['warnings'].append(
                f"❌ {duplicate_rows} duplicate rows detected based on ID and date columns."
            )
            report['pipeline_ready'] = False

    # Feature column analysis
    binary_cols = []
    onehot_like_cols = []
    non_binary_cols = []
    null_cols = []

    for col in feature_cols:
        unique_values = df[col].dropna().unique()

        if df[col].isnull().any():
            null_cols.append(col)

        # Strict binary check
        if set(unique_values).issubset({0, 1, True, False}):
            binary_cols.append(col)
        elif "=" in col:
            onehot_like_cols.append(col)
        else:
            non_binary_cols.append(col)

    total_features = len(feature_cols)
    binary_features = len(binary_cols)
    onehot_like_features = len(onehot_like_cols)

    if non_binary_cols:
        report['warnings'].append(
            f"⚠️ {len(non_binary_cols)} columns are non-binary and may violate pipeline assumptions: {non_binary_cols}"
        )
        report['pipeline_ready'] = False

    if null_cols:
        report['warnings'].append(
            f"⚠️ {len(null_cols)} columns contain null values: {null_cols}"
        )
        report['pipeline_ready'] = False

    # Column summary
    report['column_summary'] = {
        "total_columns": total_features,
        "binary_columns": binary_features,
        "onehot_like_columns": onehot_like_features,
        "non_binary_columns": len(non_binary_cols),
        "columns_with_nulls": len(null_cols)
    }

    # Text summary
    readiness = "✅ Pipeline-ready." if report['pipeline_ready'] else "❌ Dataframe NOT pipeline-ready."

    summary_lines = [
        "🔎 Pipeline Readiness Report",
        "--------------------------------",
        f"Total feature columns (excluding ID/date/skip): {total_features}",
        f"- Binary columns: {binary_features}",
        f"- One-hot like columns: {onehot_like_features}",
        f"- Non-binary columns: {len(non_binary_cols)}",
        f"- Columns with nulls: {len(null_cols)}",
        "",
        readiness
    ]

    if report['warnings']:
        summary_lines.append("\nWarnings:")
        summary_lines.extend(report['warnings'])

    report['report_text'] = "\n".join(summary_lines)

    return report