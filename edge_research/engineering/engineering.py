import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
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
) -> pd.DataFrame:
    """
    Extracts calendar-based datetime features from a specified date column.

    Intended for volume modeling, cyclical effects, auditing, or edge-case 
    signal generation—not direct price prediction.

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
    pd.DataFrame
        Dataframe with new datetime-derived feature columns appended.

    Raises
    ------
    ValueError
        If date_col is missing or cannot be converted to datetime.
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

    # Calendar numeric features
    df_out[f"{prefix}year"] = dt_series.dt.year
    df_out[f"{prefix}quarter"] = dt_series.dt.quarter
    df_out[f"{prefix}month"] = dt_series.dt.month
    df_out[f"{prefix}week"] = dt_series.dt.isocalendar().week
    df_out[f"{prefix}weekday"] = dt_series.dt.weekday  # 0=Monday

    # Calendar flags (reporting cycle indicators)
    df_out[f"{prefix}is_month_end"] = dt_series.dt.is_month_end.astype(int)
    df_out[f"{prefix}is_month_start"] = dt_series.dt.is_month_start.astype(int)
    df_out[f"{prefix}is_quarter_end"] = dt_series.dt.is_quarter_end.astype(int)
    df_out[f"{prefix}is_quarter_start"] = dt_series.dt.is_quarter_start.astype(int)
    df_out[f"{prefix}is_year_end"] = dt_series.dt.is_year_end.astype(int)
    df_out[f"{prefix}is_year_start"] = dt_series.dt.is_year_start.astype(int)

    return df_out

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
                isinstance(df_out[col].dtype, CategoricalDtype)
                or df_out[col].dtype == object
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
