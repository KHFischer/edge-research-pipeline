import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Union, Literal, Callable, Any
from scipy.stats import norm

def clean_raw_strings(
    df: pd.DataFrame,
    cols: Union[List[str], str]
) -> pd.DataFrame:
    """
    Clean raw string columns by stripping whitespace, removing control characters,
    and replacing known junk/null-like values with np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    cols : List[str] or 'all'
        List of column names to clean, or 'all' to apply to all object/string columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with cleaned string columns (other columns unchanged).
    """
    # Known junk patterns to treat as missing
    null_like = {"", " ", ".", "-", "nan", "n/a", "na", "null", "none", "None", "N/A", "NULL"}

    # Determine columns to process
    if cols == "all":
        target_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    elif isinstance(cols, list) and all(col in df.columns for col in cols):
        target_cols = cols
    else:
        raise ValueError("`cols` must be 'all' or a list of valid DataFrame column names.")

    df_clean = df.copy()

    for col in target_cols:
        series = df_clean[col].astype(str)

        cleaned = (
            series
            .str.strip()
            .str.replace(r"[\n\r\t]", "", regex=True)  # remove line breaks, tabs, etc.
            .str.lower()
        )
        # Replace known null-like values with np.nan
        df_clean[col] = cleaned.replace(null_like, np.nan)

    return df_clean

def parse_datetime_column(
    df: pd.DataFrame, column: str, floor_to_day: bool = False
) -> pd.Series:
    """
    Robustly parse a DataFrame column as datetimes, with optional flooring to midnight.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to parse.
    column : str
        Name of the column containing date or timestamp strings.
    floor_to_day : bool, optional (default=False)
        If True, set all times to midnight (i.e., floor datetimes to the day).

    Returns
    -------
    pd.Series
        Series of parsed datetimes (timezone-aware, UTC). Entries that cannot be parsed become NaT.

    Raises
    ------
    ValueError
        If `df` is not a DataFrame or `column` does not exist in `df`.

    Notes
    -----
    - Trims leading/trailing whitespace before parsing.
    - Handles mixed or messy date formats; coercion errors become NaT.
    - Returns a new Series; does not mutate the original DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'date': ['2024-07-01 08:15', ' 2024/07/02 ', 'bad date']})
    >>> parse_datetime_column(df, 'date')
    0   2024-07-01 08:15:00+00:00
    1   2024-07-02 00:00:00+00:00
    2                         NaT
    Name: date, dtype: datetime64[ns, UTC]
    >>> parse_datetime_column(df, 'date', floor_to_day=True)
    0   2024-07-01 00:00:00+00:00
    1   2024-07-02 00:00:00+00:00
    2                         NaT
    Name: date, dtype: datetime64[ns, UTC]
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    col_data = df[column].astype(str).str.strip()
    parsed = pd.to_datetime(col_data, errors='coerce', utc=True)

    if floor_to_day:
        parsed = parsed.dt.floor("D")

    return parsed

def coerce_numeric_columns(
    df: pd.DataFrame,
    cols: Union[List[str], str]
) -> pd.DataFrame:
    """
    Attempt to convert specified columns to numeric dtype (int or float),
    preserving subtype when possible and skipping columns that cannot be safely converted.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        List of column names to convert, or 'all' to apply to all object/string columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with converted numeric columns.

    Notes
    -----
    - Uses `pd.to_numeric(..., errors='coerce')`; unconvertible values become NaN.
    - Preserves integer type if no floats/NaNs are introduced.
    - Skips columns that result in all NaNs after attempted conversion.
    """
    if cols == "all":
        candidate_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    elif isinstance(cols, list) and all(col in df.columns for col in cols):
        candidate_cols = cols
    else:
        raise ValueError("`cols` must be 'all' or a list of valid DataFrame column names.")

    df_out = df.copy()

    for col in candidate_cols:
        coerced = pd.to_numeric(df_out[col], errors="coerce")

        # If entire column is unconvertible (all NaNs), skip
        if coerced.notna().sum() == 0:
            continue

        # If all values are integers and no NaNs, cast to int
        if pd.api.types.is_float_dtype(coerced):
            if coerced.dropna().apply(float.is_integer).all():
                if coerced.isna().any():
                    df_out[col] = coerced  # leave as float with NaNs
                else:
                    df_out[col] = coerced.astype(int)
            else:
                df_out[col] = coerced  # leave as float
        else:
            df_out[col] = coerced  # already numeric

    return df_out

def coerce_boolean_columns(
    df: pd.DataFrame,
    cols: Union[List[str], str]
) -> pd.DataFrame:
    """
    Attempt to convert specified columns to boolean dtype using common truthy/falsy string patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        List of column names to convert, or 'all' to apply to all object/string columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with converted boolean columns.

    Notes
    -----
    - Recognized True values: 'true', 't', 'yes', 'y', '1'
    - Recognized False values: 'false', 'f', 'no', 'n', '0'
    - Case-insensitive and ignores surrounding whitespace.
    - Unrecognized or missing values become NaN.
    - Columns that result in all NaNs are skipped.
    """
    TRUE_SET = {"true", "t", "yes", "y", "1"}
    FALSE_SET = {"false", "f", "no", "n", "0"}

    if cols == "all":
        candidate_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    elif isinstance(cols, list) and all(col in df.columns for col in cols):
        candidate_cols = cols
    else:
        raise ValueError("`cols` must be 'all' or a list of valid DataFrame column names.")

    df_out = df.copy()

    for col in candidate_cols:
        # Normalize text: lowercase, strip whitespace, convert to string
        normalized = df_out[col].astype(str).str.strip().str.lower()

        coerced = np.where(normalized.isin(TRUE_SET), True,
                  np.where(normalized.isin(FALSE_SET), False, np.nan))

        # Skip columns that are fully unrecognizable
        if pd.isna(coerced).all():
            continue

        df_out[col] = coerced.astype("boolean")  # nullable BooleanDtype in pandas

    return df_out

def coerce_categorical_columns(
    df: pd.DataFrame,
    cols: Union[List[str], str],
    lowercase: bool = True,
    strip: bool = True,
    drop_unused_categories: bool = True
) -> pd.DataFrame:
    """
    Convert specified columns to pandas Categorical dtype with optional normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        Column names to convert, or 'all' to apply to all object/string columns.
    lowercase : bool, default True
        Convert category labels to lowercase before casting.
    strip : bool, default True
        Strip whitespace from values before casting.
    drop_unused_categories : bool, default True
        Whether to drop unused categories after conversion.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with specified columns converted to categorical.

    Notes
    -----
    - Skips columns that contain only NaN or result in all NaN after cleaning.
    - Leaves existing Categorical columns untouched.
    """
    if cols == "all":
        candidate_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    elif isinstance(cols, list) and all(col in df.columns for col in cols):
        candidate_cols = cols
    else:
        raise ValueError("`cols` must be 'all' or a list of valid DataFrame column names.")

    df_out = df.copy()

    for col in candidate_cols:
        series = df_out[col]

        if pd.api.types.is_categorical_dtype(series):
            continue  # already categorical

        if strip:
            series = series.astype(str).str.strip()
        if lowercase:
            series = series.str.lower()

        # If all values are missing/blank, skip
        if series.replace("", pd.NA).isna().all():
            continue

        cat_series = pd.Categorical(series)
        if drop_unused_categories:
            cat_series = cat_series.remove_unused_categories()

        df_out[col] = cat_series

    return df_out
    
def clean_datetime_columns(df: pd.DataFrame, col_types: dict[str: list]) -> pd.DataFrame:
    """
    Short utility function to call parse_datetime_column() in case there are multiple datetime columns
    """
    for datetime_col in col_types['datetime']:
        df[datetime_col] = parse_datetime_column(df, datetime_col, floor_to_day=False)
    return df

def drop_high_missingness(
    df: pd.DataFrame, 
    row_thresh: float = 0.9, 
    col_thresh: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows and columns from a DataFrame that exceed specified missing value thresholds, with detailed logging.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    row_thresh : float, optional (default=0.9)
        Drop any row with a fraction of missing values greater than this threshold (range: 0 to 1).
    col_thresh : float, optional (default=0.9)
        Drop any column with a fraction of missing values greater than this threshold (range: 0 to 1).

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame after dropping high-missingness rows and columns.
    log_df : pd.DataFrame
        DataFrame logging the rows and columns that were dropped, including index/column name and missing fraction.

    Raises
    ------
    ValueError
        If input is not a DataFrame or if thresholds are not in (0, 1).

    Notes
    -----
    - Does not modify input DataFrame.
    - Logs dropped rows and columns together, with a 'type' field indicating 'row' or 'column'.
    - Uses strict 'greater than' (not >=) comparison for dropping.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, np.nan, np.nan], 'b': [np.nan, np.nan, 3]})
    >>> cleaned, log = drop_high_missingness(df, row_thresh=0.5, col_thresh=0.5)
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not (0 < row_thresh < 1) or not (0 < col_thresh < 1):
        raise ValueError("row_thresh and col_thresh must be between 0 and 1 (exclusive).")

    df_clean = df.copy()

    # Compute missing value fractions
    row_missing = df_clean.isna().mean(axis=1)
    col_missing = df_clean.isna().mean(axis=0)

    # Identify rows and columns to drop
    rows_to_drop = row_missing > row_thresh
    cols_to_drop = col_missing > col_thresh

    # Prepare logs for dropped rows
    dropped_rows_log = pd.DataFrame({
        "type": "row",
        "row_index": df_clean.index[rows_to_drop],
        "column": pd.NA,
        "missing_fraction": row_missing[rows_to_drop].values
    })

    # Prepare logs for dropped columns
    dropped_cols_log = pd.DataFrame({
        "type": "column",
        "row_index": pd.NA,
        "column": df_clean.columns[cols_to_drop],
        "missing_fraction": col_missing[cols_to_drop].values
    })

    log_df = pd.concat([dropped_rows_log, dropped_cols_log], ignore_index=True)

    # Drop rows and columns
    cleaned_df = df_clean.loc[~rows_to_drop, ~cols_to_drop]

    return cleaned_df, log_df

def impute_numeric_per_group(
    df: pd.DataFrame,
    id_cols: List[str],
    impute_cols: List[str],
    impute_strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in specified numeric columns, per group defined by identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to impute.
    id_cols : List[str]
        Columns to group by (e.g., instrument or entity identifiers).
    impute_cols : List[str]
        Numeric columns to impute.
    impute_strategy : str, optional (default='median')
        Strategy for imputation: 'median' or 'mean'.

    Returns
    -------
    imputed_df : pd.DataFrame
        DataFrame with imputed values in `impute_cols`.
    imputation_log : pd.DataFrame
        Log DataFrame with stats for each imputed column/group: count, percent imputed, fill value, and success flag.

    Raises
    ------
    ValueError
        If `impute_strategy` is not 'median' or 'mean'.
        If input columns do not exist in DataFrame.

    Notes
    -----
    - Only fills missing values; does not modify non-missing values.
    - If a group/statistic yields NaN (e.g., all values missing), leaves NaNs in place and logs as unsuccessful.
    - Does not mutate the input DataFrame.
    - Supports both single and multi-column grouping.

    Examples
    --------
    >>> df = pd.DataFrame({'id': ['A', 'A', 'B'], 'x': [1, None, 2], 'y': [None, 3, 4]})
    >>> imputed, log = impute_numeric_per_group(df, id_cols=['id'], impute_cols=['x', 'y'], impute_strategy='mean')
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not all(col in df.columns for col in id_cols):
        raise ValueError("Some `id_cols` not found in DataFrame.")
    if not all(col in df.columns for col in impute_cols):
        raise ValueError("Some `impute_cols` not found in DataFrame.")
    if impute_strategy not in {"median", "mean"}:
        raise ValueError("`impute_strategy` must be 'median' or 'mean'.")

    df_imputed = df.copy()
    logs = []
    grouped = df_imputed.groupby(id_cols)

    for group_keys, group_df in grouped:
        mask_na = group_df[impute_cols].isna()

        # Compute imputation values
        if impute_strategy == "median":
            fill_values = group_df[impute_cols].median()
        else:  # impute_strategy == "mean"
            fill_values = group_df[impute_cols].mean()

        # Prepare group keys as dict
        if isinstance(group_keys, tuple):
            group_dict = {col: val for col, val in zip(id_cols, group_keys)}
        else:
            group_dict = {id_cols[0]: group_keys}

        for col in impute_cols:
            n_total = len(group_df)
            n_imputed = mask_na[col].sum()
            pct_imputed = n_imputed / n_total if n_total else 0.0
            fill_value = fill_values[col]

            # Only fill if fill_value is not NaN
            if pd.notna(fill_value):
                df_imputed.loc[group_df.index, col] = group_df[col].fillna(fill_value)
                imputed_successful = True
            else:
                imputed_successful = False  # All values missing in group; nothing imputed

            logs.append({
                **group_dict,
                "column": col,
                "count_filled": int(n_imputed),
                "percent_filled": float(pct_imputed),
                "fill_value": fill_value,
                "fill_successful": imputed_successful,
            })

    imputation_log = pd.DataFrame(logs)

    return df_imputed, imputation_log

def fill_categorical_per_group(
    df: pd.DataFrame,
    id_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values in specified categorical columns per group of identifier columns, logging fill statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to fill.
    id_cols : List[str]
        Columns to group by (typically identifier columns).
    categorical_cols : List[str]
        List of categorical columns to fill (should not include id_cols).

    Returns
    -------
    filled_df : pd.DataFrame
        DataFrame with missing values in categorical columns filled by group mode.
    fill_log : pd.DataFrame
        DataFrame logging fill statistics per group and column:
            - Group key(s)
            - 'column' (name of column filled)
            - 'count_filled'
            - 'percent_filled'
            - 'fill_value'
            - 'fill_successful' (bool: whether a fill was possible)

    Raises
    ------
    ValueError
        If any identifier or categorical columns are missing from DataFrame.

    Notes
    -----
    - Fills only missing values; does not overwrite existing values.
    - If a group has no non-missing values for a column, leaves missing and logs as unsuccessful.
    - Does not mutate input DataFrame.
    - Columns listed in `id_cols` are not filled (should not be included in `categorical_cols`).

    Examples
    --------
    >>> df = pd.DataFrame({'id': ['A', 'A', 'B'], 'sector': ['Tech', None, 'Finance']})
    >>> filled, log = fill_categorical_per_group(df, id_cols=['id'], categorical_cols=['sector'])
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not isinstance(categorical_cols, list):
        raise ValueError("categorical_cols must be a list of column names.")
    if not all(col in df.columns for col in id_cols):
        raise ValueError("All id_cols must be present in DataFrame.")
    fill_cols = [col for col in categorical_cols if col not in id_cols]
    if not all(col in df.columns for col in fill_cols):
        raise ValueError("Some categorical columns are missing from DataFrame.")

    df_filled = df.copy()
    logs = []
    grouped = df_filled.groupby(id_cols)

    for group_keys, group_df in grouped:
        mask_na = group_df[fill_cols].isna()

        for col in fill_cols:
            mode_series = group_df[col].mode(dropna=True)
            if not mode_series.empty:
                fill_value = mode_series.iloc[0]
                fill_successful = True
            else:
                fill_value = None
                fill_successful = False

            n_total = len(group_df)
            n_filled = mask_na[col].sum()
            pct_filled = n_filled / n_total if n_total else 0.0

            # Only fill if mode exists
            if fill_successful:
                df_filled.loc[group_df.index, col] = group_df[col].fillna(fill_value)

            # Group keys as dict for log
            if isinstance(group_keys, tuple):
                group_dict = {id_col: val for id_col, val in zip(id_cols, group_keys)}
            else:
                group_dict = {id_cols[0]: group_keys}

            logs.append({
                **group_dict,
                "column": col,
                "count_filled": int(n_filled),
                "percent_filled": float(pct_filled),
                "fill_value": fill_value,
                "fill_successful": fill_successful
            })

    fill_log = pd.DataFrame(logs)
    return df_filled, fill_log

def mask_high_imputation(
    df: pd.DataFrame,
    log_dfs: List[pd.DataFrame],
    id_cols: List[str],
    max_imputed: float = 0.5
) -> pd.DataFrame:
    """
    Mask (set to NaN) values in groups and columns where the fraction of imputed values exceeds a threshold,
    based on combined imputation logs (numeric and/or categorical).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after imputation.
    log_dfs : List[pd.DataFrame]
        List of imputation log DataFrames (from numeric/categorical imputers).
    id_cols : List[str]
        Columns identifying the group (e.g., ['ticker']).
    max_imputed : float, optional (default=0.5)
        Threshold (0-1). For any group/column where percent imputed is greater than this,
        the imputed column is masked (set to NaN) for all rows in the group.

    Returns
    -------
    pd.DataFrame
        DataFrame with unreliable imputed values masked as NaN.

    Raises
    ------
    ValueError
        If input is not a DataFrame, log_dfs not a list of DataFrames, or id_cols missing from df.

    Notes
    -----
    - Input DataFrame is not mutated.
    - Handles both numeric and categorical imputation logs, standardizing column names.
    - For each group/column, if percent imputed exceeds threshold and imputation was successful,
      all values in that group/column are set to NaN.
    - Supports both single- and multi-column grouping.

    Examples
    --------
    >>> df_masked = mask_high_imputation(df, [numeric_log, categorical_log], ['ticker'], max_imputed=0.7)
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if not isinstance(log_dfs, list) or not all(isinstance(l, pd.DataFrame) for l in log_dfs):
        raise ValueError("log_dfs must be a list of pandas DataFrames.")
    if not all(col in df.columns for col in id_cols):
        raise ValueError("All id_cols must be present in df.")

    df_clean = df.copy()

    # Combine and standardize logs
    combined_log = pd.concat(log_dfs, ignore_index=True)

    # Flexible support for different log column names
    success_col = None
    for col in ["imputed_successful", "fill_successful", "success"]:
        if col in combined_log.columns:
            success_col = col
            break
    if success_col is None:
        raise ValueError("No 'imputed_successful', 'fill_successful', or 'success' column found in log.")

    percent_col = None
    for col in ["percent_imputed", "percent_filled", "percent"]:
        if col in combined_log.columns:
            percent_col = col
            break
    if percent_col is None:
        raise ValueError("No 'percent_imputed', 'percent_filled', or 'percent' column found in log.")

    # Standardize column names
    combined_log = combined_log.rename(
        columns={success_col: "success", percent_col: "percent"}
    )

    # Filter group/col pairs to mask
    to_mask = combined_log[
        (combined_log["success"]) & (combined_log["percent"] > max_imputed)
    ]

    for _, row in to_mask.iterrows():
        col = row["column"]
        if col not in df_clean.columns:
            continue  # Ignore missing columns (can happen if column dropped in cleaning)
        # Build mask for group
        if len(id_cols) == 1:
            mask = df_clean[id_cols[0]] == row[id_cols[0]]
        else:
            mask = pd.Series(True, index=df_clean.index)
            for id_col in id_cols:
                mask &= df_clean[id_col] == row[id_col]
        df_clean.loc[mask, col] = pd.NA

    return df_clean

def winsorize_flexible(
    df: pd.DataFrame,
    cols: List[str],
    grouping: str = "none",
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winsorize numeric columns with flexible grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : List[str] or 'all'
        List of columns to winsorize, or 'all' to use all numeric columns in df.
    grouping : str, optional
        "none", "ids", "datetime", or "datetime+ids" (default "none").
    id_cols : Optional[List[str]]
        Columns identifying groups, required if grouping includes 'ids'.
    date_col : Optional[str]
        Datetime column for rolling grouping, required if grouping includes 'datetime'.
    n_datetime_units : Optional[int]
        Number of rows per rolling window, required if grouping includes 'datetime'.
    lower_quantile : float, optional
        Lower winsorization quantile (default 0.01).
    upper_quantile : float, optional
        Upper winsorization quantile (default 0.99).

    Returns
    -------
    df_winsorized : pd.DataFrame
        Winsorized dataframe.
    log_df : pd.DataFrame
        Log of all changes (rows actually winsorized).

    Raises
    ------
    ValueError
        If required arguments are missing, invalid, or if columns do not exist.

    Notes
    -----
    - Only specified columns in `cols` are winsorized.
    - If `cols == 'all'`, all numeric columns in df are used.
    - Operates on a copy; does not mutate input df.
    - Log includes only rows that were actually clipped/winsorized.

    Examples
    --------
    >>> df_winz, log = winsorize_flexible(df, cols=['x', 'y'], grouping='ids', id_cols=['ticker'])
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError("cols must be a list of column names or 'all'.")
    if not all(col in df.columns for col in cols):
        raise ValueError("Some columns in 'cols' do not exist in the DataFrame.")

    df_winsorized = df.copy()
    log_entries = []

    # Validate grouping arguments
    if grouping in ["ids", "datetime+ids"] and not id_cols:
        raise ValueError("id_cols must be provided when grouping includes 'ids'.")
    if grouping in ["datetime", "datetime+ids"]:
        if date_col is None or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping includes 'datetime'.")

    # Determine group labels
    if grouping == "none":
        df_winsorized["_group"] = "all"
    elif grouping == "ids":
        df_winsorized["_group"] = df_winsorized[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_winsorized.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_winsorized = df_sorted.set_index("index").sort_index()
    elif grouping == "datetime+ids":
        df_winsorized["_group"] = None
        grouped = df_winsorized.groupby(id_cols, group_keys=False)
        chunks = []
        for name, group in grouped:
            group = group.sort_values(date_col).reset_index()
            group["_group"] = (group.index // n_datetime_units).astype(str)
            prefix = "_".join([str(v) for v in name]) if isinstance(name, tuple) else str(name)
            group["_group"] = prefix + "_window_" + group["_group"]
            chunks.append(group.set_index("index"))
        df_winsorized = pd.concat(chunks).sort_index()
    else:
        raise ValueError(f"Invalid grouping: {grouping}")

    grouped = df_winsorized.groupby("_group")

    for group_name, group_df in grouped:
        for col in cols:
            orig_values = group_df[col]
            lower = orig_values.quantile(lower_quantile)
            upper = orig_values.quantile(upper_quantile)
            clipped = np.clip(orig_values, lower, upper)

            changed = orig_values != clipped
            df_winsorized.loc[orig_values.index, col] = clipped

            log = pd.DataFrame({
                "group": group_name,
                "column": col,
                "row_index": orig_values.index,
                "original_value": orig_values,
                "winsorized_value": clipped,
                "was_winsorized": changed
            })
            log_entries.append(log[log["was_winsorized"]])

    df_winsorized = df_winsorized.drop(columns="_group")

    if log_entries:
        log_df = pd.concat(log_entries, ignore_index=True)
    else:
        log_df = pd.DataFrame(columns=[
            "group", "column", "row_index", "original_value", "winsorized_value", "was_winsorized"
        ])

    return df_winsorized, log_df

def zscore_flexible(
    df: pd.DataFrame,
    cols: List[str],
    grouping: str = "none",
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute z-scores for numeric columns, with optional grouping by IDs or by rolling time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns to z-score.
    cols : List[str] or 'all'
        List of columns to z-score, or 'all' to use all numeric columns in df.
    grouping : str, optional (default='none')
        Z-score grouping method: one of {'none', 'ids', 'datetime'}.
        - 'none': z-score globally
        - 'ids': z-score by groups defined in id_cols
        - 'datetime': z-score by rolling time windows
    id_cols : Optional[List[str]], required if grouping == 'ids'
        List of columns to group by for 'ids' grouping.
    date_col : Optional[str], required if grouping == 'datetime'
        Datetime column for rolling window grouping.
    n_datetime_units : Optional[int], required if grouping == 'datetime'
        Number of rows per time window (rolling window size).

    Returns
    -------
    zscored_df : pd.DataFrame
        DataFrame with z-scored columns (others unchanged).
    log_df : pd.DataFrame
        DataFrame logging mean/std used per group and column.

    Raises
    ------
    ValueError
        If grouping is invalid, required arguments are missing, or cols are not found in DataFrame.

    Notes
    -----
    - Only numeric columns specified in `cols` are z-scored; others are unchanged.
    - If std=0 for a group/column, result is NaN for all values in that group/column.
    - Does not mutate input DataFrame.
    - Helper column '_group' is dropped before return.

    Examples
    --------
    >>> zscored, log = zscore_flexible(df, cols=['x', 'y'], grouping='ids', id_cols=['ticker'])
    """
    if grouping not in {"none", "ids", "datetime"}:
        raise ValueError(f"Invalid grouping: {grouping}")

    # Support 'all' to use all numeric columns
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("`cols` must be a list of existing DataFrame columns or 'all'.")

    df_z = df.copy()

    # Validate grouping arguments
    if grouping == "ids":
        if not id_cols or not all(col in df_z.columns for col in id_cols):
            raise ValueError("id_cols must be provided and exist in DataFrame when grouping == 'ids'.")
    if grouping == "datetime":
        if date_col is None or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping == 'datetime'.")
        if date_col not in df_z.columns:
            raise ValueError("date_col not found in DataFrame.")

    # Determine group labels
    if grouping == "none":
        df_z["_group"] = "all"
    elif grouping == "ids":
        df_z["_group"] = df_z[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_z.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_z = df_sorted.set_index("index").sort_index()

    # Z-score and collect logs
    log_entries = []
    grouped = df_z.groupby("_group")
    for group_name, group_df in grouped:
        for col in cols:
            mean = group_df[col].mean()
            std = group_df[col].std(ddof=0)
            # Avoid division by zero: std==0 means all values identical or singleton
            if std == 0:
                z = np.nan
            else:
                z = (group_df[col] - mean) / std
            df_z.loc[group_df.index, col] = z

            log_entries.append({
                "group": group_name,
                "column": col,
                "mean": mean,
                "std": std,
            })

    df_z = df_z.drop(columns="_group")
    log_df = pd.DataFrame(log_entries)

    return df_z, log_df

def robust_scale_flexible(
    df: pd.DataFrame,
    cols: Union[List[str], Literal['all']],
    grouping: Literal['none', 'ids', 'datetime'] = 'none',
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
    quantile_range: Union[Tuple[float, float], List[float]] = (25, 75)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply robust scaling (median + quantile range) to DataFrame columns with flexible grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        Columns to robust scale. Use 'all' for all numeric columns.
    grouping : {'none', 'ids', 'datetime'}, default 'none'
        - 'none': Scale globally.
        - 'ids': Scale by groups defined in id_cols.
        - 'datetime': Scale by rolling windows of n_datetime_units on date_col.
    id_cols : list of str, optional
        Columns to group by if grouping == 'ids'.
    date_col : str, optional
        Datetime column for rolling window grouping (required if grouping == 'datetime').
    n_datetime_units : int, optional
        Window size for rolling grouping (required if grouping == 'datetime').
    quantile_range : tuple of (float, float), default (25, 75)
        Lower and upper quantiles to use for scaling (as percentiles, e.g. (25,75) = IQR).

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with indicated columns robust scaled (others unchanged).
    log_df : pd.DataFrame
        Log DataFrame with group, column, median, q_low, q_high, iqr, and n_obs for traceability.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.

    Notes
    -----
    - Does not mutate input DataFrame.
    - If iqr == 0 for a group/column, all values are set to NaN.
    - Only numeric columns are transformed; non-numeric are ignored if 'all' is passed.
    """
    valid_groupings = {'none', 'ids', 'datetime'}
    if grouping not in valid_groupings:
        raise ValueError(f"Invalid grouping: {grouping}. Must be one of {valid_groupings}.")

    # Column selection
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("`cols` must be a list of existing DataFrame columns or 'all'.")

    # Validate grouping arguments
    if grouping == "ids":
        if not id_cols or not all(col in df.columns for col in id_cols):
            raise ValueError("id_cols must be provided and exist in DataFrame when grouping == 'ids'.")
    if grouping == "datetime":
        if not date_col or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping == 'datetime'.")
        if date_col not in df.columns:
            raise ValueError("date_col not found in DataFrame.")

    # Accept tuple or list for quantile_range, for YAML configs etc.
    if isinstance(quantile_range, (list, tuple)) and len(quantile_range) == 2:
        q_low, q_high = quantile_range
        # Convert to tuple for downstream code if needed
        quantile_range = (q_low, q_high)
    else:
        raise ValueError("quantile_range must be a list or tuple of two numeric values, e.g., [25, 75]")
    
    if not (isinstance(q_low, (int, float)) and isinstance(q_high, (int, float))):
        raise ValueError("quantile_range values must be numeric (int or float).")
    if not (0 <= q_low < q_high <= 100):
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 100")

    df_out = df.copy()

    # Determine group labels
    if grouping == "none":
        df_out["_group"] = "all"
    elif grouping == "ids":
        df_out["_group"] = df_out[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_out.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_out = df_sorted.set_index("index").sort_index()

    # Transform and collect logs
    log_entries = []
    for group_name, group_df in df_out.groupby("_group"):
        for col in cols:
            x = group_df[col]
            n = x.notnull().sum()
            if n < 2:
                scaled = np.nan
                med = np.nan
                ql = np.nan
                qh = np.nan
                iqr = np.nan
            else:
                med = x.median()
                ql = np.percentile(x.dropna(), q_low)
                qh = np.percentile(x.dropna(), q_high)
                iqr = qh - ql
                if iqr == 0:
                    scaled = np.nan
                else:
                    scaled = (x - med) / iqr
            df_out.loc[group_df.index, col] = scaled
            log_entries.append({
                "group": group_name,
                "column": col,
                "median": med,
                f"q{q_low}": ql,
                f"q{q_high}": qh,
                "iqr": iqr,
                "n_obs": n
            })

    df_out = df_out.drop(columns="_group")
    log_df = pd.DataFrame(log_entries)

    return df_out, log_df
    
def quantile_rank_transform_flexible(
    df: pd.DataFrame,
    cols: Union[List[str], Literal['all']],
    mode: Literal['rank', 'quantile_uniform', 'quantile_normal'] = 'quantile_normal',
    grouping: Literal['none', 'ids', 'datetime'] = 'none',
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply quantile or rank normalization to DataFrame columns with flexible grouping options.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        Columns to transform. Use 'all' to select all numeric columns.
    mode : {'rank', 'quantile_uniform', 'quantile_normal'}, default='quantile_normal'
        - 'rank' or 'quantile_uniform': Map values to [0, 1] via empirical CDF (percentile rank).
        - 'quantile_normal': Map via empirical CDF, then to standard normal via inverse normal CDF.
    grouping : {'none', 'ids', 'datetime'}, default='none'
        - 'none': Transform globally.
        - 'ids': Group by id_cols.
        - 'datetime': Group by rolling blocks of n_datetime_units sorted by date_col.
    id_cols : list of str, optional
        Columns to group by for 'ids' grouping. Required if grouping == 'ids'.
    date_col : str, optional
        Datetime column for 'datetime' grouping. Required if grouping == 'datetime'.
    n_datetime_units : int, optional
        Window size for 'datetime' grouping. Required if grouping == 'datetime'.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with indicated columns transformed (others unchanged).
    log_df : pd.DataFrame
        Log DataFrame with group, column, mode, and n_obs for traceability.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.

    Notes
    -----
    - Does not mutate input DataFrame.
    - If n_obs < 2 in a group, transformed values are set to NaN for that group/column.
    - Numeric columns only; non-numeric cols are ignored if 'all' is passed.
    - 'quantile_uniform' and 'rank' are equivalent in output.
    """
    # Constants for numerical stability in quantile_normal mode
    _NORM_EPS = 1e-6

    # --- Input validation ---
    valid_groupings = {'none', 'ids', 'datetime'}
    valid_modes = {'rank', 'quantile_uniform', 'quantile_normal'}

    if grouping not in valid_groupings:
        raise ValueError(f"Invalid grouping: {grouping}. Must be one of {valid_groupings}.")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}.")

    # Select columns to transform
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("`cols` must be a list of existing DataFrame columns or 'all'.")

    # Validate grouping arguments
    if grouping == "ids":
        if not id_cols or not all(col in df.columns for col in id_cols):
            raise ValueError("id_cols must be provided and exist in DataFrame when grouping == 'ids'.")
    if grouping == "datetime":
        if not date_col or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping == 'datetime'.")
        if date_col not in df.columns:
            raise ValueError("date_col not found in DataFrame.")

    df_out = df.copy()

    # --- Assign group labels ---
    if grouping == "none":
        df_out["_group"] = "all"
    elif grouping == "ids":
        df_out["_group"] = df_out[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        # Sort by date_col, assign block labels by n_datetime_units
        df_sorted = df_out.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_out = df_sorted.set_index("index").sort_index()

    # --- Transformation ---
    log_entries = []
    for group_name, group_df in df_out.groupby("_group"):
        for col in cols:
            x = group_df[col]
            n = x.notnull().sum()
            if n < 2:
                transformed = np.nan
            else:
                # Use average rank to handle ties, keep NaN as is
                ranks = x.rank(method='average', na_option='keep')
                cdf = (ranks - 1) / (n - 1)
                if mode in {"rank", "quantile_uniform"}:
                    transformed = cdf
                elif mode == "quantile_normal":
                    # Map to normal; clip to avoid -inf/+inf at exact 0/1
                    transformed = norm.ppf(np.clip(cdf, _NORM_EPS, 1 - _NORM_EPS))
                else:
                    # Should never reach here due to earlier validation
                    raise RuntimeError(f"Unknown mode: {mode}")
            df_out.loc[group_df.index, col] = transformed
            log_entries.append({
                "group": group_name,
                "column": col,
                "mode": mode,
                "n_obs": n,
            })

    df_out = df_out.drop(columns="_group")
    log_df = pd.DataFrame(log_entries)

    return df_out, log_df

def unit_vector_scale_flexible(
    df: pd.DataFrame,
    cols: Union[List[str], Literal['all']],
    grouping: Literal['none', 'ids', 'datetime'] = 'none',
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
    norm_type: Literal['l2', 'l1', 'max'] = 'l2'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply unit vector (norm) scaling to selected DataFrame columns with flexible grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        Columns to scale. Use 'all' for all numeric columns.
    grouping : {'none', 'ids', 'datetime'}, default 'none'
        - 'none': Scale globally (all rows as one group).
        - 'ids': Scale by groups defined in id_cols.
        - 'datetime': Scale by rolling windows of n_datetime_units on date_col.
    id_cols : list of str, optional
        Columns to group by if grouping == 'ids'.
    date_col : str, optional
        Datetime column for rolling window grouping (required if grouping == 'datetime').
    n_datetime_units : int, optional
        Window size for rolling grouping (required if grouping == 'datetime').
    norm_type : {'l2', 'l1', 'max'}, default 'l2'
        Norm type for scaling:
            - 'l2': Euclidean (sqrt sum of squares)
            - 'l1': sum of absolute values
            - 'max': max absolute value

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with indicated columns unit-vector scaled (others unchanged).
    log_df : pd.DataFrame
        Log DataFrame with group, column, norm_type, norm_value, n_obs for traceability.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.

    Notes
    -----
    - Does not mutate input DataFrame.
    - Only numeric columns are transformed; non-numeric columns are ignored if 'all' is passed.
    - If norm is zero (all zeros or missing), result is NaN for that group/column.
    """
    valid_groupings = {'none', 'ids', 'datetime'}
    valid_norms = {'l2', 'l1', 'max'}
    if grouping not in valid_groupings:
        raise ValueError(f"Invalid grouping: {grouping}. Must be one of {valid_groupings}.")
    if norm_type not in valid_norms:
        raise ValueError(f"Invalid norm_type: {norm_type}. Must be one of {valid_norms}.")

    # Column selection
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("`cols` must be a list of existing DataFrame columns or 'all'.")

    # Validate grouping arguments
    if grouping == "ids":
        if not id_cols or not all(col in df.columns for col in id_cols):
            raise ValueError("id_cols must be provided and exist in DataFrame when grouping == 'ids'.")
    if grouping == "datetime":
        if not date_col or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping == 'datetime'.")
        if date_col not in df.columns:
            raise ValueError("date_col not found in DataFrame.")

    df_out = df.copy()

    # Determine group labels
    if grouping == "none":
        df_out["_group"] = "all"
    elif grouping == "ids":
        df_out["_group"] = df_out[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_out.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_out = df_sorted.set_index("index").sort_index()

    # Scaling and logging
    log_entries = []
    grouped = df_out.groupby("_group")
    for group_name, group_df in grouped:
        for col in cols:
            x = group_df[col]
            n = x.notnull().sum()
            if n < 1:
                scaled = np.nan
                norm_val = np.nan
            else:
                arr = x.values.astype(float)
                arr_nonan = arr[~np.isnan(arr)]
                if norm_type == 'l2':
                    norm_val = np.linalg.norm(arr_nonan, ord=2)
                elif norm_type == 'l1':
                    norm_val = np.linalg.norm(arr_nonan, ord=1)
                elif norm_type == 'max':
                    norm_val = np.max(np.abs(arr_nonan)) if arr_nonan.size else 0.0
                else:
                    raise ValueError(f"Unknown norm_type: {norm_type}")

                if norm_val == 0:
                    scaled = np.nan
                else:
                    scaled = arr / norm_val
            df_out.loc[group_df.index, col] = scaled
            log_entries.append({
                "group": group_name,
                "column": col,
                "norm_type": norm_type,
                "norm_value": norm_val,
                "n_obs": n,
            })

    df_out = df_out.drop(columns="_group")
    log_df = pd.DataFrame(log_entries)

    return df_out, log_df

def custom_apply_flexible(
    df: pd.DataFrame,
    cols: Union[List[str], Literal['all']],
    func: Callable[..., pd.Series],
    grouping: Literal['none', 'ids', 'datetime'] = 'none',
    id_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    n_datetime_units: Optional[int] = None,
    func_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply a user-supplied function to selected DataFrame columns with flexible grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str] or 'all'
        Columns to transform. Use 'all' for all numeric columns.
    func : callable
        Function to apply. Should accept a pandas Series as first argument, plus any additional kwargs.
    grouping : {'none', 'ids', 'datetime'}, default 'none'
        Grouping method.
    id_cols : list of str, optional
        Columns to group by if grouping == 'ids'.
    date_col : str, optional
        Datetime column for rolling window grouping (required if grouping == 'datetime').
    n_datetime_units : int, optional
        Window size for rolling grouping (required if grouping == 'datetime').
    func_kwargs : dict, optional
        Additional keyword arguments passed to func.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with indicated columns transformed, all others unchanged.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.

    Notes
    -----
    - Does not mutate input DataFrame.
    - func must return a pandas Series of the same length as input.
    """
    valid_groupings = {'none', 'ids', 'datetime'}
    if grouping not in valid_groupings:
        raise ValueError(f"Invalid grouping: {grouping}. Must be one of {valid_groupings}.")
    if not callable(func):
        raise ValueError("func must be callable.")

    # Column selection
    if cols == 'all':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("`cols` must be a list of existing DataFrame columns or 'all'.")

    # Validate grouping arguments
    if grouping == "ids":
        if not id_cols or not all(col in df.columns for col in id_cols):
            raise ValueError("id_cols must be provided and exist in DataFrame when grouping == 'ids'.")
    if grouping == "datetime":
        if not date_col or n_datetime_units is None:
            raise ValueError("date_col and n_datetime_units must be provided when grouping == 'datetime'.")
        if date_col not in df.columns:
            raise ValueError("date_col not found in DataFrame.")

    df_out = df.copy()
    func_kwargs = func_kwargs or {}

    # Determine group labels
    if grouping == "none":
        df_out["_group"] = "all"
    elif grouping == "ids":
        df_out["_group"] = df_out[id_cols].astype(str).agg("_".join, axis=1)
    elif grouping == "datetime":
        df_sorted = df_out.sort_values(date_col).reset_index()
        df_sorted["_group"] = (df_sorted.index // n_datetime_units).astype(str)
        df_out = df_sorted.set_index("index").sort_index()

    # Apply function and reconstruct DataFrame
    grouped = df_out.groupby("_group")
    for group_name, group_df in grouped:
        for col in cols:
            series = group_df[col]
            try:
                result = func(series, **func_kwargs)
            except Exception as e:
                raise RuntimeError(f"Error applying function to group '{group_name}', column '{col}': {e}")
            if not isinstance(result, pd.Series) or len(result) != len(series):
                raise ValueError(f"Function must return a Series of same length for group '{group_name}', column '{col}'.")
            df_out.loc[group_df.index, col] = result.values

    df_out = df_out.drop(columns="_group")
    return df_out

def winsorize(series, lower=0.01, upper=0.99):
    """
    Reference example for what a custom transformation func might look like.
    """
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))
    
def drop_low_variance_columns(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    variance_threshold: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop numeric columns from a DataFrame if their variance is below a specified threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for low-variance columns.
    cols : Optional[List[str]], optional
        List of columns to check; if None or 'all', all numeric columns are checked.
    variance_threshold : float, optional (default=1e-8)
        Variance threshold; columns with variance strictly less than this are dropped.

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame with low-variance columns dropped.
    variance_log : pd.DataFrame
        DataFrame logging the variance of each checked column and whether it was dropped.

    Raises
    ------
    ValueError
        If input is not a DataFrame or no numeric columns are found.

    Notes
    -----
    - Only checks numeric columns for variance; others are ignored.
    - Uses population variance (ddof=0) for computation.
    - Does not mutate the input DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [1, 1, 1], 'y': [1, 2, 3]})
    >>> cleaned, log = drop_low_variance_columns(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    # If cols is None or 'all', use all numeric columns in df
    if cols is None or cols == 'all':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("cols must be a list of column names or 'all'.")
        # Ensure all columns exist and are numeric
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame.")
        numeric_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]

    if not numeric_cols:
        raise ValueError("No numeric columns to check for low variance.")

    variances = df[numeric_cols].var(ddof=0)  # population variance

    variance_log = pd.DataFrame({
        "column": variances.index,
        "variance": variances.values,
        "dropped": variances < variance_threshold
    })

    cols_to_drop = variance_log.loc[variance_log["dropped"], "column"].tolist()
    cleaned_df = df.drop(columns=cols_to_drop)

    return cleaned_df, variance_log

def drop_highly_correlated_columns(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove numeric columns that are highly correlated (absolute correlation above a threshold)
    with any other numeric column, and log the correlated pairs and dropped columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for highly correlated columns.
    cols : Optional[List[str]], optional
        List of columns to check; if None or 'all', all numeric columns are checked.
    correlation_threshold : float, optional (default=0.95)
        Absolute correlation threshold above which columns are dropped.

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame with highly correlated columns dropped.
    correlation_log : pd.DataFrame
        DataFrame logging all correlated column pairs above threshold, and which column was dropped.

    Raises
    ------
    ValueError
        If input is not a DataFrame or no numeric columns are found.

    Notes
    -----
    - Only checks columns specified in `cols` if provided; else all numeric columns.
    - For each correlated pair, the second column in the pair is dropped.
    - Correlation is based on the upper triangle of the correlation matrix to avoid duplicate pairs.
    - Does not mutate the input DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 1, 1]})
    >>> cleaned, log = drop_highly_correlated_columns(df, correlation_threshold=0.9)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    # If cols is None or 'all', use all numeric columns in df
    if cols is None or cols == 'all':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("cols must be a list of column names or 'all'.")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame.")
        numeric_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]

    if not numeric_cols:
        raise ValueError("No numeric columns to check for high correlation.")

    corr_matrix = df[numeric_cols].corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    log_entries = []

    for col in upper.columns:
        high_corr = upper[col][upper[col] > correlation_threshold]
        for row_idx, corr_value in high_corr.items():
            to_drop.add(col)
            log_entries.append({
                "column_1": row_idx,
                "column_2": col,
                "correlation": corr_value,
                "dropped_column": col
            })

    cleaned_df = df.drop(columns=list(to_drop))
    correlation_log = pd.DataFrame(
        log_entries,
        columns=["column_1", "column_2", "correlation", "dropped_column"]
    )

    return cleaned_df, correlation_log