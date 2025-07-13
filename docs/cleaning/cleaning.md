## 🧠 Function: clean_raw_strings

### 📄 What It Does
- Cleans raw string columns by stripping whitespace, removing control characters like `\n` and `\t`, and replacing known junk or placeholder values (e.g., `"null"`, `"none"`, empty strings) with `np.nan`.

### 🚦 When to Use
- Use this as an early step in a cleaning pipeline when ingesting messy datasets with inconsistent string formatting or placeholder null values.
- Especially useful before applying type coercion or missing value imputation.
- Not useful if columns are already cleaned or if control characters are meaningful (e.g., structured logs).

### 🔢 Inputs and Outputs
**Inputs**
- `df`: `pd.DataFrame` — The dataset containing raw text fields to clean.
- `cols`: `List[str]` or `'all'` — Columns to clean, or `'all'` to apply to all object/string columns.

**Outputs**
- `pd.DataFrame`: A copy of the original DataFrame with specified columns cleaned in-place.

### ⚠️ Design Notes / Gotchas
- Does not mutate the original DataFrame — returns a copy.
- Converts all processed string values to lowercase by default, which may not be desired for case-sensitive applications.
- Replaces known junk strings (`"null"`, `"n/a"`, `"none"`, etc.) with `np.nan`; may inadvertently remove edge-case values if user-defined semantics rely on those.
- Columns that are already `np.nan` or contain numeric/boolean types are ignored unless manually specified.
- Assumes the junk value replacement list is exhaustive for general use — users needing stricter control should extend or override the null-like set.

### 🔗 Related Functions
- `coerce_numeric_columns` — Converts cleaned strings to numeric types where possible.
- `coerce_boolean_columns` — Maps normalized strings to `True`, `False`, or `NaN`.
- `parse_datetime_column` — Robustly parses cleaned string dates to `datetime`.

### 🧪 Testing Status
- Unit tests exist and cover:
  - Normal usage across varied string formats
  - Handling of junk/null patterns
  - Behavior on edge cases (e.g., empty column, non-string input)
  - Column selection via both `'all'` and list inputs

---

## 🧠 Function: `parse_datetime_column`

### 📄 What It Does
- Converts a column in a pandas DataFrame to UTC-aware datetimes, handling various date string formats and trimming whitespace.
- Can optionally floor all parsed datetimes to midnight (i.e., set time to 00:00:00).

### 🚦 When to Use
- Use when you have a DataFrame column with potentially messy, inconsistently formatted date/time strings that need to be converted for analysis.
- Useful before time-series analysis, index setting, or any operation requiring reliable datetime types.
- Not intended for parsing multiple columns at once—apply per column as needed.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): The DataFrame containing your raw date/time data.
  - `column` (`str`): The name of the column to parse.
  - `floor_to_day` (`bool`, optional): If `True`, removes the time portion, default is `False`.
- **Outputs:**
  - Returns a `pd.Series` with parsed datetimes in UTC. Any values that can’t be parsed become `NaT` (missing timestamp).

### ⚠️ Design Notes / Gotchas
- Input DataFrame is **not** mutated.
- Raises `ValueError` if the DataFrame or specified column is missing.
- Handles leading/trailing spaces automatically.
- Works on columns already containing datetime objects (will convert or coerce to UTC as needed).
- Invalid or ambiguous formats become `NaT`, so always check for nulls after parsing.
- Performance: Suitable for typical DataFrame sizes; not specially optimized for very large data.
- Always returns timezone-aware (UTC) datetimes—even if the input didn’t have a timezone.

### 🔗 Related Functions
- [`pd.to_datetime`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) (core parsing engine)
- Other internal cleaning functions (see: `clean_and_attempt_numeric`)

### 🧪 Testing Status
- Unit tested with `pytest`:
    - Handles typical, messy, empty, and single-row inputs
    - Checks error handling for wrong input type and missing column
    - Ensures the function does not mutate the input DataFrame
    - Validates correct flooring to midnight
- No known untested edge cases; typical and edge behaviors are covered.

---

## 🧠 Function: coerce_numeric_columns

### 📄 What It Does
- Attempts to convert specified DataFrame columns to numeric types (int or float) while preserving clean numeric structure and avoiding unwanted coercion errors.

### 🚦 When to Use
- Use when ingesting messy data with numerics stored as strings (e.g., `"42"`, `"3.14"`, or `"1e5"`).
- Particularly useful after initial text cleanup (e.g., `clean_raw_strings`) and before applying numeric operations or models.
- Avoid applying to columns with mixed semantic types (e.g., ID codes that shouldn't be numeric).

### 🔢 Inputs and Outputs
**Inputs**
- `df`: `pd.DataFrame` — Input data containing columns to convert.
- `cols`: `List[str]` or `'all'` — Columns to coerce, or `'all'` for all object/string-typed columns.

**Outputs**
- `pd.DataFrame`: A copy of the input DataFrame with converted columns (others unchanged).

### ⚠️ Design Notes / Gotchas
- Internally uses `pd.to_numeric(errors="coerce")`, so non-numeric strings become `NaN`.
- Preserves `int` dtype if all values are integer-compatible and no `NaN`s are introduced.
- Columns that result in 100% `NaN` after conversion are skipped silently.
- Does not modify columns already of numeric dtype.
- Does not mutate the input DataFrame — safe for chaining.

### 🔗 Related Functions
- `clean_raw_strings` — Useful precursor for standardizing numeric string formats.
- `coerce_boolean_columns` — Similar logic for handling boolean values from strings.
- `coerce_categorical_columns` — For converting string labels to categorical type.

### 🧪 Testing Status
- Unit tested with cases for:
  - Pure integers, floats, and mixed inputs
  - Columns with partial or complete non-numeric data
  - Preservation of numeric subtypes where safe
  - Input validation errors and edge conditions

---

## 🧠 Function: coerce_boolean_columns

### 📄 What It Does
- Converts string-based representations of booleans in specified DataFrame columns to actual `boolean` dtype using common truthy/falsy patterns.

### 🚦 When to Use
- Use when ingesting raw data where boolean fields are stored as strings like `"yes"`, `"no"`, `"1"`, `"0"`, etc.
- Helpful before binary encoding, filtering logic, or model preprocessing steps.
- Avoid on columns with mixed semantic meaning (e.g., `"maybe"`, `"pending"`).

### 🔢 Inputs and Outputs
**Inputs**
- `df`: `pd.DataFrame` — The input data.
- `cols`: `List[str]` or `'all'` — Target columns to coerce, or `'all'` for all string/object-typed columns.

**Outputs**
- `pd.DataFrame`: A new DataFrame with specified columns converted to pandas nullable boolean (`dtype='boolean'`).

### ⚠️ Design Notes / Gotchas
- Recognized as `True`: `'true'`, `'t'`, `'yes'`, `'y'`, `'1'`
- Recognized as `False`: `'false'`, `'f'`, `'no'`, `'n'`, `'0'`
- Case-insensitive and trims whitespace before evaluation.
- All other values become `NaN` (missing).
- Skips columns where all values are unrecognized (100% NaNs after coercion).
- Does not mutate the original DataFrame — safe for use in pipelines.

### 🔗 Related Functions
- `coerce_numeric_columns` — Converts stringified numerics to `int` or `float`.
- `clean_raw_strings` — Preprocesses raw strings (whitespace, junk cleanup) before coercion.

### 🧪 Testing Status
- Unit tested with:
  - Mixed case and whitespace variations
  - Partial or complete coercion failures
  - Full conversion edge cases (e.g., `"TRUE"`, `" n "` → True)

---

## 🧠 Function: coerce_categorical_columns

### 📄 What It Does
- Converts specified DataFrame columns into `pandas.Categorical` dtype, with optional normalization like lowercasing and whitespace stripping.

### 🚦 When to Use
- Use when preparing string columns for efficient memory storage or categorical modeling.
- Helpful before encoding, aggregation, or groupby operations that treat category labels as discrete values.
- Avoid on columns where every value is unique (e.g. UUIDs or free-text) — no gain from converting to categorical.

### 🔢 Inputs and Outputs
**Inputs**
- `df`: `pd.DataFrame` — Input DataFrame.
- `cols`: `List[str]` or `'all'` — Columns to process; `'all'` selects all object/string-typed columns.
- `lowercase`: `bool` — If `True`, lowercases values before conversion (default: `True`).
- `strip`: `bool` — If `True`, trims leading/trailing whitespace (default: `True`).
- `drop_unused_categories`: `bool` — Whether to remove unused categories post-conversion (default: `True`).

**Outputs**
- `pd.DataFrame`: A copy of the input with specified columns converted to `CategoricalDtype`.

### ⚠️ Design Notes / Gotchas
- Leaves existing categorical columns untouched.
- Skips columns with only blank or missing values.
- If values differ only by case or spacing (e.g., `"Yes"` vs `" yes "`), normalization can consolidate them — disable `lowercase`/`strip` if needed.
- Safe to use in pipelines — does not mutate original DataFrame.

### 🔗 Related Functions
- `clean_raw_strings` — Useful before converting to categorical for heavy cleaning.
- `coerce_boolean_columns`, `coerce_numeric_columns` — Analogous helpers for type coercion.

### 🧪 Testing Status
- Unit tested with:
  - Mixed-case categories
  - Columns with missing and blank values
  - Pre-existing categorical dtypes
  - Behavior toggles (`lowercase`, `strip`, `drop_unused_categories`)

---

## 🧠 Function: `drop_high_missingness`

### 📄 What It Does
- Removes rows and columns from a DataFrame if the proportion of missing values exceeds user-defined thresholds.
- Returns the cleaned DataFrame plus a log of all dropped rows/columns with their missingness details.

### 🚦 When to Use
- Use this to clean a dataset before analysis or modeling, especially when you want to systematically drop incomplete rows/columns and track what was removed.
- Helpful when dealing with real-world data that may have structural gaps, or when you want an auditable record of cleaning steps.
- **Not intended for imputation, filling, or repairing missing values—use only for dropping.**

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): DataFrame to clean.
  - `row_thresh` (`float`): Drop any row with a fraction of missing values greater than this (default `0.9`).
  - `col_thresh` (`float`): Drop any column with a fraction of missing values greater than this (default `0.9`).
- **Outputs:**
  - Returns a tuple:
    - Cleaned `pd.DataFrame` with dropped rows/columns removed.
    - `pd.DataFrame` log listing dropped rows/columns, with fields: `type`, `row_index`, `column`, `missing_fraction`.

### ⚠️ Design Notes / Gotchas
- Input DataFrame is **not** modified; a copy is made internally.
- Thresholds must be strictly between 0 and 1; otherwise a `ValueError` is raised.
- "Dropped" means strictly greater than the threshold (`>`, not `>=`).
- Log DataFrame uses `"row"`/`"column"` in the `type` field, and `pd.NA` for irrelevant columns.
- Handles empty DataFrames and single-row/single-column edge cases.
- Function is not optimized for massive datasets, but fine for typical data cleaning use.
- Always returns both outputs, even if nothing is dropped (log will be empty).

### 🔗 Related Functions
- See also: `pandas.DataFrame.dropna`
- Typically paired with further imputation or feature engineering utilities.

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Covers: typical cases, empty frames, invalid input types and thresholds, no drops, full drops, and log structure.
    - Function is robust to edge cases and non-standard DataFrame layouts.

---

## 🧠 Function: `impute_numeric_per_group`

### 📄 What It Does
- Fills missing values in specified numeric columns, separately for each group defined by one or more identifier columns.
- Produces both an imputed DataFrame and a detailed log describing which values were filled, with what statistic, and in which group.

### 🚦 When to Use
- Use when your dataset has repeated entities (e.g., panel data by ticker, region, or customer) and you want to impute missing numeric features within each group, not globally.
- Suitable for financial panel data, multi-country series, or any grouped DataFrame where within-group imputation is preferred.
- Not intended for categorical data, single-row groups, or time-series interpolation (see pandas' interpolate for that).

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): DataFrame to process.
  - `id_cols` (`List[str]`): Column(s) to group by for per-group imputation.
  - `impute_cols` (`List[str]`): Numeric columns to fill.
  - `impute_strategy` (`str`): Statistic to use for imputation: `'median'` (default) or `'mean'`.
- **Outputs:**
  - Returns a tuple:
    - `imputed_df` (`pd.DataFrame`): DataFrame with filled missing values in `impute_cols`.
    - `imputation_log` (`pd.DataFrame`): Log table with group keys, column, fill value, count/percent filled, and success flag.

### ⚠️ Design Notes / Gotchas
- The input DataFrame is **not** mutated; imputation happens on a copy.
- Only missing values are filled; existing values are untouched.
- If an entire group is missing for a given column, imputation is not performed and the log marks it as unsuccessful.
- Raises a `ValueError` if required columns are missing or strategy is not `'median'` or `'mean'`.
- Works for both single and multi-column group keys.
- No support for time-based, ordered, or forward/backward fill—statistic is always computed on group as-is.

### 🔗 Related Functions
- See also: `pandas.DataFrame.groupby`, `pandas.DataFrame.fillna`
- May be used with `drop_high_missingness` or other cleaning utilities.

### 🧪 Testing Status
- Unit tested with `pytest`:
    - Covers typical and edge cases, invalid input, empty frames, group/column missingness, and both strategies.
    - Log output and immutability are explicitly checked.
    - No known gaps for practical usage.

---

## 🧠 Function: `fill_categorical_per_group`

### 📄 What It Does
- Fills missing values in one or more categorical columns, separately for each group defined by identifier columns, using the mode (most common value) in each group.
- Returns both the filled DataFrame and a detailed log describing what was filled (or not) and with which value.

### 🚦 When to Use
- Use when you want to impute missing categorical data by group (e.g., by ticker, customer, or segment), and want transparent tracking of what was actually filled.
- Useful in preprocessing steps before encoding or statistical modeling where missing categorical values should be resolved, but global filling is inappropriate.
- Not intended for numeric imputation or for use with columns where the mode does not make sense (e.g., high-cardinality identifiers).

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): The DataFrame containing missing categorical values.
  - `id_cols` (`List[str]`): Columns to group by (e.g., entity or instrument IDs).
  - `col_types` (`Dict[str, List[str]]`): Must contain `'categorical'` key listing all categorical columns to fill.
- **Outputs:**
  - Returns a tuple:
    - `filled_df` (`pd.DataFrame`): DataFrame with missing values in specified categorical columns filled (by group mode).
    - `fill_log` (`pd.DataFrame`): Log DataFrame with group keys, column, count/percent filled, fill value, and success flag.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame—works on a copy.
- Only fills missing values; existing non-missing values are untouched.
- If a group for a column contains only missing values, nothing is filled and this is logged as unsuccessful.
- Columns listed in `id_cols` are never filled, even if also in `categorical`.
- Expects all listed columns to exist in the DataFrame; otherwise raises a `ValueError`.
- Handles both single- and multi-column grouping keys.
- Log structure is consistent regardless of whether any values are filled.

### 🔗 Related Functions
- See also: `impute_numeric_per_group` for numeric imputation
- Related: `pandas.DataFrame.mode`, `pandas.DataFrame.fillna`

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Normal use (single/multi group, multi-column)
    - Edge cases (all missing, nothing to fill, empty DataFrame)
    - Input validation for bad columns or col_types
    - Input immutability and log structure verified

---

## 🧠 Function: `mask_high_imputation`

### 📄 What It Does
- Sets all values to missing (`NaN`) in columns and groups where the proportion of imputed values is above a specified threshold, using info from one or more imputation logs.
- Helps ensure that "over-imputed" features don't introduce spurious signal or noise in downstream analysis.

### 🚦 When to Use
- Use immediately after numeric/categorical imputation, especially if you want to prevent highly imputed columns/groups from biasing your models.
- Example: After filling missing values by group, call this function to mask out any group/column with >50% imputed entries.
- Not useful if you want to retain all imputed values or if your logs don’t provide percent-imputed info.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): The DataFrame after imputation (not mutated).
  - `log_dfs` (`List[pd.DataFrame]`): List of imputation log DataFrames (e.g., from `impute_numeric_per_group`, `fill_categorical_per_group`).
  - `id_cols` (`List[str]`): Columns that define the group (e.g., `['ticker']`).
  - `max_imputed` (`float`): Fraction above which to mask the group/column (default: `0.5`).
- **Outputs:**
  - Returns a new `pd.DataFrame` with the same shape as input, but with over-imputed values set to `pd.NA`.

### ⚠️ Design Notes / Gotchas
- Does **not** modify the input DataFrame.
- Relies on the log DataFrames to have standard columns for "success" and "percent" (e.g., `imputed_successful`/`fill_successful` and `percent_imputed`/`percent_filled`). Raises an error if these are missing.
- Ignores masking for columns not present in `df` (safe if a column was dropped earlier).
- Supports both single- and multi-column grouping.
- Edge case: If all groups/columns are below the threshold, returns the DataFrame unchanged.

### 🔗 Related Functions
- `impute_numeric_per_group`
- `fill_categorical_per_group`
- See also: any ETL pipeline step that tracks imputation statistics.

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Single/multi-group, multi-column, over/under-threshold, empty input, missing log columns, bad types, non-mutating behavior.
    - Edge cases and input validation covered.

---

## 🧠 Function: `winsorize_flexible`

### 📄 What It Does
- Applies winsorization (clipping outliers to quantile limits) to specified numeric columns in a DataFrame, with flexible grouping options for per-entity, rolling, or combined rolling/entity windowing.
- Returns the winsorized DataFrame and a detailed log of all clipped (changed) values.

### 🚦 When to Use
- Use to robustly cap outliers in numeric features, either globally or per-group (by IDs, by rolling windows, or both), e.g., in financial or scientific data pipelines.
- Essential before modeling when extreme values can distort learning or statistics, but you need to retain all rows.
- Not appropriate for categorical/text columns, or if you require robust scaling or normalization rather than clipping.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): DataFrame to winsorize.
  - `cols` (`List[str]` or `'all'`): List of numeric columns to winsorize, or `'all'` for all numeric columns.
  - `grouping` (`str`): How to group rows for separate winsorization. One of `'none'`, `'ids'`, `'datetime'`, or `'datetime+ids'`.
  - `id_cols` (`Optional[List[str]]`): Columns to use as ID/group keys (required if grouping includes `'ids'`).
  - `date_col` (`Optional[str]`): Date/datetime column for time-based windows (required for `'datetime'` groupings).
  - `n_datetime_units` (`Optional[int]`): Window size for time-based grouping (required for `'datetime'` groupings).
  - `lower_quantile`, `upper_quantile` (`float`): Lower/upper quantile to winsorize (defaults: 0.01, 0.99).
- **Outputs:**
  - Returns a tuple:
    - `df_winsorized` (`pd.DataFrame`): The DataFrame after winsorizing specified columns.
    - `log_df` (`pd.DataFrame`): Log of all actual winsorized changes (one row per value changed), including group, column, index, original value, clipped value, and a boolean flag.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame—operates on a copy.
- Only values outside the quantile range are clipped; others are untouched.
- If grouping is enabled, winsorization is performed **separately** for each group/window.
- If `cols='all'`, all numeric columns in the DataFrame are detected and used.
- Raises `ValueError` if required arguments (columns or grouping info) are missing or invalid.
- Log only includes rows that were actually changed.
- If no values were clipped, the log DataFrame is empty but with correct columns.
- Not optimized for very large, high-frequency data; batching/grouping can be tuned as needed.

### 🔗 Related Functions
- See also: `zscore_flexible` for z-scoring per group/window
- Related: `pandas.DataFrame.clip`, `scipy.stats.mstats.winsorize`
- Pairs well with `fill_categorical_per_group`, `impute_numeric_per_group` in data cleaning pipelines.

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Normal and grouped winsorization (ids, rolling, combined)
    - Edge cases (already within range, single-value, empty, invalid arguments)
    - Input immutability and log structure are explicitly checked

---

## 🧠 Function: `zscore_flexible`

### 📄 What It Does
- Computes z-scores (standard scores) for specified numeric columns in a DataFrame, supporting flexible grouping by IDs, rolling time windows, or globally.
- Returns both the transformed DataFrame and a log of the means and standard deviations used for each group and column.

### 🚦 When to Use
- Use to normalize numeric features—globally, per-entity, or in temporal blocks—while preserving the ability to audit the exact statistics applied.
- Examples: Per-ticker z-scoring in panel data; rolling normalization over time; or global standardization for modeling.
- Not suitable for non-numeric columns, non-pandas data, or if you want z-scores relative to out-of-sample data.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): DataFrame containing the numeric columns to transform.
  - `cols` (`List[str]` or `'all'`): Columns to z-score; `'all'` uses all numeric columns in the DataFrame.
  - `grouping` (`str`): One of `'none'`, `'ids'`, or `'datetime'`—determines grouping logic.
  - `id_cols` (`Optional[List[str]]`): Required if grouping is `'ids'`; columns to group by.
  - `date_col` (`Optional[str]`): Required if grouping is `'datetime'`; column to sort and chunk by time.
  - `n_datetime_units` (`Optional[int]`): Required if grouping is `'datetime'`; size of each time window.
- **Outputs:**
  - `zscored_df` (`pd.DataFrame`): DataFrame with the selected columns z-scored (others unchanged).
  - `log_df` (`pd.DataFrame`): Table of mean and std for every group and column, for reproducibility/audit.

### ⚠️ Design Notes / Gotchas
- The input DataFrame is **not** mutated; all changes are on a copy.
- If a group has std=0 or only one value, z-score for that group/column is set to NaN.
- All columns in `cols` must exist in the DataFrame, or a ValueError is raised.
- Group labels are constructed by joining the string representations of all id_cols, or by window index for datetime grouping.
- The helper `_group` column is dropped from the output.
- Not designed for extremely large, high-frequency data without further optimization.

### 🔗 Related Functions
- `pandas.DataFrame.groupby`
- `sklearn.preprocessing.StandardScaler` (for non-grouped global scaling)
- See also: `winsorize_flexible`, `drop_low_variance_columns` in the same package.

### 🧪 Testing Status
- Unit tested with `pytest`:
    - All grouping modes, edge cases (constant/empty/singleton groups), invalid arguments, and log output.
    - Output immutability and shape are checked.
    - All practical audit and edge cases are included.

---

## 🧠 Function: `robust_scale_flexible`

### 📄 What It Does
- Applies robust scaling to specified DataFrame columns, centering each by its median and scaling by a configurable quantile range (default: interquartile range, or IQR).
- Supports flexible grouping: global, by ID columns, or by rolling datetime windows, so different subsets can be independently robust-scaled.

### 🚦 When to Use
- Use when you want to normalize numeric features in a way that is resistant to outliers—especially useful for financial, trading, or other data with heavy tails.
- Suitable for both cross-sectional and time-series workflows where each group/entity should be scaled separately (e.g., per ticker, per time window).
- Not suitable for non-numeric columns or if you require scaling relative to mean/std (use z-score for that).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input DataFrame to process.
- `cols` (`List[str]` or `'all'`): Columns to robust-scale, or `'all'` for all numeric columns.
- `grouping` (`'none' | 'ids' | 'datetime'`): How to group before scaling (`'none'` = all rows, `'ids'` = by ID columns, `'datetime'` = rolling window).
- `id_cols` (`Optional[List[str]]`): Columns to use for `'ids'` grouping.
- `date_col` (`Optional[str]`): Column to use for rolling datetime grouping.
- `n_datetime_units` (`Optional[int]`): Size of rolling window for `'datetime'` grouping.
- `quantile_range` (`Tuple[float, float]` or `List[float]`): Lower and upper percentiles to use for scaling; can be tuple or list for YAML compatibility.

**Outputs:**
- `df_out` (`pd.DataFrame`): Output DataFrame with specified columns robust-scaled; all other columns untouched.
- `log_df` (`pd.DataFrame`): Log DataFrame recording group, column, scaling stats (median, quantiles, IQR), and number of non-null observations.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame; returns a copy.
- If IQR (or other specified quantile range) is zero for a group/column, all values for that group/column are set to NaN to avoid divide-by-zero.
- Only numeric columns are transformed; non-numeric columns are ignored when using `'all'`.
- Accepts both tuple and list for `quantile_range` for YAML/config compatibility.
- Will raise `ValueError` if required grouping arguments are missing, or if quantile ranges are invalid.
- Grouping is **crucial** for financial panel data—global scaling may leak future/regime info across groups.
- May not be suitable for extremely large DataFrames (uses groupby and row-wise ops).

### 🔗 Related Functions
- See also: `zscore_flexible` (z-score with flexible grouping), `quantile_rank_transform_flexible` (quantile/rank normalization)
- Related: scikit-learn `RobustScaler` (stateless, not group-aware)

### 🧪 Testing Status
- Covered by comprehensive pytest-based unit tests:
    - Global, by-ID, and rolling datetime grouping
    - Various quantile ranges (tuple and list types)
    - Edge cases: empty input, singleton groups, IQR=0, invalid args
- See `test_robust_scale_flexible.py` for coverage details.

---

## 🧠 Function: `robust_scale_flexible`

### 📄 What It Does
- Applies robust scaling to specified DataFrame columns, centering each by its median and scaling by a configurable quantile range (default: interquartile range, or IQR).
- Supports flexible grouping: global, by ID columns, or by rolling datetime windows, so different subsets can be independently robust-scaled.

### 🚦 When to Use
- Use when you want to normalize numeric features in a way that is resistant to outliers—especially useful for financial, trading, or other data with heavy tails.
- Suitable for both cross-sectional and time-series workflows where each group/entity should be scaled separately (e.g., per ticker, per time window).
- Not suitable for non-numeric columns or if you require scaling relative to mean/std (use z-score for that).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input DataFrame to process.
- `cols` (`List[str]` or `'all'`): Columns to robust-scale, or `'all'` for all numeric columns.
- `grouping` (`'none' | 'ids' | 'datetime'`): How to group before scaling (`'none'` = all rows, `'ids'` = by ID columns, `'datetime'` = rolling window).
- `id_cols` (`Optional[List[str]]`): Columns to use for `'ids'` grouping.
- `date_col` (`Optional[str]`): Column to use for rolling datetime grouping.
- `n_datetime_units` (`Optional[int]`): Size of rolling window for `'datetime'` grouping.
- `quantile_range` (`Tuple[float, float]` or `List[float]`): Lower and upper percentiles to use for scaling; can be tuple or list for YAML compatibility.

**Outputs:**
- `df_out` (`pd.DataFrame`): Output DataFrame with specified columns robust-scaled; all other columns untouched.
- `log_df` (`pd.DataFrame`): Log DataFrame recording group, column, scaling stats (median, quantiles, IQR), and number of non-null observations.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame; returns a copy.
- If IQR (or other specified quantile range) is zero for a group/column, all values for that group/column are set to NaN to avoid divide-by-zero.
- Only numeric columns are transformed; non-numeric columns are ignored when using `'all'`.
- Accepts both tuple and list for `quantile_range` for YAML/config compatibility.
- Will raise `ValueError` if required grouping arguments are missing, or if quantile ranges are invalid.
- Grouping is **crucial** for financial panel data—global scaling may leak future/regime info across groups.
- May not be suitable for extremely large DataFrames (uses groupby and row-wise ops).

### 🔗 Related Functions
- See also: `zscore_flexible` (z-score with flexible grouping), `quantile_rank_transform_flexible` (quantile/rank normalization)
- Related: scikit-learn `RobustScaler` (stateless, not group-aware)

### 🧪 Testing Status
- Covered by comprehensive pytest-based unit tests:
    - Global, by-ID, and rolling datetime grouping
    - Various quantile ranges (tuple and list types)
    - Edge cases: empty input, singleton groups, IQR=0, invalid args
- See `test_robust_scale_flexible.py` for coverage details.

---

## 🧠 Function: `quantile_rank_transform_flexible`

### 📄 What It Does
- Applies quantile or rank normalization to one or more DataFrame columns.
- Supports flexible grouping: global, by ID columns, or by rolling datetime windows.
- Returns both the transformed DataFrame and a detailed log of how each column/group was processed.

### 🚦 When to Use
- Use when you need to normalize features for quantitative models, especially for trading, finance, or ML workflows.
- Appropriate when different columns or groups should be normalized independently, e.g.:
  - Cross-sectional normalization (by date, across tickers)
  - Time-series normalization (within instrument over time)
  - Rolling block normalization (sliding time windows)
- Not suitable if you need per-row normalization or want to normalize non-numeric data.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Source DataFrame containing columns to transform.
- `cols` (`List[str]` or `'all'`): Which columns to normalize; use `'all'` for all numeric columns.
- `mode` (`'rank' | 'quantile_uniform' | 'quantile_normal'`): How to transform data:
  - `'rank'` / `'quantile_uniform'`: Map to [0, 1] via empirical CDF.
  - `'quantile_normal'`: Map to standard normal via inverse normal CDF.
- `grouping` (`'none' | 'ids' | 'datetime'`): How to group before normalization:
  - `'none'`: treat as one group
  - `'ids'`: by unique values of `id_cols`
  - `'datetime'`: rolling blocks of size `n_datetime_units` sorted by `date_col`
- `id_cols` (`Optional[List[str]]`): Required if grouping by `'ids'`.
- `date_col` (`Optional[str]`): Required if grouping by `'datetime'`.
- `n_datetime_units` (`Optional[int]`): Required if grouping by `'datetime'`.

**Outputs:**
- `df_out` (`pd.DataFrame`): DataFrame with indicated columns transformed (all other columns unchanged).
- `log_df` (`pd.DataFrame`): Log DataFrame with one row per group/column, includes group name, column, mode, and number of observations.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame.
- Non-numeric columns are ignored if `'all'` is specified.
- If a group has fewer than 2 valid (non-NaN) values, that column/group is set to NaN in the output.
- `'quantile_uniform'` and `'rank'` produce identical outputs.
- For `'quantile_normal'`, CDF values are clipped to avoid infinite values from `norm.ppf(0)` and `norm.ppf(1)`.
- Will raise `ValueError` if required grouping or column arguments are missing or incorrect.
- Log output is helpful for traceability and debugging, especially in research pipelines.
- Not designed for high-performance use on extremely large DataFrames (uses groupby and row-wise operations).

### 🔗 Related Functions
- See also: `zscore_flexible` (z-score normalization with similar API)
- Related: pandas `rank`, scikit-learn `QuantileTransformer`

### 🧪 Testing Status
- Function is covered by unit tests using pytest, including:
  - All normalization modes
  - All grouping types
  - Edge cases: empty DataFrame, single-element group, invalid inputs
- See `test_quantile_rank_transform_flexible.py` for details.

---

## 🧠 Function: `unit_vector_scale_flexible`

### 📄 What It Does
- Scales specified DataFrame columns within each group so their vector norm (L2, L1, or max) equals 1, leaving all other columns untouched.
- Supports group-wise normalization: global, by IDs, or by rolling datetime windows.

### 🚦 When to Use
- Use when you want to normalize the magnitude of numeric features—typically for cross-sectional factor construction or portfolio weight normalization—so all groups are directly comparable.
- Useful in quant finance to avoid overweighting any group/entity due to raw scale differences, or before using distance/similarity-based algorithms.
- Not appropriate if you want per-row normalization (across columns), or if non-numeric columns need scaling.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input DataFrame.
- `cols` (`List[str]` or `'all'`): Columns to scale, or `'all'` for all numeric columns.
- `grouping` (`'none' | 'ids' | 'datetime'`): How to group before scaling; required grouping args as in other flexible scaling functions.
- `id_cols` (`Optional[List[str]]`): Columns to group by for `'ids'` grouping.
- `date_col` (`Optional[str]`): Column to use for rolling datetime grouping.
- `n_datetime_units` (`Optional[int]`): Window size for rolling datetime grouping.
- `norm_type` (`'l2' | 'l1' | 'max'`): Which vector norm to use for scaling.

**Outputs:**
- `df_out` (`pd.DataFrame`): DataFrame with selected columns scaled within each group.
- `log_df` (`pd.DataFrame`): Per-group/column log, with norm type, norm value, group name, and count of non-null entries.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame.
- If all values in a group/column are zero (or NaN), outputs NaN for that group/column to avoid divide-by-zero.
- Only numeric columns are scaled if `'all'` is specified; others are ignored.
- Requires proper grouping arguments (`id_cols`, `date_col`, `n_datetime_units`) for non-global groupings.
- Designed for panel (feature-on-column) datasets—**does not** support per-row (across columns) normalization.
- Not tuned for very large DataFrames (uses groupby and applies vectorized ops).

### 🔗 Related Functions
- See also: `zscore_flexible`, `robust_scale_flexible`, `quantile_rank_transform_flexible` (all group-aware scalers with similar APIs)
- Related concept: `sklearn.preprocessing.normalize` (but not group-aware or DataFrame-friendly)

### 🧪 Testing Status
- Comprehensive pytest-based unit tests cover:
    - All supported norm types and groupings
    - Edge cases (all-zero vectors, empty DataFrames, singleton groups)
    - Exception handling for invalid input and grouping arguments
- See `test_unit_vector_scale_flexible.py` for details.

---

## 🧠 Function: `custom_apply_flexible`

### 📄 What It Does
- Applies any user-supplied function to selected columns in a DataFrame, with full support for flexible grouping (global, by IDs, or by rolling datetime windows).
- Allows passing custom keyword arguments to the user function for maximum flexibility.

### 🚦 When to Use
- Use when you need to perform a custom data transformation or cleaning step that isn’t covered by built-in functions—especially if you want it to respect the same grouping patterns as other package methods.
- Great for advanced users who want to inject domain-specific logic, special outlier handling, advanced winsorization, or novel feature engineering.
- Not needed for trivial or built-in cleaning steps (use the appropriate core function instead).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input DataFrame.
- `cols` (`List[str]` or `'all'`): Columns to which the function will be applied; `'all'` uses all numeric columns.
- `func` (`Callable[..., pd.Series]`): User-supplied function to apply; should take a Series and return a Series of same length.
- `grouping` (`'none' | 'ids' | 'datetime'`): How to group before applying the function.
- `id_cols` (`Optional[List[str]]`): Columns to use for grouping by IDs.
- `date_col` (`Optional[str]`): Column for rolling datetime grouping.
- `n_datetime_units` (`Optional[int]`): Window size for datetime grouping.
- `func_kwargs` (`Optional[Dict[str, Any]]`): Additional keyword arguments for `func`.

**Outputs:**
- `df_out` (`pd.DataFrame`): Output DataFrame with selected columns transformed by `func`. All other columns are left unchanged.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame—always returns a new DataFrame.
- `func` **must** return a Series of the same length as its input; otherwise, a `ValueError` is raised.
- Errors in user-supplied functions are wrapped as `RuntimeError` with group and column context.
- Handles missing or singleton groups gracefully.
- Non-numeric columns are ignored if `'all'` is specified.
- Grouping logic is identical to other functions in this package.
- Example: a `winsorize` function is included as a template for users who want a reference for custom transforms.

### 🔗 Related Functions
- See also: `winsorize` (reference function included for user examples)
- Related: `zscore_flexible`, `robust_scale_flexible`, `quantile_rank_transform_flexible` (all follow similar grouping and API conventions)

### 🧪 Testing Status
- Unit tests (pytest) cover:
    - All groupings and valid/invalid function returns
    - Passing custom kwargs
    - Edge cases (empty input, singleton group, non-callable, invalid returns, user error handling)
    - See `test_custom_apply_flexible.py` for full details

---

## 🧠 Function: `drop_low_variance_columns`

### 📄 What It Does
- Removes numeric columns from a DataFrame if their variance is below a specified threshold, and returns a log of which columns were dropped.

### 🚦 When to Use
- Use during data cleaning or feature selection to automatically discard features that carry virtually no signal (i.e., are constant or nearly constant).
- Especially helpful before modeling, to reduce dimensionality and avoid degenerate features.
- Not appropriate for non-numeric columns, or if you want to preserve all features regardless of variance.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): The input DataFrame.
  - `cols` (`Optional[List[str]]`): List of column names to check for low variance. If `None` or `'all'`, uses all numeric columns in `df`.
  - `variance_threshold` (`float`): Variance below which columns are dropped (default: `1e-8`).
- **Outputs:**
  - Returns a tuple:
    - `cleaned_df` (`pd.DataFrame`): The input DataFrame minus any dropped columns.
    - `variance_log` (`pd.DataFrame`): Table with each checked column, its variance, and a boolean indicating if it was dropped.

### ⚠️ Design Notes / Gotchas
- Input DataFrame is **not** mutated.
- Uses population variance (`ddof=0`), so will not exactly match `np.var(..., ddof=1)`.
- Only columns listed in `cols` and present in `df` are checked (others ignored); if `cols` is `None` or `'all'`, all numeric columns are checked.
- Raises `ValueError` if no numeric columns are found or if input types are invalid.
- Safe for DataFrames with missing values (those are ignored in variance calculation).

### 🔗 Related Functions
- See also: `pandas.DataFrame.var`
- Related: any feature selection or cleaning utility that removes degenerate columns.

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Typical, degenerate, and empty DataFrames
    - Edge cases (no numeric, constant columns, invalid input)
    - Output log and immutability are explicitly checked

---

## 🧠 Function: `drop_highly_correlated_columns`

### 📄 What It Does
- Removes numeric columns from a DataFrame if they are highly correlated (above a threshold) with any other numeric column, and logs all correlated pairs and dropped columns.

### 🚦 When to Use
- Use to automatically reduce feature redundancy before modeling or feature selection, especially if highly correlated features could cause instability or multicollinearity.
- Example: Dropping duplicate or derived signals in a quant pipeline, or removing features that are proxies for each other.
- Not suitable for categorical or string columns, or if you want to keep all features regardless of correlation.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): Input DataFrame with numeric columns.
  - `cols` (`Optional[List[str]]`): List of column names to check for high correlation. If `None` or `'all'`, uses all numeric columns in `df`.
  - `correlation_threshold` (`float`): Columns with absolute correlation above this value are considered highly correlated (default: `0.95`).
- **Outputs:**
  - Returns a tuple:
    - `cleaned_df` (`pd.DataFrame`): The DataFrame with dropped columns removed.
    - `correlation_log` (`pd.DataFrame`): Log of all correlated column pairs above the threshold, with which column was dropped.

### ⚠️ Design Notes / Gotchas
- Only the *second* column in each correlated pair is dropped (see `log['dropped_column']`).
- Operates on the upper triangle of the absolute correlation matrix—avoids duplicate reporting.
- Only columns listed in `cols` and present in `df` are checked; if `cols` is `None` or `'all'`, all numeric columns are checked.
- Raises a `ValueError` if no numeric columns are present or if input types are invalid.
- Does **not** mutate the input DataFrame.
- Does not attempt to optimize the subset of features retained for modeling—removes as soon as a high-correlation pair is found (greedy, order-dependent).

### 🔗 Related Functions
- See also: `drop_low_variance_columns`
- Related pandas tools: `pandas.DataFrame.corr`
- Consider combining with feature importance or selector utilities for model pipelines.

### 🧪 Testing Status
- Fully unit tested with `pytest`:
    - Covers perfect, partial, and no correlation; column selection; edge cases (degenerate, empty, invalid inputs); log structure and output immutability.
    - All practical and audit-critical scenarios are covered.

