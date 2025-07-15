## 🧠 Function: `generate_ratio_features`

### 📄 What It Does
- Automatically generates all unique pairwise ratio features from selected numeric columns of a dataframe.
- Adds new ratio columns and returns a log describing each generated feature.

### 🚦 When to Use
- Use when you want to systematically engineer ratio-based features from numeric columns—common in financial datasets or relative metric modeling.
- Example: Creating ratios like `price / earnings` or `revenue / assets` across all combinations of financial variables.
- Avoid using when your dataframe has fewer than two numeric columns or when ratios between features have no meaningful interpretation.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): The input dataframe containing numeric columns.
- `columns` (`Union[str, List[str]]`, optional):
  - `"all"`: Automatically detect all numeric columns.
  - `List[str]`: Specify exact columns to use.
- `suffix` (`str`, optional): String suffix added to generated column names (default `"ratio"`).
- `max_replacement` (`float`, optional): Value used to replace infinite division results (default `1e60`).

**Outputs:**
- Returns a tuple:
  1. `df_new` (`pd.DataFrame`): Original dataframe plus new ratio columns.
  2. `log_df` (`pd.DataFrame`): Log dataframe with columns:
     - `numerator`: Column used as numerator.
     - `denominator`: Column used as denominator.
     - `new_column`: Name of generated column.

### ⚠️ Design Notes / Gotchas
- Only one direction of each ratio pair is created (e.g., `A/B`, not `B/A`).
- Handles divide-by-zero automatically; infinite results are capped to `max_replacement` (default 1e60).
- Raises:
  - `ValueError` if fewer than 2 valid columns are found.
  - `TypeError` if `columns` argument is incorrectly typed.
- Does **not** mutate the input dataframe; returns a new copy.
- Performance scales quadratically with the number of selected columns (`n*(n-1)/2` ratios).
- Relies only on Pandas and NumPy (no external dependencies).
- New columns follow the naming pattern: `<numerator>_div_<denominator>_<suffix>`.

### 🔗 Related Functions
- No direct dependencies, but integrates well with general feature engineering pipelines.

### 🧪 Testing Status
- ✅ Unit tested using `pytest`.
- Tests cover:
  - `"all"` vs manual column lists.
  - Zero-division and infinite value handling.
  - Edge cases like single-column input, non-numeric-only dataframe.
  - Custom replacement values.
- Edge cases not yet tested:
  - Very large datasets (performance scaling).

## 🧠 Function: `generate_temporal_pct_change`

### 📄 What It Does
- Adds new columns to a dataframe representing the percent change of specified numeric features over a lag of `n_dt` rows, calculated within entity-specific groups.

### 🚦 When to Use
- Use when analyzing time-series or panel data where you need to capture short- or long-term relative change patterns of numeric features over sequential records.
- Example: Generating daily or monthly percent change features within each asset or customer group.
- Avoid using when:
  - The dataframe lacks proper datetime ordering.
  - Lagged comparisons are not meaningful (e.g., unordered datasets).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataframe with numeric columns and datetime ordering.
- `columns` (`Union[str, List[str]]`):
  - `"all"`: Auto-detect all numeric columns (excluding IDs and datetime).
  - List of column names to generate features from.
- `id_cols` (`List[str]`): Columns used to group entities (e.g., asset ID, customer ID).
- `datetime_col` (`str`): Name of datetime column for intra-group ordering.
- `n_dt` (`int`): Number of rows (not time units) to lag for computing percent change.
- `suffix` (`str`): Suffix appended to the generated feature column names.

**Outputs:**
- Tuple of:
  1. `pd.DataFrame`: Copy of input dataframe with new percent change columns added.
  2. `pd.DataFrame`: Log dataframe documenting generated columns with:
     - `original_column`: Column used.
     - `new_column`: Name of generated feature.
     - `n_lag`: Lag value used.

### ⚠️ Design Notes / Gotchas
- **Row-based lagging:** `n_dt` counts rows, not fixed time intervals (flexible but requires properly ordered data).
- **In-place safety:** Returns a modified copy of the dataframe, does not alter original.
- **NaN Handling:** New columns will contain `NaN` for the first `n_dt` rows in each group due to lagging.
- **Performance:** Suitable for moderate-sized datasets. Groupby and shifting per entity group may be slow for millions of rows.
- **Error Handling:**
  - Raises `ValueError` if `datetime_col` is missing or columns are invalid.
  - Raises `TypeError` if `columns` argument is incorrectly specified.
- Assumes all groups are properly sorted using `datetime_col` for meaningful temporal calculations.

### 🔗 Related Functions
- `generate_ratio_features()`: If combining relative feature scaling and ratio-based features.

### 🧪 Testing Status
- ✅ Unit tested via `pytest`:
  - `"all"` vs manual column usage.
  - Missing columns, invalid input handling.
  - Empty dataframe rejection.
  - Correct feature naming and output shape.
  - Multi-lag param (`n_dt`) variations.
- Edge cases not yet tested:
  - High-cardinality grouping.
  - Handling of missing/incomplete time sequences.

## 🧠 Function: `extract_date_features`

### 📄 What It Does
- Adds standard calendar-based features to a dataframe using a specified datetime column.
- Features include year, quarter, month, week, weekday, and boolean flags indicating month-end, quarter-end, and year-end/start.

### 🚦 When to Use
- Use when you want to augment financial time-series data with calendar-based metadata for:
  - Modeling volume or seasonal patterns.
  - Audit reporting or diagnostic analysis.
  - Investigating reporting cycles (e.g., quarter-end effects).
- Avoid using when:
  - Working with irregular time intervals or non-calendar data.
  - Predicting price movement directly—calendar-based features often add little direct predictive value.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataframe.
- `date_col` (`str`): Name of the column containing datetime data.
- `prefix` (`str`, optional): Prefix for generated columns (default: `"dt_"`).

**Outputs:**
- Returns a **copy of the input dataframe** with additional feature columns:
  - `{prefix}year`, `{prefix}quarter`, `{prefix}month`, `{prefix}week`, `{prefix}weekday`
  - `{prefix}is_month_end`, `{prefix}is_month_start`
  - `{prefix}is_quarter_end`, `{prefix}is_quarter_start`
  - `{prefix}is_year_end`, `{prefix}is_year_start`

### ⚠️ Design Notes / Gotchas
- The input column must be convertible to `pd.Timestamp` or the function raises an error.
- If the date column is missing or invalid, a `ValueError` is raised.
- Flags are returned as 0/1 integers for compatibility with downstream models.
- The function returns a dataframe copy — input is never modified in-place.
- Week numbers follow ISO standard (1–53).
- Calendar-based features generally encode structural/reporting patterns rather than market signals.
- Be cautious applying to datasets with missing or irregular timestamps.

### 🔗 Related Functions
- None dir

## 🧠 Function: `bin_columns_flexible`

### 📄 What It Does
- Converts numeric columns into quantile-based categorical bins, with optional grouping based on IDs, datetime segments, or both.
- Supports labeling bins with custom or automatically assigned categories.

### 🚦 When to Use
- Use when you need to discretize numeric features before encoding (e.g., one-hot encoding), especially when binning should respect entity IDs, temporal windows, or both.
- Useful for preparing datasets where relative ranking (quantile binning) is more meaningful than absolute numeric values.
- Avoid when:
  - Columns are already categorical or ordinal.
  - Dataset lacks sufficient unique numeric values for quantile binning.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataframe containing numeric columns to bin.
- `columns` (`Union[str, List[str]]`):
  - `"all"` (default): Automatically detect numeric columns.
  - List of column names to bin.
- `quantiles` (`List[float]`): Quantile cut points (0-1).
- `quantile_labels` (`List[str]` or `None`): Labels for each bin. If `None`, numerical labels are used.
- `id_cols` (`List[str]` or `None`): Columns identifying entity groups (used if grouping includes `'ids'`).
- `date_col` (`str` or `None`): Column used for chronological ordering (required for datetime-based grouping).
- `grouping` (`str`): `"none"`, `"ids"`, `"datetime"`, or `"datetime+ids"`.
- `n_datetime_units` (`int` or `None`): Row count per temporal segment (required if grouping uses datetime).
- `nan_placeholder` (`str`): Value to assign to missing/unassignable bins (default `"no_data"`).

**Outputs:**
- `binned_df` (`pd.DataFrame`): Copy of input dataframe with binned columns replaced as categorical strings.
- `log_df` (`pd.DataFrame`): Log dataframe documenting binning parameters:
  - `group`: Group label.
  - `column`: Column name.
  - `quantiles`: List of quantile edges used.
  - `labels`: Bin labels assigned.

### ⚠️ Design Notes / Gotchas
- **Only numeric columns** can be binned. Non-numeric columns are ignored.
- **Grouping Modes**:
  - `"none"`: Bin globally across all data.
  - `"ids"`: Bin within each entity group.
  - `"datetime"`: Bin sequential blocks using row count (`n_datetime_units`), ignoring real time deltas.
  - `"datetime+ids"`: Bin within temporal segments separately per entity.
- Fails gracefully (using `nan_placeholder`) if a column can't be binned (e.g., constant columns).
- Does **not modify the input dataframe**; returns a modified copy.
- If using custom bin labels, ensure `len(labels) == len(quantiles) - 1`.
- Performance degrades on large datasets with complex grouping (e.g., many IDs + fine-grained time windows).

### 🔗 Related Functions
- `pd.qcut()` (used internally for quantile binning).
- Consider pairing with your pipeline’s encoding functions (e.g., one-hot encoding after binning).

### 🧪 Testing Status
- ✅ Unit tested with `pytest`:
  - Supports all grouping modes.
  - Handles invalid input and missing required parameters.
  - Handles constant columns (uses fallback bin).
  - Tested on both auto-detected and manual column lists.
- ⚠️ Edge cases not yet tested:
  - Large datasets with high-cardinality groupings.
  - Handling of sparse or highly skewed distributions.

## 🧠 Function: `sweep_low_count_bins`

### 📄 What It Does
- Collapses rare categories in specified (or all) categorical columns into a single `"others"` label.
- Returns a modified dataframe and a log of which categories were swept.

### 🚦 When to Use
- Use when you want to simplify categorical variables before encoding by removing noise from low-frequency categories.
- Especially useful after binning continuous variables or when working with naturally sparse categories.
- Avoid using when:
  - Preserving all categories is critical for modeling.
  - Data is already optimally bucketed.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataframe.
- `columns` (`Union[str, List[str]]`):
  - `"all"`: Automatically process all categorical/object columns.
  - List of specific columns to sweep.
- `min_count` (`int`, optional): Absolute minimum count below which a category is swept.
- `min_fraction` (`float`, optional): Fractional threshold (0-1). Categories representing less than this fraction are swept.
- `reserved_labels` (`List[str]`, optional): Categories that should never be swept (e.g., `"no_data"`).
- `sweep_label` (`str`, optional): Label to assign to swept categories (default: `"others"`).

**Outputs:**
- Tuple:
  1. **`df_out`** (`pd.DataFrame`): DataFrame with rare categories swept to `"others"`.
  2. **`log_df`** (`pd.DataFrame`): Log of swept categories, with columns:
     - `column`: Column name where sweep occurred.
     - `bin_swept`: Category that was swept.
     - `count_swept`: Number of occurrences replaced.

### ⚠️ Design Notes / Gotchas
- **Threshold logic:** Uses the maximum of `min_count` and `min_fraction` (scaled to row count) as sweep threshold.
- **Preserves reserved labels**: Categories listed in `reserved_labels` are always excluded from sweeping.
- **Column inference (`columns='all'`)**:
  - Targets columns of type `CategoricalDtype` or `object`.
  - Allows sweeping both post-binning columns and raw categorical features.
- **Returns a dataframe copy**: Original dataframe remains unchanged.
- **Performance**: Efficient for typical dataframe sizes; avoid applying unnecessarily to large high-cardinality columns.
- **Warning**: Does not preserve `CategoricalDtype` after sweeping—columns are treated as generic object columns.

### 🔗 Related Functions
- Binning functions (e.g., `bin_columns_flexible`), as this function typically follows binning in the pipeline.

### 🧪 Testing Status
- ✅ Unit tested:
  - Absolute and fractional thresholds.
  - Reserved label preservation.
  - `"all"` vs manual column lists.
  - Combined threshold handling.
  - Empty dataframe and invalid input handling.
  - Output log correctness.
- ⚠️ Edge cases not yet covered:
  - Very large datasets.
  - Edge behavior with mixed-type object columns.

## 🧠 Function: `one_hot_encode_features`

### 📄 What It Does
- Performs one-hot encoding on a dataframe while excluding specified ID, date, and drop columns.
- Optionally removes one-hot encoded columns representing missing data (e.g., "no_data").

### 🚦 When to Use
- Use when preparing transactional or categorical data for machine learning models or rule mining, while preserving key identifier columns (e.g., IDs, timestamps).
- Allows selective exclusion of columns that should not be encoded (e.g., metadata, non-feature columns).
- Optionally removes missing-data indicators from the encoded result, depending on modeling needs.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataframe.
- `id_cols` (`List[str]`): Columns treated as identifiers; retained but not encoded.
- `date_col` (`Optional[str]`): Optional datetime column to retain but not encode.
- `drop_cols` (`List[str]`): Columns to exclude from encoding and retain as-is.
- `no_data_label` (`str`, optional): Label representing missing data (default: `"no_data"`).
- `drop_no_data_columns` (`bool`, optional): Whether to drop one-hot columns representing missing data.

**Outputs:**
- Returns a dataframe where:
  - All non-excluded columns are one-hot encoded using pandas `get_dummies()`.
  - ID, date, and drop columns are retained in original form.
  - Columns representing `"no_data"` are optionally removed if requested.

### ⚠️ Design Notes / Gotchas
- Original columns are replaced with their one-hot encoded forms, except for retained columns.
- One-hot encoding uses `dtype=bool` for compatibility with rule mining algorithms expecting binary features.
- `no_data_label` controls how missing categories are handled post-encoding, allowing flexibility based on modeling philosophy.
- Output dataframe preserves the row count of the input dataframe.
- Relies internally on `pandas.get_dummies()`, ensuring sparse/memory-friendly one-hot encoding for moderate dataset sizes.
- Does not return sparse matrices—this is a dataframe-focused utility.
- No mutation of the input dataframe (safe copy logic).

### 🔗 Related Functions
- `pandas.get_dummies()` (used internally).
- Can be paired with binning or feature engineering functions upstream.

### 🧪 Testing Status
- ✅ Unit tested with `pytest`:
  - Exclusion of ID/date/drop columns.
  - Drop vs retain of "no_data" encoded columns.
  - Edge cases (all columns excluded, empty dataframe).
  - Structural correctness and feature presence.
- ⚠️ Large dataframe scalability and sparse matrix outputs are not currently tested.

