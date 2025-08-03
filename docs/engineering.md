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

## 🧠 Function: generate_and_encode_temporal_trends

### 📄 What It Does
- Computes percent change features over multiple temporal lags for selected columns.
- Encodes those lagged features as trend categories ("up", "flat", "down") based on a user-defined threshold.
- Optionally combines individual lag encodings into a single pattern feature (e.g., "up_down_flat").
- Returns a dataframe with features plus a log of what was generated.

---

### 🚦 When to Use
- Use when you need multi-period trend indicators from time-series numeric data, typically financial or transactional datasets.
- Especially useful for preparing transactional one-hot datasets for feature testing or rule mining.
- Example: Turning 1-month, 3-month, and 6-month returns into trend sequences like "up_up_down" for downstream modeling.

**Not recommended if:**
- Your data lacks temporal ordering or meaningful lag structure.
- You need time-weighted lags instead of row-based shifts (this function uses row-index lagging).

---

### 🔢 Inputs and Outputs

#### **Inputs**
- `df` (`pd.DataFrame`): Input dataset with numeric features, ID columns, and a datetime column.
- `n_dt_list` (`List[int]`): List of lag intervals (in row steps) to compute percent changes for.
- `columns` (`Union[str, List[str]]`): `"all"` to process all numeric columns, or a list of specific column names.
- `id_cols` (`List[str]`): Columns used to group data before computing lags.
- `datetime_col` (`str`): Column used for sorting within each group.
- `flat_threshold` (`List[float]`): Two-element list defining the range treated as "flat" change.
- `return_mode` (`str`): Controls which columns are returned. Options:
  - `"combined_only"`: Only combined pattern columns.
  - `"encoded_and_combined"`: Encoded lag columns + combined pattern.
  - `"raw_and_combined"`: Raw pct change columns + combined pattern.

#### **Outputs**
- `df_out` (`pd.DataFrame`): Input dataframe plus generated feature columns (structure depends on `return_mode`).
- `combined_log` (`pd.DataFrame`): Log dataframe documenting original columns, generated column names, and lag intervals.

---

### ⚠️ Design Notes / Gotchas
- **No in-place modification**: Input dataframe is copied.
- Lags are **row-index based**, not time-delta based.
- Trend encoding order in combined patterns is **from highest lag to lowest**, ensuring consistent feature sequencing.
- Thresholds for "flat" must be passed as a **two-element list** to ensure YAML compatibility.
- Raw lag columns and encoded columns are dropped unless explicitly requested via `return_mode`.
- Function assumes **numeric input columns** when `columns="all"` — non-numeric columns are ignored.
- Grouping by `id_cols` is required to prevent lag spillover across entities (e.g., across tickers).
- Sorting within each group is based on `datetime_col`; unsorted data will break feature logic.
- Dependent on external function: `generate_temporal_pct_change`.

---

### 🔗 Related Functions
- `generate_temporal_pct_change`: Computes raw lag percent change features (single-lag).
- Consider wrapping or pairing with dataset cleaning functions before use.

---

### 🧪 Testing Status
- ✅ Unit tested using `pytest`:
  - Covers normal cases, empty inputs, invalid parameters, edge cases (single-row), and varied lag intervals.
- Edge case not yet tested:
  - Dataframes with missing dates or irregular time intervals (but handled correctly via row-index lagging).


## 🧠 Function: engineer_features

### 📄 What It Does
- Applies configurable feature engineering steps (date-based features, ratios, temporal lag features) to a pandas dataframe.
- Returns the transformed dataframe and a detailed log of what features were created in each step.

---

### 🚦 When to Use
- Use this function when preparing tabular time-series or transactional data for modeling or rule mining.
- Useful when you want:
  - Temporal lag features based on rolling changes.
  - Derived calendar features (year, month, weekday, etc.).
  - Automated ratio feature generation between numeric columns.
- Only suitable for datasets with clear datetime and ID columns (grouping assumed).

**Do not use if:**
- Your data lacks a proper datetime index or meaningful temporal order.
- You do not want grouped, entity-based feature generation.

---

### 🔢 Inputs and Outputs

#### Inputs

| Parameter            | Type                      | Description                                                    |
|----------------------|---------------------------|----------------------------------------------------------------|
| `df`                 | `pd.DataFrame`            | Input dataset.                                                 |
| `date_col`           | `str`                     | Column containing datetime values.                             |
| `id_cols`            | `List[str]`               | Columns used for entity/group identification.                  |
| `engineer_cols`      | `str`                     | Either `"base"` (numeric columns only) or `"all"`.             |
| `to_engineer_dates`  | `bool`                    | Whether to extract date-based features.                        |
| `to_engineer_ratios` | `bool`                    | Whether to create ratio features between columns.              |
| `to_engineer_lags`   | `bool`                    | Whether to generate temporal lag-based features.               |
| `lag_mode`           | `str`                     | Controls lag output style: `"raw_only"`, `"combined_only"`, `"encoded_and_combined"`, or `"raw_and_combined"`. |
| `n_dt_list`          | `List[int]`               | Lag intervals to compute. Only first element used for `"raw_only"`. |
| `flat_threshold`     | `List[float]`             | Thresholds defining "flat" trends in encoded lags.             |

#### Outputs

| Return Value      | Type                                  | Description                                              |
|-------------------|---------------------------------------|----------------------------------------------------------|
| `df`              | `pd.DataFrame`                         | DataFrame with new engineered feature columns appended.   |
| `logs`            | `Dict[str, pd.DataFrame]`              | Dictionary of logs, keyed as `date_log`, `ratio_log`, and/or `lag_log`. Contains audit trails of feature creation. |

---

### ⚠️ Design Notes / Gotchas
- Input dataframe is **copied internally**, but returned dataframe is fully augmented with new features.
- Column selection depends on whether `engineer_cols` is `"base"` (numeric-only) or `"all"`.
- Logs allow downstream inspection of what features were generated per step.
- Function assumes meaningful entity grouping via `id_cols` when generating lags.
- If no features are generated in a step (e.g., feature type disabled), that step is omitted from the log.
- Raises `ValueError` if invalid `engineer_cols` or `lag_mode` are passed.
- Quietly handles empty dataframes — returns empty outputs without failure.

---

### 🔗 Related Functions
- `extract_date_features()`: Generates date-derived categorical features.
- `generate_ratio_features()`: Creates ratio columns from numeric columns.
- `generate_temporal_pct_change()`: Computes raw lag percent change features.
- `generate_and_encode_temporal_trends()`: Encodes lags into categorical trends or combined sequences.

---

### 🧪 Testing Status
- ✅ Unit tested via pytest:
  - Covers typical input combinations.
  - Validates error raising for invalid parameters.
  - Handles empty dataframe edge case.
  - Covers small datasets (single-row).
- Tests do **not** cover downstream model performance using these features (by design).

## 🧠 Function: encode_data

### 📄 What It Does
- Transforms a dataframe into a one-hot encoded, fully categorical format using binning, bin sweeping, and pattern filtering steps.
- Returns the transformed dataframe and stepwise logs of the feature engineering process.

---

### 🚦 When to Use
- Use this function when you need to convert continuous features into categorical or binned features for downstream binary-only models or pipelines.
- Useful for preparing data before passing it into systems that require strictly boolean features (one-hot encoding).
- Particularly suited to workflows that require tracking of binning decisions and category pruning (via logs).

**Do not use if:**
- You require continuous features in your modeling pipeline.
- Your data lacks suitable columns for binning or categorical encoding.

---

### 🔢 Inputs and Outputs

#### Inputs

| Parameter             | Type                      | Description                                                         |
|-----------------------|---------------------------|---------------------------------------------------------------------|
| `df`                  | `pd.DataFrame`            | Input dataframe to encode.                                          |
| `bin_cols`            | `Union[str, List[str]]`   | Columns to bin, or "all" to auto-select.                            |
| `bin_quantiles`       | `List[float]`             | Quantile cutoffs for binning.                                       |
| `bin_quantile_labels` | `Union[List[str], None]`  | Optional custom bin labels.                                         |
| `id_cols`             | `List[str]`               | Columns used for entity identification.                             |
| `date_col`            | `str`                     | Name of datetime column.                                            |
| `drop_cols`           | `List[str]`               | Columns to drop before one-hot encoding.                            |
| `bin_grouping`        | `str`                     | Grouping method for binning (e.g., "none", "by_id").                |
| `bin_dt_units`        | `int`                     | Units of time for time-based binning logic.                         |
| `to_sweep`            | `bool`                    | Whether to collapse rare categories into a sweep label.             |
| `to_drop_no_data`     | `bool`                    | Whether to drop columns containing incomplete pattern encodings.    |
| `min_bin_obs`         | `int`                     | Absolute count threshold for bin sweeping.                          |
| `min_bin_fraction`    | `float`                   | Relative frequency threshold for bin sweeping.                      |
| `lag_num_missing`     | `int`                     | Maximum allowed 'no_data' tokens in pattern columns before dropping.|

#### Outputs

| Return Value      | Type                                  | Description                                                        |
|-------------------|---------------------------------------|--------------------------------------------------------------------|
| `ohe_df`          | `pd.DataFrame`                         | One-hot encoded dataframe ready for modeling.                      |
| `logs`            | `Dict[str, pd.DataFrame]`              | Logs of binning and sweeping steps, keyed as `'bin_log'` and `'sweep_log'`. |

---

### ⚠️ Design Notes / Gotchas
- Assumes binning is appropriate for specified columns.
- Columns targeted for binning must be numeric if automatic detection is used (`bin_cols="all"`).
- Rare categories are swept to `'others'` only if `to_sweep=True`.
- After binning and sweeping, `pd.get_dummies()` is implicitly applied via `one_hot_encode_features`.
- Columns with excessive `'no_data'` in combined patterns can be filtered via `lag_num_missing`.
- Designed for pipelines where **only binary columns are permitted downstream**.
- Input dataframe is processed in-place and returned as a fully encoded dataframe.
- Logs track binning and optional sweeping for auditability and reproducibility.
- Depends on external modular functions:
  - `bin_columns_flexible`
  - `sweep_low_count_bins`
  - `one_hot_encode_features`
  - `drop_no_data_patterns`

---

### 🔗 Related Functions
- `bin_columns_flexible()`
- `sweep_low_count_bins()`
- `one_hot_encode_features()`
- `drop_no_data_patterns()`

---

### 🧪 Testing Status
- ✅ Unit tested via pytest:
  - Covers typical dataframe inputs.
  - Edge cases include empty dataframes and single-row dataframes.
  - Validates exception handling for invalid parameters.
  - Tests both sweep and non-sweep pathways.
- Column-specific effects (e.g., number of dropped no_data patterns) should be tested more granularly downstream.

## 🧠 Function: engineer_pipeline

### 📄 What It Does
- Executes a full feature engineering and encoding pipeline on a dataframe using a config-driven parameter system.
- Logs each stage’s output and configuration if a logger is provided.

---

### 🚦 When to Use
- Use this function to apply consistent, reproducible feature engineering (dates, ratios, lags) and encoding (binning, one-hot encoding) to tabular datasets.
- Recommended when you want:
  - Config-driven, reproducible feature pipelines.
  - Optional runtime overrides of config parameters.
  - Stepwise pipeline logging.
- Not suitable for datasets lacking a temporal component or entity grouping.

---

### 🔢 Inputs and Outputs

#### Inputs

| Parameter      | Type                      | Description                                                   |
|----------------|---------------------------|---------------------------------------------------------------|
| `df`           | `pd.DataFrame`            | Input dataframe for feature processing.                       |
| `cfg`          | `Any`                     | Config object with default pipeline parameters as attributes. |
| `logger`       | `Optional[Any]`           | Logger object supporting `.log_step()`. If None, disables logging. |
| `**overrides`  | `dict`                    | Parameter overrides (keys must match config attribute names).  |

#### Outputs

| Return Value       | Type                          | Description                                                     |
|--------------------|-------------------------------|-----------------------------------------------------------------|
| `df`               | `pd.DataFrame`                | Output dataframe with engineered and encoded features.          |
| `logs`             | `Dict[str, Any]`              | Logs of feature engineering and encoding steps, keyed separately.|

---

### ⚠️ Design Notes / Gotchas
- Parameter resolution prioritizes `**overrides` before falling back to `cfg`.
- The returned dataframe contains **all engineered and encoded features**.
- Logs are returned even if no logger is supplied (logs populated from inner functions).
- Assumes the following functions are implemented elsewhere:
  - `engineer_features()`: for feature creation.
  - `encode_data()`: for binning and encoding.
- Logging requires `logger.log_step()` accepting step name, info dict, dataframe list, and max_rows.
- Runtime errors will propagate from downstream functions (no try/except in pipeline).
- Does not mutate the input dataframe.

---

### 🔗 Related Functions
- `engineer_features()`
- `encode_data()`
- `logger.log_step()` (external dependency)

---

### 🧪 Testing Status
- ✅ Unit tested with pytest:
  - Covers normal, empty, and single-row dataframe cases.
  - Validates parameter overrides.
  - Logger calls are tested via mocks.
- Edge cases not yet tested:
  - Failure handling in downstream functions (TODO).

## 🧠 Function: validate_pipeline_input

### 📄 What It Does
- Analyzes a dataframe and assesses whether it meets strict binary-only pipeline requirements.
- Returns a structured report detailing column types, missing values, schema conformance, and pipeline readiness.

---

### 🚦 When to Use
- Use to validate input dataframes before running downstream binary-only or one-hot encoded pipelines.
- Helpful for:
  - Debugging feature engineering steps.
  - Verifying external data before integration.
  - Generating a pre-flight check report before model training.

**Do not use if:**
- Your pipeline supports continuous or multi-valued categorical features.
- You need automated data fixing (this function only reports issues, it doesn't fix them).

---

### 🔢 Inputs and Outputs

#### Inputs

| Parameter    | Type               | Description                                                              |
|--------------|--------------------|--------------------------------------------------------------------------|
| `df`         | `pd.DataFrame`     | The input dataframe to assess.                                           |
| `id_cols`    | `List[str]`        | List of columns that together uniquely identify each entity/record.      |
| `date_col`   | `str`              | Name of the column containing datetime values.                           |
| `skip_cols`  | `Optional[List[str]]` | Columns to exclude from feature validation checks (optional).         |

#### Outputs

| Return Key         | Type             | Description                                                               |
|--------------------|------------------|---------------------------------------------------------------------------|
| `column_summary`   | `Dict[str, int]` | Summary of feature column counts by type.                                 |
| `warnings`         | `List[str]`      | List of human-readable warning messages about detected schema violations. |
| `pipeline_ready`   | `bool`           | True if dataframe conforms to pipeline expectations, else False.          |
| `report_text`      | `str`            | Formatted summary report ready for logging or display.                    |

---

### ⚠️ Design Notes / Gotchas
- Assumes a **binary-only pipeline**: all feature columns (except ID/date/skip columns) must be strictly boolean or one-hot encoded.
- Automatically identifies one-hot encoded columns using column name pattern detection (`colname=value`).
- Checks for:
  - Missing or unparseable ID/date columns.
  - Duplicate rows based on ID and date.
  - Presence of continuous or invalid feature columns.
  - Nulls within feature columns.
- Reports violations but does not modify or fix the dataframe.
- Returns both structured data and human-readable report.

---

### 🔗 Related Functions
- None directly linked. Could be used prior to:
  - `engineer_pipeline()`
  - `encode_data()`
  - Custom modeling pipelines.

---

### 🧪 Testing Status
- ✅ Unit tested via pytest:
  - Validates behavior with binary, non-binary, missing ID/date columns.
  - Tests for null handling, duplicate detection, and skip column exclusion.
  - Covers normal, empty, and single-row dataframe cases.
- TODO: Extend to large datasets for performance edge case validation.
