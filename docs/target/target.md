## 🧠 Function: compute_forward_return

### 📄 What It Does
- Computes forward-looking returns over a specified horizon for each instrument group in a price dataset.
- Supports multiple return calculation modes including simple % change, log returns, volatility-adjusted returns, and smoothed forward returns.

### 🚦 When to Use
- Use when you need to generate a **target variable** for predictive modeling from time-series price data.
- Suitable for quantitative finance workflows involving:
  - Signal labeling
  - Supervised learning targets
  - Feature engineering pipelines

**Example Use Case:**  
Compute the 10-day log return for each ticker based on adjusted close prices.

**Avoid If:**
- You need rolling returns (this computes future returns).
- Your dataset lacks reliable price and date columns.

### 🔢 Inputs and Outputs

**Inputs:**

| Argument           | Type                     | Description                                                    |
|--------------------|--------------------------|----------------------------------------------------------------|
| `price_df`         | `pd.DataFrame`           | DataFrame containing time-series price data.                   |
| `n_periods`        | `int`                    | Forward horizon (in rows/periods). Must be > 0.                |
| `id_cols`          | `List[str]`              | Columns identifying instruments (default: `["ticker"]`).       |
| `date_col`         | `str`                    | Name of date column (default: `"date"`).                       |
| `target_col`       | `str`                    | Name for the output column with computed returns.              |
| `price_col`        | `str`                    | Name of the price column (default: `"adj_close"`).             |
| `return_mode`      | `Literal[...]`           | Return type: `"pct_change"`, `"log_return"`, `"vol_adjusted"`, or `"smoothed"`. |
| `vol_window`       | `Optional[int]`          | Required for `"vol_adjusted"` mode — window for volatility calc. |
| `smoothing_method` | `Literal[...]`           | Method for `"smoothed"` mode: `"median"`, `"mean"`, `"max"`, `"min"`. |

**Outputs:**

- `pd.DataFrame`: Columns include:
  - Grouping columns (`id_cols`), converted to `str`
  - Date column (`date_col`)
  - Target column (`target_col`) containing forward returns as floats

Rows where future returns cannot be computed (e.g. due to end-of-series) are dropped.

### ⚠️ Design Notes / Gotchas

- **Non-destructive:** input DataFrame is copied internally.
- **Sorting enforced:** Groups are internally sorted by date.
- **Timezone-naive dates:** Date column converted for consistency.
- **Mode-specific behavior:**
  - `"vol_adjusted"` requires `vol_window`.
  - `"smoothed"` requires valid `smoothing_method`; uses rolling window.
- Rows lacking enough forward data are removed automatically.
- Realized volatility is calculated as rolling std of simple returns, aligned to the future return.
- Supports multi-ticker datasets via `groupby`.

**Performance Note:**  
Volatility adjustment and smoothed returns use rolling windows — performance may degrade on very large datasets.

**Potential Caveats:**
- Assumes price column is strictly positive (for log returns).
- No missing data handling prior to computation (caller responsibility).
- Multi-index grouping unsupported — grouping uses ID columns directly.

### 🔗 Related Functions

- None yet linked. Possible future:
  - `compute_forward_log_return()` (if separated)
  - `compute_realized_volatility()` helper (for modularity)

### 🧪 Testing Status

- ✅ Unit tested via `pytest`.
- Covers:
  - All return modes
  - Mode-specific argument errors
  - Value correctness for simple % change and log returns
  - Edge cases: empty DataFrame, missing columns, invalid modes
  - Smoothing methods individually tested
- Volatility-adjusted return computation is tested functionally but not deeply for statistical accuracy across all parameter combinations.

## 🧠 Function: merge_features_with_returns

### 📄 What It Does
- Merges a feature dataframe with a returns (target) dataframe using a forward-looking (or configurable) as-of join.
- Handles timezone, case, and sorting normalization automatically, ensuring safe merges across real-world financial datasets.

### 🚦 When to Use
- Use when you have:
  - Features sampled on arbitrary calendar dates (e.g. accounting data, fundamental factors).
  - Returns or targets based on trading dates (e.g. next available trading day).
- Useful for preparing machine learning datasets where feature dates do not always align with trading days (due to weekends, holidays, etc.).
- Defaults to *forward* merge to avoid lookahead bias — typical in financial pipelines.

**Avoid If:**
- You require backward-looking merges without explicitly setting `direction='backward'`.
- Your IDs are numeric and require case normalization (IDs are assumed to be strings).

### 🔢 Inputs and Outputs

**Inputs:**

| Argument            | Type               | Description                                                            |
|---------------------|--------------------|------------------------------------------------------------------------|
| `feature_df`        | `pd.DataFrame`     | Feature dataframe with IDs and feature date column.                    |
| `returns_df`        | `pd.DataFrame`     | Returns dataframe with IDs and returns date column.                    |
| `id_cols`           | `List[str]`        | Columns defining the unique ID (e.g. ticker, asset).                   |
| `feature_date_col`  | `str`              | Column name in feature_df containing feature dates.                    |
| `returns_date_col`  | `str`              | Column name in returns_df containing returns dates.                    |
| `direction`         | `str`, default=`"forward"` | Merge direction: `"forward"`, `"backward"`, or `"nearest"`.     |
| `tolerance`         | `Optional[pd.Timedelta]` | Max allowed gap between feature and returns date.                 |
| `id_case`           | `str`, default=`"upper"` | Case handling for ID columns: `"upper"`, `"lower"`, `"original"`. |

**Outputs:**

- `pd.DataFrame`:
  - All columns from `feature_df`.
  - All non-ID, non-date columns from `returns_df` merged as-of.
  - NaNs in target columns preserved when no valid merge was found.

### ⚠️ Design Notes / Gotchas

- Assumes ID columns are strings or coercible to strings for case normalization.
- Feature and returns dates are converted to **timezone-naive UTC** before merging.
- Input dataframes are **copied internally** — original dataframes are not modified.
- `feature_df` and `returns_df` must each be correctly sorted by date and ID — sorting is enforced inside the function.
- Only columns not present in `id_cols` or `returns_date_col` are merged from returns dataframe.
- Time window tolerance (`tolerance`) applies to as-of matching — consider setting to avoid excessive merging across long gaps.

**Performance Considerations:**
- Suitable for reasonably sized datasets; relies on pandas `merge_asof` after sorting.

### 🔗 Related Functions

- N/A – self-contained utility function for dataset preparation.

### 🧪 Testing Status

- ✅ Unit tested via `pytest`.
- Covers:
  - All merge `direction` options.
  - All `id_case` options.
  - Forward-looking bias prevention verified.
  - Tolerance handling.
  - Error handling for:
    - Invalid `direction` and `id_case`.
    - Unparsable datetime columns.
    - Empty inputs.
- NaN preservation and merge correctness validated in typical financial dataframes.

## 🧠 Function: summarize_merge

### 📄 What It Does
- Summarizes the result of merging feature and target datasets by:
  - Counting matched and unmatched rows (based on NaNs in the target column).
  - Producing per-ID counts of matched/unmatched rows.
  - Sampling a subset of unmatched rows for inspection.

### 🚦 When to Use
- Use after performing a merge step (typically feature-data merged with returns/targets) to:
  - Quantify how many rows successfully matched to a target.
  - Diagnose unmatched rows via sampling.
- Particularly helpful in debugging incomplete merges or assessing data coverage before modeling.

**Example Use Case:**
- After merging features with returns, inspect how much of your dataset is labeled (i.e., has a valid forward return) and sample unmatched cases for error analysis.

**Avoid If:**
- You are working with datasets without NaN-masked target columns — this function assumes unmatched rows are identified via NaNs.

### 🔢 Inputs and Outputs

**Inputs:**

| Argument            | Type               | Description                                                    |
|---------------------|--------------------|----------------------------------------------------------------|
| `merged`            | `pd.DataFrame`     | Merged dataframe containing features and targets.              |
| `id_cols`           | `List[str]`        | Columns identifying unique entities (e.g., ticker).            |
| `returns_date_col`  | `str`              | Returns date column (included for interface symmetry but unused in computation). |
| `feature_date_col`  | `str`              | Feature date column name, included in the unmatched sample.    |
| `target_col`        | `str`              | Name of the target column (default: `"forward_return"`).       |
| `sample_size`       | `int`              | Max number of unmatched rows to include in the sample output.  |

**Outputs:**

- `log_summary` (`dict`):
  - Counts of `total_rows`, `matched_rows`, `unmatched_rows`.

- `counts_per_id` (`pd.DataFrame`):
  - Breakdown per ID (from `id_cols`) showing:
    - Total rows
    - Matched rows
    - Unmatched rows

- `unmatched_sample` (`pd.DataFrame`):
  - Sample of unmatched rows:
    - Includes only `id_cols` + `feature_date_col`
    - Useful for debugging missing merges.

### ⚠️ Design Notes / Gotchas

- **Rows are considered matched** if `target_col` is non-null.
- Requires presence of:
  - `target_col`
  - `feature_date_col`
  - All specified `id_cols`
- **Raises ValueError** if required columns are missing.
- Input dataframe is **copied internally**; function is non-destructive.
- `returns_date_col` is required in the function signature for symmetry but not actually used — may warrant future cleanup.

**Performance:**  
- Efficient for typical post-merge datasets; no heavy computation.

**Known Limitation:**  
- No logic to explain why unmatched rows failed — just reports and samples them.

### 🔗 Related Functions

- Typically used after:
  - `merge_features_with_returns` (or any as-of merge function producing a labeled dataset).

### 🧪 Testing Status

- ✅ Unit tested using `pytest`.
- Covered cases:
  - Typical dataset with mixed matches/unmatches.
  - Empty dataframe input.
  - Missing required columns (error raised).
  - Single-row edge case.
  - Sample size enforcement.

- No outstanding test gaps identified.

## 🧠 Function: bin_target_column

### 📄 What It Does
- Bins a continuous target column into discrete categories using one of three methods:
  - Quantile binning
  - Custom threshold binning
  - Binary encoding
- Optionally applies grouping logic for quantile binning (by IDs, datetime, or both).

### 🚦 When to Use
- Use when you need to transform a continuous target variable (e.g. returns) into discrete labels for classification tasks or analysis.
- Quantile binning helps normalize across IDs or time windows.
- Custom binning allows applying user-defined thresholds (e.g. financial return bands).
- Binary encoding quickly converts continuous values into a two-class target.

**Avoid If:**
- You need multi-label encoding outside of the three supported methods.
- Your input data lacks sufficient variability for the chosen binning method (e.g. all identical returns).

### 🔢 Inputs and Outputs

**Inputs:**

| Argument            | Type                             | Description |
|---------------------|----------------------------------|-------------|
| `df`                | `pd.DataFrame`                   | Input dataframe. |
| `binning_method`    | `'quantile'`, `'custom'`, `'binary'` | Binning strategy to use. |
| `bins`              | `List[float]`                     | Quantiles, thresholds, or binary threshold. |
| `labels`            | `List[str]`                       | Labels to assign to bins. |
| `target_col`        | `str`                             | Column to bin (default: `'return'`). |
| `id_cols`           | `Optional[List[str]]`             | ID columns for quantile grouping. |
| `date_col`          | `Optional[str]`                   | Date column for time-based grouping in quantile binning. |
| `grouping`          | `'none'`, `'ids'`, `'datetime'`, `'datetime+ids'` | Quantile binning grouping mode. |
| `n_datetime_units`  | `Optional[int]`                   | Number of rows per datetime window (quantile binning). |
| `nan_placeholder`   | `str`                             | Fallback label for NaN or failed binning (default: `'no_data'`). |

**Outputs:**

- `binned_df` (`pd.DataFrame`):  
  Original dataframe with `target_col` replaced by string labels according to binning.

- `log_df` (`pd.DataFrame`):  
  Log describing binning process, including:
  - Method used
  - Group or window ID (for quantile binning)
  - Quantiles, bins, or thresholds applied
  - Labels assigned

### ⚠️ Design Notes / Gotchas

- **Quantile binning:**  
  - Supports grouping by ID, datetime chunks, or both.  
  - Uses raw row counts for time windows (n_datetime_units) to stay agnostic to time scale.

- **Custom binning:**  
  - `len(bins) - 1` must equal `len(labels)`.  
  - Incorrect bin/label combinations fallback to `nan_placeholder`.

- **Binary binning:**  
  - Threshold is read from first element of `bins`.  
  - Labels should contain exactly two entries.

- **NaN handling:**  
  - Unbinable or NaN values receive `nan_placeholder`.

- Input dataframe is **copied** internally; original dataframe remains unmodified.

- **Performance:**  
  - Suitable for large datasets but beware of overusing fine-grained grouping.

- **Binning log:**  
  - Returned as a structured DataFrame for auditability or downstream reporting.

### 🔗 Related Functions

- None currently linked. Typically used after target creation or prior to classification modeling.

### 🧪 Testing Status

- ✅ Unit tested with `pytest`.
- Coverage includes:
  - All binning modes (quantile, custom, binary)
  - Grouping edge cases
  - Label mismatch in custom binning
  - Empty dataframe handling
  - Parameter validation (invalid method, missing inputs)
- Binary edge cases and improper bin-label alignments explicitly tested.
