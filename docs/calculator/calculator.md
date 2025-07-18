## 🧠 Function: _get_binary_counts

### 📄 What It Does
- Computes how often each binary feature in a dataframe takes the value 0 or 1 in combination with each target class.
- Returns a tidy (long-format) dataframe summarizing these counts, used for building contingency tables in association metric calculations.

### 🚦 When to Use
- Use this function when you need pairwise occurrence counts of binary features against a target column, such as for calculating support, confidence, lift, etc.
- Designed specifically for **binary features** — use only when features are strictly 0/1 or boolean (True/False).
- Not suitable for multi-class, continuous, or improperly encoded features.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - The input dataset containing all feature columns and the target column.
- `feature_cols` (`list[str]`):
  - Names of the columns to process as binary features.
  - All columns must contain only 0/1 or boolean values.
- `target_col` (`str`):
  - Name of the target column (can be categorical, binary, etc.).

**Outputs:**

- `pd.DataFrame`:
  - Long-format dataframe with columns:
    - `'feature'`: Feature name.
    - `'feature_value'`: Binary feature state (0 or 1).
    - `<target_col>`: Target class value.
    - `'count'`: Number of times this feature_value/target combination occurs.

Example:
| feature   | feature_value | target | count |
|-----------|---------------|--------|-------|
| feature1  | 1             | A      | 42    |
| feature1  | 0             | B      | 58    |
| feature2  | 1             | A      | 30    |

### ⚠️ Design Notes / Gotchas
- Raises `ValueError` if any feature contains non-binary values — strict enforcement.
- Coerces boolean feature columns to uint8 internally for consistency.
- Does **not** validate or transform the target column — assumes it's pre-cleaned.
- Returns a new dataframe — original `df` is not modified.
- Internally uses pandas `melt()` and `groupby()`, which may become a bottleneck on datasets with many features.

### 🔗 Related Functions
- `calculate_association_metrics()`: Consumes the output of `_get_binary_counts` for metrics computation.
- `get_stat_registry()`: Defines metrics that eventually use counts derived from this function.

### 🧪 Testing Status
- ✅ Unit tested with:
  - Normal binary/boolean inputs
  - Non-binary feature rejection
  - Empty dataframe
  - Single-row edge case
- ⚠️ Performance not stress-tested on extremely wide datasets

## 🧠 Function: apply_statistic_filters

### 📄 What It Does
- Applies configurable threshold-based filters to a dataframe of rule metrics.
- Marks each rule as selected (`True`) or not (`False`) based on whether it passes all filter conditions.

### 🚦 When to Use
- Use this function after computing rule-level statistics (e.g., support, lift, confidence).
- Appropriate when you need to dynamically select rules based on YAML-defined or user-provided filtering criteria.
- Supports a wide range of filter types (min, max, bounds, range) without modifying core logic.
- Not suitable for non-numeric metrics or where filtering conditions are non-threshold-based.

### 🔢 Inputs and Outputs

**Inputs:**

- `result_df` (`pd.DataFrame`):
  - Dataframe containing rule-level metrics.
  - Must include columns matching the metric names referenced in `filter_config`.

- `filter_config` (`dict[str, object]`):
  - Dictionary of filter conditions and thresholds.
  - Keys must follow the format:
    - `stat_min_<metric>` → metric ≥ threshold
    - `stat_max_<metric>` / `stat_upper_<metric>` → metric ≤ threshold
    - `stat_lower_<metric>` → metric ≥ threshold
    - `stat_bounds_<metric>` → metric ≤ lower_bound or ≥ upper_bound
    - `stat_range_<metric>` → metric inside [lower_bound, upper_bound]

**Outputs:**

- `pd.DataFrame`:
  - Copy of the input dataframe with an added `selected` boolean column.
  - Each row is marked `True` if it passes all filters, `False` otherwise.

### ⚠️ Design Notes / Gotchas
- Strictly enforces key naming conventions and validates threshold structures.
- Raises `ValueError` for:
  - Invalid filter key format.
  - Unknown filter conditions.
  - Non-existent metrics.
  - Malformed threshold values.
- Assumes metric columns in `result_df` are numeric.
- Does not mutate the input dataframe.
- Performance: fully vectorized filtering over Pandas columns — suitable for large rule sets.

### 🔗 Related Functions
- `build_filter_config()`:
  - Generates the dictionary used as `filter_config`.
- `calculate_association_metrics()`:
  - Prepares the dataframe this function filters.
- `extract_stat_filter_config()`:
  - Extracts YAML-based thresholds for use here.

### 🧪 Testing Status
- ✅ Unit tested:
  - All filter types (min, max, bounds, range).
  - Invalid inputs (bad keys, unknown metrics, malformed thresholds).
  - Empty dataframe.
  - Override conditions.
- ✅ Uses `pytest` for failure mode testing via `pytest.raises`.

## 🧠 Function: apply_statistic_filters

### 📄 What It Does
- Applies dynamic threshold-based filtering rules to a dataframe of rule metrics.
- Flags each rule as selected (`True`) or not (`False`) based on whether it passes all filters.
- Also returns a summary of rule counts and basic metric statistics.

### 🚦 When to Use
- Use after generating association rule metrics (e.g., support, lift, confidence).
- Allows applying configurable selection thresholds from YAML, CLI, or manual inputs.
- Supports simple >= / <= filters as well as interval and bounds-based conditions.
- Not suitable for non-numeric or non-threshold-based filtering tasks.

### 🔢 Inputs and Outputs

**Inputs:**

- `result_df` (`pd.DataFrame`):
  - DataFrame containing rule metrics (numeric columns).
  - Must include columns matching all metrics referenced in `filter_config`.

- `filter_config` (`dict[str, object]`):
  - Dictionary specifying filtering conditions using flat key format:
    - `stat_min_<metric>` → metric >= threshold
    - `stat_max_<metric>` or `stat_upper_<metric>` → metric <= threshold
    - `stat_bounds_<metric>` → metric <= lower_bound or >= upper_bound
    - `stat_range_<metric>` → lower_bound <= metric <= upper_bound

**Outputs:**

- `result_df` (`pd.DataFrame`):
  - Copy of input dataframe with new `selected` boolean column.
  - `True` if the rule passes all filters, else `False`.

- `summary_df` (`pd.DataFrame`):
  - Single-row dataframe with:
    - Total rule count.
    - Number of selected/unselected rules.
    - Min, max, mean for all numeric columns.

### ⚠️ Design Notes / Gotchas
- Strict key parsing: invalid key formats or unknown conditions raise `ValueError`.
- Only supports numeric columns for filtering.
- Does not mutate the input dataframe.
- Assumes metric column names align exactly with those in the filter keys.
- Fully vectorized — performance should scale to large rule sets.

### 🔗 Related Functions
- `build_filter_config()` — generates the filter dictionary from config files or overrides.
- `calculate_association_metrics()` — generates the input dataframe consumed here.
- `extract_stat_filter_config()` — YAML parsing utility for this function.

### 🧪 Testing Status
- ✅ Unit tested:
  - All filter conditions (`min`, `max`, `bounds`, `range`).
  - Incorrect keys, unknown metrics, and malformed thresholds raise errors.
  - Behavior validated on empty dataframe.
- ⚠️ Not stress-tested with extremely large rule sets (TODO).

## 🧠 Function: generate_statistics

### 📄 What It Does
- Executes the full statistics pipeline: drops non-feature columns, calculates rule-level association metrics, applies configurable selection filters, and returns annotated results.
- Designed as a wrapper to produce filtered, rule-level statistics directly from a raw input dataframe.

### 🚦 When to Use
- Use after feature engineering when you have:
  - A dataframe with binary features and a categorical/binary target column.
  - A config object defining which columns to exclude.
  - Threshold filters defined via YAML or manual overrides.
- Suitable when you need a **ready-to-use dataframe of rule metrics and filtering results**, ready for reporting or downstream analysis.
- Do not use if features are non-binary or improperly encoded.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - Full input dataframe including identifiers, dates, features, and target column.

- `cfg` (config dataclass instance):
  - Must include:
    - `id_cols`: List of ID columns to drop.
    - `date_col`: Name of date column to drop.
    - `drop_cols`: Additional non-feature columns to drop.
    - `target_col`: Name of the target column.

- `overrides` (`dict`, optional):
  - Manual threshold overrides. Values take precedence over YAML-configured defaults.

**Outputs:**

- `result_df` (`pd.DataFrame`):
  - Row-per-rule dataframe with:
    - Association metrics (support, lift, etc.).
    - A `'selected'` boolean column (rules passing all filters).

- `stat_log` (`pd.DataFrame`):
  - Single-row summary dataframe showing:
    - Total rule counts.
    - Count of selected/unselected rules.
    - Min, max, and mean values of key metrics.

### ⚠️ Design Notes / Gotchas
- Drops ID/date/drop columns silently — if target_col is mistakenly dropped, raises a ValueError.
- Relies on:
  - `calculate_association_metrics()` for metric computation.
  - `build_filter_config()` to merge YAML and manual overrides.
  - `apply_statistic_filters()` for rule selection.
- Does not mutate the input dataframe.
- Expects all features to be strictly binary.
- Performance scales with number of binary features × target classes.

### 🔗 Related Functions
- `calculate_association_metrics()`: Core metric computation.
- `apply_statistic_filters()`: Filters rules based on thresholds.
- `build_filter_config()`: Merges YAML/default filters with manual overrides.

### 🧪 Testing Status
- ✅ Unit tested:
  - Typical binary dataframe.
  - Missing target column triggers error.
  - Manual overrides correctly alter filtering.
  - Empty dataframe handled gracefully.
- ⚠️ Edge cases like extreme feature counts not yet stress-tested.
