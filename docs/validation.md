## 🧠 Function: generate_time_splits

### 📄 What It Does
- Splits a pandas DataFrame into a sequence of chronological time windows, either as N equally sized periods or user-specified date ranges.
- Also returns a detailed DataFrame logging the configuration and outcome of each split, including start/end dates and row counts.

### 🚦 When to Use
- Use this when you need to divide time series data for validation, backtesting, walk-forward analysis, or any train/test scenario requiring time-aware partitioning.
- Ideal for pipelines that must avoid lookahead bias or enforce strict temporal order.
- **Don’t use:** if your data isn’t time-indexed or if you need randomized (non-chronological) splits.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): The source DataFrame, must include a datetime column.
  - `date_col` (`str`): Name of the datetime column used to split/sort.
  - `n_splits` (`int`): Number of splits/windows (used if `date_ranges` not specified).
  - `date_ranges` (`Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]`): List of (start, end) tuples for custom windows (optional).
- **Outputs:**
  - `splits` (`List[pd.DataFrame]`): List of split DataFrames, ordered by window start.
  - `log_df` (`pd.DataFrame`): Table summarizing configuration and actual content of each split (columns include indices, types, start/end, counts).

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame.
- Date column is always normalized to UTC then made timezone-naive before splitting, to avoid empty splits due to tz mismatches.
- With `date_ranges`, splits may overlap or leave gaps; the user is responsible for range coverage.
- If `n_splits` is provided and `date_ranges` is `None`, splits are equal in span and mutually exclusive.
- Throws `ValueError` if `n_splits < 1` or if a date range is malformed (`start >= end`).
- Throws `KeyError` if `date_col` missing.
- Logs a warning if `date_ranges` length differs from `n_splits`, but continues.
- Final bins always include the endpoint (inclusive on the last split).
- All edge cases (empty DataFrame, one-row splits, unsorted input) are handled.
- Performance: Underlying DataFrame is always sorted by the date column before splitting.

### 🔗 Related Functions
- See also: `generate_fraction_splits`, which supports splitting by fractions or with explicit overlap.
- Downstream: Any pipeline modules expecting time windows (e.g., validation tests, walk-forward, rolling statistics).

### 🧪 Testing Status
- Thoroughly unit tested (see `test_generate_time_splits`) including:
  - Typical usage (equal splits, overlapping custom ranges)
  - Empty, degenerate, and one-row DataFrames
  - Invalid or malformed input scenarios (raises and warnings)
- TODO: Test with large, irregular, or real-world time series for performance and corner cases.

## 🧠 Function: generate_fraction_splits

### 📄 What It Does
- Splits a pandas DataFrame into a series of chronologically ordered windows (subsets), where each window covers a specified fraction of the total time span. 
- Supports overlapping and non-overlapping windows, and returns both the split DataFrames and a log DataFrame summarizing each split.

### 🚦 When to Use
- Use when you need to partition a time series or event data set by rolling, sliding, or custom-defined time windows for backtesting, cross-validation, or walk-forward analysis.
- Works well for both classical train/test/validation workflows and more advanced strategies like sliding-window model validation.
- **Not suitable:** if your data isn't time-indexed, if your splits should be random rather than chronological, or if you require splits based on fixed counts rather than date spans.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `df` (`pd.DataFrame`): DataFrame containing the data to split; must have a datetime column.
  - `date_col` (`str`): Name of the datetime column to use for ordering and splitting.
  - `n_splits` (`Optional[int]`): Number of equal-length, non-overlapping windows (optional).
  - `window_frac` (`Optional[float]`): Length of each window as a fraction of the total date span (used with `step_frac`).
  - `step_frac` (`Optional[float]`): Step size for window starts, as a fraction of the total date span.
  - `fractions` (`Optional[List[float]]`): List of window lengths, each as a fraction of the total date span (can be overlapping or not).
  - `overlap` (`bool`, default `False`): If `True`, allows windows to overlap.

- **Outputs:**
  - `splits` (`List[pd.DataFrame]`): List of DataFrames, one for each window, ordered by window start.
  - `log_df` (`pd.DataFrame`): Summary table for each split, including start/end times, actual data ranges, row counts, and diagnostic notes.

### ⚠️ Design Notes / Gotchas
- Does **not** mutate the input DataFrame; always returns new views/copies.
- Always sorts and normalizes the datetime column (timezone-naive, UTC).
- Raises `ValueError` if configuration is ambiguous or impossible (e.g., no valid splits, sum of fractions >1 without overlap, or invalid window/step fractions).
- Returns splits in chronological order, but some splits may be empty if date boundaries don't align perfectly with the data.
- Designed for **date-based** partitioning, not for count-based or random sampling.
- All edge cases (empty input, one-row DataFrame, fractional windows that cover all or none of the data) are handled and reflected in the log.

### 🔗 Related Functions
- See also: `generate_time_splits` for splitting by fixed time intervals or user-supplied date ranges.
- Often used together with data mining, rule evaluation, or validation/test pipelines that require pre-specified data partitions.

### 🧪 Testing Status
- Fully unit tested for a wide range of scenarios:
  - Typical and edge-case inputs
  - Invalid and degenerate cases (empty data, mismatched or nonsensical parameters)
  - Overlapping and non-overlapping splits
- If you extend for randomized splits or custom time zones, add further tests.

## 🧠 Function: split_datasets

### 📄 What It Does
- Splits a time-indexed DataFrame into multiple sub-DataFrames for train/test or rolling validation use.
- Supports two strategies: fixed date windows ("temporal") or fractional rolling windows across the dataset time span ("fractional").

### 🚦 When to Use
- Use when you need to divide a DataFrame into time-aware subsets for model training/validation, walk-forward analysis, or backtesting.
- Ideal for pipeline stages that require multiple train/test folds or WFA using real-world time order.
- Not appropriate for randomly shuffled datasets or if you're not working with datetime-indexed data.

### 🔢 Inputs and Outputs

#### Inputs
- `df: pd.DataFrame`  
  DataFrame with a datetime column to split by.
- `date_col: str`  
  Name of the datetime column in `df` used for ordering and slicing.
- `train_test_splits: Optional[int]`  
  Number of splits to generate. Required if no explicit ranges or fractions are passed.
- `train_test_ranges: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]`  
  Custom (start, end) date ranges per split (used in temporal mode).
- `method: Optional[str]`  
  Either `"temporal"` or `"fractional"`, defines the splitting strategy.
- `train_test_window_frac: Optional[float]`  
  Length of each split as a fraction of the dataset time span (fractional mode).
- `train_test_step_frac: Optional[float]`  
  Step size between split starts as a fraction of the time span (fractional mode).
- `train_test_fractions: Optional[List[float]]`  
  Explicit list of window sizes as fractions of time (fractional mode).
- `train_test_overlap: Optional[bool]`  
  If `True`, allows splits to overlap (fractional mode only).

#### Outputs
- `splits: List[pd.DataFrame]`  
  List of sliced DataFrames (one per split), ordered chronologically.
- `split_log: pd.DataFrame`  
  Structured log with metadata per split (start/end dates, actual rows, coverage fraction, etc.).

### ⚠️ Design Notes / Gotchas
- Does not mutate the input DataFrame.
- Expects `date_col` to be parseable as datetime; non-datetime input will raise or fail silently downstream.
- `train_test_overlap` defaults to `False` if not passed (safe but conservative).
- When using `train_test_ranges`, mismatched length with `train_test_splits` triggers a warning but continues.
- Outputs may contain empty splits — check log’s `n_rows` and `note` columns to diagnose.
- Fractional mode requires careful coordination between `window_frac`, `step_frac`, or `fractions`.

### 🔗 Related Functions
- `generate_time_splits()` — handles temporal mode internals
- `generate_fraction_splits()` — handles fractional mode logic

### 🧪 Testing Status
- ✅ Unit tested with `pytest` (temporal + fractional)
- Covers: fixed windows, overlapping windows, invalid inputs, empty dataframe
- Does not yet test: timezones, daylight savings transitions (assumed naive timestamps)

## 🧠 Function: rules_series_to_unique_rules

### 📄 What It Does
- Converts a pandas Series of rule strings into a list of structured rule representations.
- Each rule is parsed into a list of (feature_name, binary_value) conditions, with optional provenance tracking.

### 🚦 When to Use
- Use when you have mined rule strings (e.g., from RuleFit, Apriori, etc.) and need to convert them into a consistent, structured format for downstream processing or activation logic.
- Especially useful before calling `generate_rule_activation_dataframe()` or similar rule evaluation functions.
- Not suitable for rules containing inequality operators, non-binary conditions, or unstructured free-text logic.

### 🔢 Inputs and Outputs

#### Inputs
- `rule_series: pd.Series`  
  Series of rule strings. Each string should be a conjunction (`AND`) of clauses formatted like `('feature' == 0)` or `('feature' == 1)`.
- `provenance: bool = False`  
  If `True`, includes the original rule string as a set in the output for traceability.

#### Outputs
- `List[Tuple[List[Tuple[str, int]], Set[str]]]`  
  A list where each element is:
  - A list of (feature, value) tuples (parsed from the rule string)
  - A provenance set (with the original rule string if `provenance=True`, or an empty set)

### ⚠️ Design Notes / Gotchas
- Rules must match the exact format: `('feature' == 0)` or `('feature' == 1)`. Any deviation raises a `ValueError`.
- Whitespace and string formatting are normalized during parsing.
- Empty, null, or non-string entries result in an empty rule with an empty provenance set.
- Only supports binary values (`0` or `1`). Rules with other values are invalid.
- Function does not mutate the input `Series`.

### 🔗 Related Functions
- `generate_rule_activation_dataframe` — downstream consumer of this format
- Rule mining modules that output conjunctive rule strings

### 🧪 Testing Status
- ✅ Fully unit tested with `pytest`
- Covers valid rules, empty strings, non-string types, malformed inputs, and provenance behavior
- Edge cases like whitespace-only rules and mixed-type series entries are tested

## 🧠 Function: make_combined_rule_feature_df

### 📄 What It Does
- Builds a combined feature matrix for a test dataset using both single-variate and multivariate rules extracted from prior mining results.
- Returns a DataFrame with all activated rule features and the target column, ready for downstream evaluation or modeling.

### 🚦 When to Use
- Use when you have:
  - A set of mined rules (with metadata like depth and selection status), and
  - A test dataset with binary features and a known target column,
  and you want to create a rule-activation matrix aligned with that test data.
- Especially useful before model scoring or validation steps using rule-based features.

### 🔢 Inputs and Outputs

#### Inputs
- `train_mining_res: pd.DataFrame`  
  DataFrame of mined rules with columns `antecedents`, `rule_depth`, and `selected`.
- `test_df: pd.DataFrame`  
  Test data to evaluate rules on. Must include binary columns and the `target_col`.
- `target_col: str`  
  Name of the target variable column in `test_df`.

#### Outputs
- `combined_df: pd.DataFrame`  
  DataFrame with rule-based features as columns (binary activations) and the target column.  
- `multivar_map: pd.DataFrame`  
  Mapping between multivariate rule column names and human-readable rule descriptions.

### ⚠️ Design Notes / Gotchas
- Requires the following columns in `train_mining_res`: `antecedents`, `rule_depth`, `selected`.
- If no rules are selected, the result will contain only the `target_col`.
- Avoids duplicate inclusion of `target_col` if present in both single-variate and multivariate rule sets.
- Uses helper functions:
  - `rules_series_to_unique_rules()` for parsing multivariate rules.
  - `extract_rule_feature_names()` for single-variate rule feature extraction.
  - `generate_rule_activation_dataframe()` to apply multivariate rules to the test data.
- Does not modify inputs in place.
- Multivariate rules must follow a parsable string format; otherwise, parsing will fail upstream.

### 🔗 Related Functions
- `rules_series_to_unique_rules`
- `extract_rule_feature_names`
- `generate_rule_activation_dataframe`

### 🧪 Testing Status
- ✅ Unit tested with pytest
- Covers:
  - Both rule types active
  - Each rule type alone
  - No rules selected
  - Missing columns in inputs
- Does not yet test malformed rule parsing (assumes helper functions are already validated)

## 🧠 Function: test_mined_rules

### 📄 What It Does
- Applies mined rules to a test dataset and computes rule-level statistics including lift, confidence, and other metrics.
- Enhances output with human-readable rule descriptions and rule depth information, producing a clean, standardized result for analysis.

### 🚦 When to Use
- Use after rule mining is complete and you want to validate the selected rules against a holdout or test dataset.
- Especially useful for evaluating rule robustness, ranking rule quality, or visualizing rule distributions in analysis dashboards.
- Not intended for use on raw or unbinarized feature sets — test_df should be one-hot encoded or preprocessed to match rule assumptions.

### 🔢 Inputs and Outputs

#### Inputs
- `train_mining_res: pd.DataFrame`  
  DataFrame of mined rules with at least these columns: `antecedents`, `rule_depth`, `selected`.
- `test_df: pd.DataFrame`  
  Test dataset with binary features and a target column.
- `cfg: dict`  
  Configuration dictionary passed to the `generate_statistics` function.
- `target_col: str`  
  Name of the target column present in `test_df`.

#### Outputs
- `test_stats: pd.DataFrame`  
  Rule-level statistics with columns like `antecedents`, `lift`, `rule_depth`, and others.
- `test_stats_log: pd.DataFrame`  
  Log or metadata from the statistics computation (e.g. coverage, warnings, or intermediate diagnostics).

### ⚠️ Design Notes / Gotchas
- If the rule column mappings (`rule_column`) do not cover all rules, fallback logic ensures `antecedents` are retained.
- Assumes `generate_statistics` returns a DataFrame with `rule_column` and `antecedents` to merge on.
- Fills missing `rule_depth` with `1` for single-variate rules (default assumption).
- Drops columns like `rule_column` and renames `human_readable_rule` to `antecedents` for consistency.
- Relies on external functions: `make_combined_rule_feature_df`, `generate_statistics`, `compute_rule_depth`, and `merge_multivar_map_into_stats`.

### 🔗 Related Functions
- `make_combined_rule_feature_df`
- `generate_statistics`
- `compute_rule_depth`
- `merge_multivar_map_into_stats`

### 🧪 Testing Status
- ✅ Unit tested with `pytest`  
- Tests include: valid rule evaluation, multivariate and single-variate fallbacks, schema validation, and no-rule scenarios.
- Does not test rule parsing internals (assumed validated elsewhere).

## 🧠 Function: split_mining_pipeline

### 📄 What It Does
- Iterates over a list of dataset splits to run rule mining or testing, collecting per-split outputs and logs.
- Supports either walk-forward re-mining or fixed-rule evaluation across future splits.

### 🚦 When to Use
- Use when validating mined rules across multiple time-based or logical splits of your dataset.
- Suitable for both:
  - Re-mining rules independently per split (walk-forward-style evaluation)
  - Mining once and applying rules to subsequent splits (classic train/test setup)

### 🔢 Inputs and Outputs

#### Inputs
- `splits: List[pd.DataFrame]`  
  Ordered list of ≥2 pre-split DataFrames. Each split is processed independently.
- `cfg: params.config_validator.Config`  
  Configuration object for the data prep, mining, and testing steps.
- `re_mine: bool`  
  If `True`, re-mines rules on every split. If `False`, mines once and applies to remaining splits.
- `target_col: str`  
  Column name of the target variable in each split. Used during testing.
- `logger: Optional[Any]`  
  Optional logger that supports `.log_step(...)`. Used internally for logging (optional).

#### Outputs
- `results: List[pd.DataFrame]`  
  One DataFrame per split containing either mined rules or test statistics.
- `rule_counts: List[int]`  
  Number of rules mined/tested per split.
- `logs: List[Dict[str, Any]]`  
  Per-split logs with keys like `"prep_log"`, `"mine_log"`, or `"test_log"`.
- `initial_rules: Optional[pd.DataFrame]`  
  Rules mined from the first split if `re_mine=False`, otherwise `None`.

### ⚠️ Design Notes / Gotchas
- Will raise:
  - `ValueError` if `splits` has fewer than 2 items.
  - `TypeError` if any element in `splits` is not a `pd.DataFrame`.
- Assumes the presence of global helper functions:
  - `data_prep_pipeline`
  - `mining_pipeline`
  - `test_mined_rules`
- `initial_rules` is only returned when using fixed-rule evaluation (`re_mine=False`).
- Logs per split are returned as plain dicts — you may want to flatten/aggregate downstream.
- `target_col` must be present in every test split if testing is enabled.

### 🔗 Related Functions
- `mining_pipeline` — mines rules from a preprocessed DataFrame.
- `test_mined_rules` — evaluates previously mined rules on new data.
- `data_prep_pipeline` — handles preprocessing before mining or testing.

### 🧪 Testing Status
- ✅ Fully unit tested with `pytest`
- Covers:
  - Both `re_mine=True` and `re_mine=False` modes
  - Handling of missing or invalid inputs
  - Return structure and log content
  - Edge case: zero or malformed splits

## 🧠 Function: combine_split_results

### 📄 What It Does
- Aligns and merges rule-level statistics from multiple validation or mining splits into one unified DataFrame.
- Each rule's metrics from different splits are prefixed (e.g., `split_0_lift`, `split_1_lift`) to allow side-by-side comparison.

### 🚦 When to Use
- Use this after running rule mining or validation over multiple train/test splits to consolidate results for comparative analysis.
- Especially useful for backtesting frameworks where rule performance must be tracked across folds or time-based partitions.
- Not appropriate if input DataFrames lack `'antecedents'` or `'consequents'`.

### 🔢 Inputs and Outputs

**Inputs:**
- `results` (`List[pd.DataFrame]`):  
  List of per-split rule statistics, each with at least `antecedents` and `consequents` columns (and optionally `rule_depth`).
- `split_prefixes` (`Optional[List[str]]`):  
  Custom prefix per split (e.g., `["train", "val", "test"]`). Defaults to `["split_0", ..., "split_N"]`.

**Outputs:**
- `combined_df` (`pd.DataFrame`):  
  A single DataFrame indexed by rule identifiers, with metric columns prefixed by split name.

### ⚠️ Design Notes / Gotchas
- Joins are done on all identifier columns, so consistency of `rule_depth` values across splits is assumed but not enforced.
- Any missing metric per split will result in `NaN` for that rule/split.
- Input validation is strict: 
  - Must pass at least two DataFrames.
  - All must be of type `pd.DataFrame`.
  - Each must contain at least `antecedents` and `consequents`.
- All metric columns are unioned across splits to allow for flexible schema evolution.

### 🔗 Related Functions
- `split_mining_pipeline` – upstream generator of per-split results that this function consolidates.
- `test_mined_rules` – produces one of the DataFrames typically passed into this combiner.

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
- ✅ Covers custom prefixes, rule_depth handling, edge cases (missing metrics, schema mismatch)
- ⚠️ Assumes consistent formatting of `antecedents` and `consequents` across splits (not normalized inside this function)

## 🧠 Function: create_validation_log_df

### 📄 What It Does
- Summarizes rule-level validation results across one or more data splits into a single-row log.
- Computes per-split statistics (rule counts, selection rates, means/medians) and cross-split overlap metrics.

### 🚦 When to Use
- Use this function after combining rule mining results from multiple train/test splits (e.g., via `combine_split_results`).
- Especially useful when analyzing rule persistence, selection stability, or metric consistency across multiple datasets.
- Do *not* use this on an empty DataFrame or one lacking the expected column naming pattern.

### 🔢 Inputs and Outputs

#### Inputs:
- `train_test_results` (`pd.DataFrame`): Merged rule-level statistics across splits; must contain columns like `"split_0_lift"`, `"split_1_selected"`, etc.
- `splits` (`Optional[List[str]]`): Optional list of split name prefixes (e.g., `["split_0", "split_1"]`). If not provided, inferred from column headers.

#### Output:
- `pd.DataFrame`: A single-row DataFrame with:
  - Per-split counts (`n_rules_*`, `n_selected_*`)
  - Mean/median metrics (`mean_lift_*`, `median_confidence_*`, etc.)
  - Overlap counts and deltas between the first two splits (`n_overlap_rules`, `mean_delta_lift_overlap`, etc.)

### ⚠️ Design Notes / Gotchas
- Assumes input columns follow the `"split_N_metric"` naming convention. If split prefixes are inconsistent, results may be incomplete or incorrect.
- Sets `pd.set_option("future.no_silent_downcasting", True)` to avoid pandas `FutureWarning`; this may affect global state.
- The function only computes overlap metrics between the *first two* splits.
- Uses `ensure_bool()` helper to robustly coerce stringified booleans like `"1"`, `"true"`, etc., into `True`.

### 🔗 Related Functions
- [`combine_split_results`](#function-combine_split_results): used to prepare `train_test_results`
- [`ensure_bool`](#function-ensure_bool): internal helper used to sanitize selection flags

### 🧪 Testing Status
- Fully unit tested with `pytest`:
  - Typical input with and without overlap
  - Split inference logic
  - Stringified boolean edge cases
  - Empty DataFrame handling

## 🧠 Function: `validate_train_test`

### 📄 What It Does
- Runs a full train/test rule mining validation loop over a time-series-style dataset.
- Splits the data into multiple temporal slices, applies mining or testing on each, and aggregates results into a combined performance matrix.

### 🚦 When to Use
- Use this function when you want to evaluate the robustness of mined rules across time or other logical splits.
- Ideal for rolling-window validation, walk-forward analysis, or testing generalization of rule-based features.
- Don't use this function if you only want to mine rules on a single dataset without validation or if your data has no meaningful time/order structure.

### 🔢 Inputs and Outputs

#### Inputs:
- `df` (`pd.DataFrame`): Full input dataset, chronologically indexed.
- `cfg` (`Any`): Configuration object/dictionary for the mining and testing pipelines.
- `target_col` (`str`): Name of the target column to validate against.
- `date_col` (`str`): Column representing chronological order, required for windowing.
- `train_test_splits` (`int`): Number of train/test splits to generate.
- `train_test_ranges` (`List[List[str]] | List[Tuple[str, str]]`): Optional manual overrides for time ranges of splits.
- `train_test_split_method` (`str`): Split logic to use (e.g., `"rolling"`, `"fixed"`).
- `train_test_window_frac` (`float`): Fraction of total data used in each training window.
- `train_test_step_frac` (`float`): Step size to move window between splits.
- `train_test_fractions` (`List[float]`): Fractions to allocate within each split (e.g., `[0.7, 0.3]` for train/test).
- `train_test_overlap` (`bool`): Whether split windows may overlap.
- `train_test_re_mine` (`bool`): Whether to re-mine rules on each split (`True`) or just test mined rules on future splits (`False`).
- `logger` (`Optional[Any]`): Optional logger that supports `.log_step(...)`.

#### Outputs:
- `combined_results` (`pd.DataFrame`): Aggregated rule-level statistics across all validation splits.
- `metadata` (`dict`): Dictionary with:
  - `"train_test_rule_counts"`: List of number of rules per split.
  - `"train_test_logs"`: List of log dictionaries per split.
  - `"train_test_initial_rules"`: The mined rules from the first split (if not re-mining).

### ⚠️ Design Notes / Gotchas
- Assumes `date_col` is properly sortable and has no nulls.
- Will fail if `train_test_fractions` don’t sum to a valid allocation (e.g., >1.0).
- Underlying pipeline functions (`split_datasets`, `split_mining_pipeline`, `combine_split_results`) must be available and tested separately.
- This function does not handle deduplication or filtering of low-signal rules — that should happen downstream.
- Very sensitive to small datasets if `train_test_window_frac` or `step_frac` are too small.

### 🔗 Related Functions
- `split_datasets`
- `split_mining_pipeline`
- `combine_split_results`

### 🧪 Testing Status
- Covered by unit tests including:
  - Normal execution with mocked dependencies
  - Invalid input handling (e.g., `None` instead of DataFrame)
  - Edge case: only one split or re-mining enabled
- Mocking required for end-to-end isolation of dependent pipelines

## 🧠 Function: validate_wfa

### 📄 What It Does
- Runs a full Walk Forward Analysis (WFA) validation using a rule mining pipeline over time-ordered data.
- Splits the dataset into walk-forward windows, applies rule mining or testing per split, and merges results into a unified performance summary.

### 🚦 When to Use
- Use this function when you want to assess the consistency and stability of rules across time using WFA.
- Ideal for evaluating time-dependent strategies where past rule performance is tested on future data.
- Not appropriate for unordered datasets or non-temporal validation tasks.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Full dataset to validate, must include a sortable date column.
- `cfg` (`Any`): Configuration object passed into each pipeline (e.g., mining/test settings).
- `target_col` (`str`): Name of the prediction target column.
- `date_col` (`str`): Name of the date or time column used for ordering.
- `wfa_splits` (`int`): Number of sequential train/test splits to create.
- `wfa_ranges` (`List[List[str]] | List[Tuple[str, str]]`): Optional manual overrides for time windows.
- `wfa_split_method` (`str`): Method for creating splits (`"rolling"`, `"expanding"`, etc.).
- `wfa_window_frac` (`float`): Proportion of data used per training window.
- `wfa_step_frac` (`float`): Step size for rolling forward (fraction of full data).
- `wfa_fractions` (`List[float]`): Fractional sizes of train/test partitions (e.g., `[0.7, 0.3]`).
- `wfa_overlap` (`bool`): Whether train/test windows may overlap.
- `wfa_re_mine` (`bool`): Whether to re-mine rules on each split (`True`) or reuse initial rules (`False`).
- `logger` (`Optional[Any]`): Optional logger supporting `.log_step()`.

**Outputs:**
- `combined_results` (`pd.DataFrame`): Combined rule statistics across all WFA splits.
- `metadata` (`Dict[str, Any]`): Dictionary containing:
  - `"wfa_rule_counts"`: Number of rules mined/tested per split.
  - `"wfa_logs"`: Per-split logs (e.g., metadata, debug info).
  - `"wfa_initial_rules"`: Initial ruleset (if not re-mining).

### ⚠️ Design Notes / Gotchas
- Assumes the input `df` is chronologically ordered by `date_col` and contains sufficient data to split.
- Relies on several downstream functions (`split_datasets`, `split_mining_pipeline`, `combine_split_results`) — failures in those will propagate.
- Edge cases (e.g., too few rows for splits, bad fractions) will raise exceptions from downstream.
- Does not mutate input; all operations are on split copies.

### 🔗 Related Functions
- `validate_train_test()`: Similar API but performs train/test validation instead of WFA.
- `split_datasets()`, `split_mining_pipeline()`, `combine_split_results()`: Core components used within this wrapper.

### 🧪 Testing Status
- Covered by unit tests including:
  - Typical path with mocks
  - Degenerate/empty input
  - Parametrized invalid configurations (e.g., bad fractions)

## 🧠 Function: train_test_pipeline

### 📄 What It Does
- Runs a complete train/test rule mining validation pipeline using config parameters or dynamic overrides.
- Computes per-rule statistics and a summary log, with optional logging support.

### 🚦 When to Use
- Use this function when you want to run a full train/test validation pass over a dataset with time-based splitting and rule mining.
- Helpful during backtesting or feature validation stages of a rule discovery pipeline.
- Avoid using it for real-time inference or when full control over individual steps (e.g. mining or splitting) is required.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Input dataset containing feature, date, and target columns.
- `cfg` (`Any`): Configuration object holding default values for pipeline execution.
- `logger` (`Optional[Any]`): Logger instance supporting `.log_step(...)`, used to record pipeline stages.
- `**overrides` (`dict`): Optional keyword arguments that override values in `cfg`.

**Outputs:**
- `train_test_results` (`pd.DataFrame`): Per-rule performance metrics across splits.
- `train_test_log` (`pd.DataFrame`): One-row summary log with mean lift, selection stats, and overlap metrics.
- `pipeline_logs` (`Dict[str, List[pd.DataFrame]]`): Metadata containing rule counts, logs, and mined rule objects.

### ⚠️ Design Notes / Gotchas
- Uses a helper `param()` to prefer overrides over `cfg` values — be cautious about name mismatches.
- Assumes `cfg` contains all required fields or raises `AttributeError` at runtime.
- Hardcodes `splits=["split_0", "split_1"]` for summary logging — this may not generalize if more splits are configured.
- Assumes `logger.log_step()` accepts `step_name`, `info`, `df`, and `max_rows` arguments — this is a soft dependency.

### 🔗 Related Functions
- `validate_train_test`: Core validation logic and data splitting.
- `create_validation_log_df`: Computes summary statistics from validation results.

### 🧪 Testing Status
- Unit tested using `pytest` with fixtures and mocks.
- Covers default execution, override resolution, logger behavior, and invalid `cfg`.
- Edge case: test assumes at least 2 splits (split_0, split_1) — more general logging logic could improve robustness.

## 🧠 Function: wfa_pipeline

### 📄 What It Does
- Executes a complete Walk Forward Analysis (WFA) validation workflow using rule mining.
- Automatically pulls WFA configuration parameters from a config object and applies overrides if specified.
- Logs results and generates a summary metrics DataFrame across splits.

### 🚦 When to Use
- Use this function when you want to evaluate the robustness of rule-based models using a walk-forward validation scheme.
- Suitable when testing over temporally ordered datasets where performance stability over time is important.
- Helpful for detecting rule overfitting or instability across rolling windows.
- Not appropriate if you don't have a time-based column or if your config is incomplete.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Dataset to run WFA validation on. Must include a sortable datetime or ordinal column.
- `cfg` (`Any`): Config object containing named WFA parameters (e.g. `wfa_splits`, `wfa_window_frac`, etc.).
- `logger` (`Optional[Any]`): Logger object with a `.log_step(...)` method (optional).
- `**overrides` (`dict`): Optional named parameters to override values in the config object.

**Outputs:**
- `wfa_results` (`pd.DataFrame`): Combined per-rule performance results for each split.
- `wfa_log` (`pd.DataFrame`): One-row summary DataFrame of key metrics (counts, means, overlap).
- `pipeline_logs` (`Dict[str, List[pd.DataFrame]]`): Dictionary containing:
  - `"wfa_rule_counts"`: Rule counts per split.
  - `"wfa_logs"`: Internal logs per window.
  - `"wfa_initial_rules"`: Initial mined rule set (if `wfa_re_mine=False`).

### ⚠️ Design Notes / Gotchas
- Assumes `validate_wfa` and `create_validation_log_df` are present and working as intended.
- Expects all WFA-relevant parameters to exist either in `cfg` or `overrides`. Will raise `AttributeError` if missing.
- Dynamically constructs `split_prefixes` based on detected column names — assumes column naming convention like `"split_0_lift"`.
- If `logger` is provided, it must support `.log_step(...)`; otherwise, log step is skipped silently.
- Uses config’s `log_max_rows` for output truncation — ensure this is set.

### 🔗 Related Functions
- `validate_wfa`: Runs the actual walk-forward mining and evaluation.
- `create_validation_log_df`: Summarizes rule-level results into a single row.
- `train_test_pipeline`: Similar wrapper for simpler train/test (non-walk-forward) validation.

### 🧪 Testing Status
- Unit tested using `pytest`, with fixtures mocking `validate_wfa` and `create_validation_log_df`.
- Covered test cases include:
  - Standard execution
  - Config override behavior
  - Logger integration
  - Missing or malformed input edge cases

## 🧠 Function: `resample_dataframe`

### 📄 What It Does
- Resamples a Pandas DataFrame using one of three bootstrap modes:  
  traditional (i.i.d. rows), block (consecutive row chunks), or group-wise block sampling.

### 🚦 When to Use
- Use when you need to test the stability or robustness of patterns, rules, or metrics via resampling.
- Especially useful in statistical validation workflows (e.g. bootstrap testing of mined rules).
- Recommended for pipelines involving repeated testing on perturbed versions of the same dataset.

**Example**:  
To evaluate if discovered rules remain stable under noisy re-sampling of the data.

**Avoid if**:
- You need stratified sampling or sampling with complex dependencies.
- Your dataset is too short to support the requested block size.

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame` — Input DataFrame to be resampled.
- `mode: str` — Resampling mode. One of:
  - `"traditional"`: i.i.d. resampling
  - `"block"`: fixed-length blocks across full DataFrame
  - `"block_ids"`: block resampling applied per group (e.g. per ticker)
- `block_size: int = 5` — Number of rows per block for block bootstrap.
- `date_col: str = "date"` — Column used to sort data for block sampling.
- `id_cols: Optional[List[str]] = None` — Required when `mode='block_ids'`; columns to group by.
- `random_state: int = 42` — Seed for reproducible randomness.

**Output**:
- `pd.DataFrame` — Resampled DataFrame with approximately the same number of rows as input. Columns are preserved; row order will change.

### ⚠️ Design Notes / Gotchas
- The input must not be empty — raises `ValueError` otherwise.
- `block_size` must be ≥ 1 and not larger than the group/data length — validated explicitly.
- In `'block'` or `'block_ids'` mode, the data is sorted by `date_col` and reset.
- Does **not** mutate the original DataFrame.
- All randomness is controlled via `np.random.default_rng(random_state)`.
- Assumes `date_col` exists and is sortable — missing or misformatted date columns will silently error later.
- Groups in `block_ids` mode are assumed large enough to allow block sampling.

### 🔗 Related Functions
- `block_sample()` — Internal helper function used to handle block bootstrap logic.
- Should be tested in tandem with validation tools (e.g. rule testing, p-value simulation).

### 🧪 Testing Status
- ✅ Covered by `pytest` tests:
  - All three modes (`traditional`, `block`, `block_ids`)
  - Empty input, invalid parameters, and small data edge cases
  - Group size vs. block size conflicts
- 🔍 Future enhancement: add tests for varying `block_size` and `date_col` behavior

## 🧠 Function: `summarize_rule_metrics`

### 📄 What It Does
- Aggregates rule performance statistics across multiple evaluation runs for each unique (antecedents, consequents) rule pair.
- Produces summary statistics like mean, std, min, max, quantiles, and selection counts.

### 🚦 When to Use
- Use this function when you have a dataframe of rule evaluation results (e.g. from bootstrap or walk-forward tests) and want to summarize their behavior across repetitions.
- Helpful for ranking, filtering, or visualizing rule robustness and statistical performance.

**Example**: After running a rule mining algorithm across 100 resampled datasets, use this function to compute the average lift and confidence of each rule along with how often it was selected.

**Avoid if**:
- The input dataframe lacks required columns like `selected`, `antecedents`, or metric columns.
- You want per-run detail — this function summarizes across all rows.

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame`  
  Dataframe containing rule evaluation results across test runs. Must include:
  - `"antecedents"` and `"consequents"` columns (used as group keys)
  - `"selected"` (boolean column indicating rule passed in that run)
  - One or more metric columns (e.g. `"lift"`, `"confidence"`)

- `metrics: List[str]`  
  Names of numeric metric columns to summarize for each rule.

**Outputs**:
- `pd.DataFrame`  
  One row per rule pair, with:
  - Summary stats for each metric: mean, std, min, max, q05, q95
  - Counts of how often rule was selected vs tested
  - A `selected_fraction` column showing pass-rate

### ⚠️ Design Notes / Gotchas
- Raises `ValueError` if any required column is missing.
- Assumes `selected` is boolean; unexpected types may yield misleading counts.
- Aggregation is done via `.groupby(["antecedents", "consequents"])`; any column with higher granularity will be collapsed.
- Quantiles are hardcoded to 5th and 95th percentiles.
- Lambda functions used for quantiles will appear as generic names (`<lambda_0>`, etc.) unless renamed explicitly — already handled in this function.

### 🔗 Related Functions
- [`resample_dataframe`](#resample_dataframe): Often used upstream to generate the input data.
- Any rule evaluation or testing pipeline that generates repeated metric observations.

### 🧪 Testing Status
- ✅ Fully unit tested using `pytest`
- Covers:
  - Normal multi-rule input
  - Single-group edge case
  - Missing column errors
  - Quantile output structure
- 🔍 Additional test ideas:
  - NaN metric values or zero test_count edge case
  - Explicit test for deterministic ordering (if needed)

## 🧠 Function: `create_validation_summary_log`

### 📄 What It Does
- Aggregates a set of rule-level validation metrics into a single-row summary report.
- Computes high-level stats like total rule count, test count, average selection rate, and metric summaries.

### 🚦 When to Use
- Use this after running `summarize_rule_metrics()` to produce a dashboard-friendly summary of your validation run.
- Ideal for high-level reporting or comparisons across experiments (e.g., WFA, bootstrap, null tests).

**Example**:  
Generate an executive-friendly summary for how rules performed on average during bootstrap validation.

**Avoid if**:
- You only have per-run data; this expects already-aggregated per-rule metrics.

### 🔢 Inputs and Outputs

**Inputs**:
- `summary_df: pd.DataFrame`  
  The result from `summarize_rule_metrics()`. Must contain:
  - `selected_test_count`: number of times each rule was evaluated
  - `selected_fraction`: how often each rule passed validation
  - `<metric>_mean` for each metric in `metrics`

- `metrics: List[str]`  
  Metric base names (e.g. `["lift", "confidence"]`) whose `_mean` columns will be summarized.

**Outputs**:
- `pd.DataFrame`  
  A one-row dataframe with:
  - `total_rules`, `total_tests`, `avg_selection_rate`
  - For each metric: `_mean_mean`, `_mean_std`, `_mean_min`, `_mean_max`

### ⚠️ Design Notes / Gotchas
- Raises `ValueError` if required columns are missing.
- Only processes `<metric>_mean` columns — will not summarize full distributions or quantiles.
- Uses `.max()` to infer total tests, assuming all rules were evaluated equally.
- Assumes input dataframe has already been grouped at the rule level — do not use on raw resampled data.

### 🔗 Related Functions
- [`summarize_rule_metrics`](#summarize_rule_metrics) — this function expects its output as input
- Can be paired with resampling/testing functions like `bootstrap_resample_test`, `run_wfa_test`

### 🧪 Testing Status
- ✅ Unit tested via `pytest`
- Covers:
  - Normal multi-metric input
  - Edge case with a single rule
  - Missing columns and invalid input
- 🔍 Future test idea: behavior with NaN values in metric columns

## 🧠 Function: `validate_bootstrap`

### 📄 What It Does
- Runs bootstrap validation by re-testing a fixed set of mined rules on multiple resampled datasets.
- Returns a rule-level metrics summary and a high-level log summarizing selection robustness.

### 🚦 When to Use
- Use when you want to assess the statistical stability and noise sensitivity of discovered rules using bootstrap resampling.
- Helpful in determining which rules are consistently strong across different samples.

**Example**:  
After running rule mining on your full dataset, call this function to evaluate how robust those rules are to sampling variation by re-testing them across 50 resampled versions of the data.

**Avoid if**:
- You need to discover rules in each resample — this function assumes a fixed rule set from the original full dataset.

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame`  
  Raw input dataset containing features and target.

- `cfg: Any`  
  Configuration dict or object. Must contain at least:
  - `"target_col"`: column name of the prediction target
  - `"date_col"` (optional): used for block bootstrapping
  - `"id_cols"` (optional): group identifiers for grouped block bootstrapping

- `logger: Optional[Any]`  
  Logger object (can be None).

- `random_state: int`  
  Seed for reproducible resampling.

- `n_bootstrap: int`  
  Number of bootstrap rounds to run.

- `verbose: bool`  
  Show progress bar if True.

- `resample_method: str`  
  One of `'traditional'`, `'block'`, `'block_ids'`.

- `block_size: int`  
  Size of resampling blocks (used in `'block'`/`'block_ids'`).

**Outputs**:
- `bootstrap_results: pd.DataFrame`  
  Rule-level summary metrics (mean, std, quantiles, etc.) aggregated across bootstrap runs.

- `bootstrap_log: pd.DataFrame`  
  Single-row summary of the overall bootstrap results (e.g., average selection rate).

### ⚠️ Design Notes / Gotchas
- Assumes `cfg["target_col"]` is present — will raise `KeyError` if missing.
- Initial rule mining is only done once, on the full dataset; this does not rediscover rules in each bootstrap sample.
- Resampling is done *before* data preparation (so cleaning/scaling is deterministic post-sample).
- All downstream functions (`data_prep_pipeline`, `mining_pipeline`, etc.) must follow consistent schema expectations.

### 🔗 Related Functions
- `resample_dataframe`: performs i.i.d. or block-based resampling
- `summarize_rule_metrics`: aggregates rule-level metrics across tests
- `create_validation_summary_log`: builds summary table of validation run
- `test_mined_rules`: re-evaluates rules against new data

### 🧪 Testing Status
- ✅ Unit tested via `pytest` with full mock patching
- Covers:
  - Typical run with minimal config
  - Zero-iteration edge case
  - Missing config fields (e.g. `target_col`)
- 🔍 Future test suggestions:
  - Resampling with very small datasets
  - Failure propagation from downstream pipeline components

## 🧠 Function: `bootstrap_pipeline`

### 📄 What It Does
- Runs the full bootstrap validation pipeline using configuration and optional overrides.
- Wraps `validate_bootstrap()` to match the modular pipeline architecture and optionally logs results.

### 🚦 When to Use
- Use this function when integrating bootstrap validation into a larger pipeline with config-driven execution.
- Ideal when working with structured config objects and you want to keep all logic centralized and pluggable.
- Use if you want to override specific parameters like `n_bootstrap` or `resample_method` at runtime.

**Avoid if**:
- You want to run bootstrap logic outside of the pipeline abstraction — use `validate_bootstrap()` directly instead.

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame`  
  The input dataset (usually unprocessed raw data).

- `cfg: Any`  
  Config object with attributes such as:
  - `date_col`, `id_cols`, `target_col`
  - `n_bootstrap`, `bootstrap_verbose`, `resample_method`, `block_size`
  - `log_max_rows`

- `logger: Optional[Any]`  
  Optional logger object supporting `.log_step(...)`.

- `**overrides: dict`  
  Runtime overrides for any config attribute (e.g., `n_bootstrap=5`).

**Outputs**:
- `bootstrap_results: pd.DataFrame`  
  Rule-level metric summary from `summarize_rule_metrics`.

- `bootstrap_log: pd.DataFrame`  
  One-row summary from `create_validation_summary_log`.

### ⚠️ Design Notes / Gotchas
- Config object must support attribute access (`cfg.target_col`, not `cfg["target_col"]`).
- Missing required config values will raise `AttributeError` or `ValueError` via `validate_bootstrap`.
- Only known config parameters are passed to `validate_bootstrap` — additional extras are ignored unless you extend the wrapper.
- Logging is optional but must implement `.log_step(...)` if provided.

### 🔗 Related Functions
- `validate_bootstrap`: Core function performing the actual bootstrap testing
- `summarize_rule_metrics`: Aggregates test results per rule
- `create_validation_summary_log`: Builds high-level run summary
- `mining_pipeline`, `test_mined_rules`: Upstream dependencies in the pipeline

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
- Covers:
  - Default config usage
  - Overrides passed in correctly
  - Logging integration
  - Missing config key (e.g., `target_col`) edge case

## 🧠 Function: `shuffle_dataframe`

### 📄 What It Does
- Returns a shuffled copy of a DataFrame using one of three modes: shuffling the target column, shuffling rows, or independently shuffling each column.
- Commonly used in null hypothesis testing to break data dependencies while preserving structure.

### 🚦 When to Use
- Use to simulate null distributions when validating rule significance or testing model robustness.
- Especially useful before applying statistical tests like FDR correction or when generating permutation baselines.

**Examples**:
- `mode="target"`: shuffle just the label column for null hypothesis testing.
- `mode="rows"`: resample the full rows while keeping column coherence intact.
- `mode="columns"`: independently permute each feature column to destroy multivariate structure.

**Avoid if**:
- You need stratified or grouped shuffling — this function does not preserve group structures.
- You expect deterministic output without setting `random_state`.

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame`  
  The input dataset to be shuffled.

- `mode: str`  
  One of `"target"`, `"rows"`, or `"columns"`. Determines the shuffle behavior.

- `target_col: Optional[str]`  
  Required only if `mode="target"`. The name of the column to shuffle.

- `random_state: Optional[int]`  
  Optional seed for reproducibility.

**Output**:
- `pd.DataFrame`  
  A new DataFrame with the same structure but with specified shuffling applied.

### ⚠️ Design Notes / Gotchas
- Does **not mutate** the input DataFrame; always returns a copy.
- Raises `ValueError` if `mode` is invalid or if `target_col` is missing when required.
- In `"columns"` mode, each column is permuted independently using NumPy's `default_rng`.
- In `"target"` mode, only the specified column is changed — other columns remain exactly as-is.
- The `"rows"` mode is equivalent to a row-wise permutation and resets the index.

### 🔗 Related Functions
- `validate_null_test` (if used downstream in permutation testing)
- `resample_dataframe` — similar interface for resampling strategies

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
- Covers:
  - All three shuffle modes
  - Reproducibility with fixed seed
  - Invalid `mode` and missing `target_col` exceptions
  - Preservation of structure and value sets
  - Empty dataframe behavior

## 🧠 Function: `compute_relative_error`

### 📄 What It Does
- Calculates the relative error (standard deviation divided by mean) of a numeric metric over the most recent M iterations.
- Used to assess convergence or stability during iterative null distribution or bootstrap calculations.

### 🚦 When to Use
- Use when you need to monitor the stability of a metric (e.g., quantile estimates) across iterations.
- Especially useful in null/permutation testing to trigger early stopping once the distribution stabilizes.

**Example**:  
If you're computing the 95th percentile of lift under the null distribution every 10 permutations, this function tells you how consistent those recent estimates are.

**Avoid if**:
- You're working with unordered data (ensure iteration column reflects computation order).
- Your metric can legitimately be zero (division by zero returns `NaN`, which may be undesired).

### 🔢 Inputs and Outputs

**Inputs**:
- `df: pd.DataFrame`  
  A dataframe containing at least one numeric column and one iteration identifier column.

- `metric_col: str`  
  The name of the column with metric values (e.g., `"lift_q95"`).

- `iteration_col: str`  
  Column used to sort by iteration order (e.g., `"perm_num"` or `"step"`).

- `m_recent: int`  
  Number of most recent rows (after sorting) to use for the relative error calculation.

**Output**:
- `float`  
  The relative error (standard deviation / mean). Returns `np.nan` if the mean is zero.

### ⚠️ Design Notes / Gotchas
- Data is sorted by `iteration_col` — ensure that column is properly incrementing and numeric.
- Raises `ValueError` if:
  - `metric_col` or `iteration_col` is missing from the dataframe
  - `m_recent < 1`
  - The dataframe has fewer rows than `m_recent`
- Returns `np.nan` if mean of selected values is zero (to avoid divide-by-zero).
- Does **not** mutate the original dataframe.

### 🔗 Related Functions
- Can be used with convergence diagnostics or null-distribution generators (e.g., `validate_null_test`)
- Pair with `summarize_rule_metrics` to evaluate rule stability across tests

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
- Covers:
  - Normal cases
  - `mean == 0` edge case
  - Single-row and minimal data cases
  - Missing column validation
  - `m_recent` too large
  - `m_recent == 0` (invalid)

## 🧠 Function: `summarize_null_distribution`

### 📄 What It Does
- Computes a single-row summary of a null distribution from permutation test results.
- Outputs key statistics such as mean, std, quantiles, and total observation counts.

### 🚦 When to Use
- Use after running a null test or permutation test where metric values are collected over multiple permutations.
- Helpful for logging diagnostics, convergence checks, or comparing the null distribution to observed values.

**Example**:  
Summarize the distribution of `lift_q95` over 100 permutations to report average behavior and range of false positives.

**Avoid if**:
- Your input DataFrame is empty or missing key columns.
- You need grouped summaries (this only produces a flat one-row summary).

### 🔢 Inputs and Outputs

**Inputs**:
- `null_df: pd.DataFrame`  
  DataFrame containing metric results for each permutation or iteration.

- `metric_col: str`  
  Name of the numeric column to summarize (e.g., `"lift_q95"`).

- `iteration_col: str`  
  Column used to count unique permutations or iterations (e.g., `"perm_num"`).

**Outputs**:
- `pd.DataFrame`  
  A one-row DataFrame with the following columns:
  - `metric_mean`, `metric_std`, `metric_min`, `metric_max`
  - `metric_q05`, `metric_q50`, `metric_q95`
  - `n_permutations`, `n_observations`

### ⚠️ Design Notes / Gotchas
- Raises `ValueError` if `metric_col` or `iteration_col` is not present.
- Assumes `iteration_col` values identify unique permutation runs.
- All statistics are computed across the entire DataFrame — no grouping or filtering is applied.
- Returns quantiles using `.quantile()` and median using `.median()` (slightly different methods, by design).

### 🔗 Related Functions
- `compute_relative_error` — assesses stability of null distribution
- `shuffle_dataframe` — used to generate null distribution via shuffling
- `validate_null_test` (if implemented) — orchestrates full null testing pipeline

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
- Covers:
  - Valid input with multiple permutations
  - Missing column errors
  - One-row edge case
  - Empty DataFrame exception

## 🧠 Function: validate_null

### 📄 What It Does
- Runs a null hypothesis test by shuffling the input data and re-evaluating previously mined rules to simulate a distribution of rule statistics under randomness.
- Optionally stops early if the metric of interest stabilizes, based on a relative error threshold.

### 🚦 When to Use
- Use when you want to assess the statistical significance of discovered rules by generating a null distribution through permutation.
- Commonly used for FDR correction, empirical p-values, or robustness validation of mined signal quality.
- Don’t use if rules must be re-mined per permutation — this function assumes rules are fixed after the first discovery.

### 🔢 Inputs and Outputs

**Inputs**
- `df` (`pd.DataFrame`): Raw input data including features and target column.
- `cfg` (`Any`): Configuration object passed to downstream pipeline functions (e.g., preprocessing, testing).
- `target_col` (`str`): Name of the column to shuffle during null testing.
- `logger` (`Optional[Any]`): Optional structured logger supporting `.log_step(...)`.
- `n_null` (`int`, default=1000): Maximum number of null permutations.
- `shuffle_mode` (`str`, default="target"): One of `"target"`, `"rows"`, or `"columns"` to determine how shuffling is performed.
- `early_stop_metric` (`str`, default="lift"): Metric used to monitor convergence for early stopping.
- `es_m_permutations` (`int`, default=50): Window size for checking convergence.
- `rel_error_threshold` (`float`, default=0.01): Relative error threshold for triggering early stopping.
- `verbose` (`bool`, default=True): Whether to display progress via tqdm or similar.

**Outputs**
- `results_df` (`pd.DataFrame`): Full null distribution of rule evaluation results across permutations.
- `summary_log` (`pd.DataFrame`): Single-row summary of null distribution stats, including final relative error.

### ⚠️ Design Notes / Gotchas
- Assumes mining happens only once on the original (unshuffled) data — mined rules are reused across all null samples.
- Early stopping only activates if `n_null ≥ es_m_permutations`; otherwise, all permutations will run.
- Requires metric used for early stopping (`early_stop_metric`) to exist in the result DataFrame — ensure consistency between rule testing output and selected metric.
- Input `df` is not modified in-place.

### 🔗 Related Functions
- `shuffle_dataframe`: Handles the permutation logic across various shuffle modes.
- `mining_pipeline`: Performs the initial rule discovery.
- `test_mined_rules`: Evaluates mined rules on each shuffled sample.
- `compute_relative_error`: Used to monitor convergence of the null metric.
- `summarize_null_distribution`: Aggregates statistics of the final null distribution.

### 🧪 Testing Status
- Unit tested with coverage for:
  - Early stopping condition
  - Full run with no early stop
  - Edge case: `n_null < es_m_permutations` triggers `ValueError`
- Integration tested via pipeline using synthetic test cases.

## 🧠 Function: null_pipeline

### 📄 What It Does
- Runs the null distribution validation process using configuration and optional parameter overrides.
- Wraps the `validate_null()` function for modular pipeline compatibility and structured logging.

### 🚦 When to Use
- Use this when you want to generate a null distribution for hypothesis testing or statistical validation of mined rules.
- Common in pipelines where you evaluate signal strength under random target permutations.
- Not suitable if rule mining needs to be redone on each permutation — this assumes fixed rules mined once.

### 🔢 Inputs and Outputs
**Inputs**
- `df` (`pd.DataFrame`): Raw input data including features and target.
- `cfg` (`Any`): Configuration object with parameter access via attributes.
- `logger` (`Optional[Any]`): Optional logger supporting `.log_step(...)`.
- `**overrides` (`dict`): Keyword overrides to temporarily change config values.

**Outputs**
- `null_df` (`pd.DataFrame`): DataFrame of per-rule evaluation results across permutations.
- `null_log` (`pd.DataFrame`): Single-row summary of null distribution metrics (mean, std, quantiles, etc.).

### ⚠️ Design Notes / Gotchas
- Assumes `cfg` allows attribute-style access — using `cfg.key`, not `cfg['key']`.
- Will raise `AttributeError` if required parameters (like `target_col`) are missing and not overridden.
- Uses early stopping via relative error if enabled (`es_m_permutations`, `rel_error_threshold`).
- Logs output only if a `logger` is provided.
- Relies on `validate_null()` internally — ensure that function is robustly tested.

### 🔗 Related Functions
- `validate_null`: Core logic for generating and evaluating null distribution.
- `compute_relative_error`: Used for early stopping condition.
- `shuffle_dataframe`, `data_prep_pipeline`, `test_mined_rules`: Internal steps of null testing pipeline.

### 🧪 Testing Status
- Unit tested via `pytest`, including override handling, logger logging, and exception cases.
- Edge cases tested: missing config values, overridden arguments, logger presence.

## 🧠 Function: summarize_fdr_results

### 📄 What It Does
- Generates a summary of False Discovery Rate (FDR) test results from a DataFrame.
- Optionally formats the output as a markdown string for easy logging or reporting.

### 🚦 When to Use
- Use this after performing multiple hypothesis testing with FDR correction (e.g., Benjamini–Hochberg).
- Helpful for assessing how many rules or hypotheses remain statistically significant post-correction.
- Use `groupby_col` when you want to break down significance metrics by rule depth, algorithm, or other categories.

### 🔢 Inputs and Outputs

**Inputs**
- `df: pd.DataFrame`  
  → DataFrame with columns for p-values and boolean FDR significance flags.
- `pval_col: str = "pval"`  
  → Name of the column with raw p-values.
- `fdr_sig_col: str = "fdr_significant"`  
  → Name of the column indicating FDR significance (boolean).
- `correction_alpha: float = 0.05`  
  → The alpha threshold used for comparison and summary counts.
- `groupby_col: Optional[str] = None`  
  → If specified, results are grouped and summarized by this column.
- `as_markdown: bool = False`  
  → If True, returns markdown-formatted summary string; else, returns a DataFrame.

**Outputs**
- `Union[pd.DataFrame, str]`  
  → A single-row or grouped summary DataFrame (default) or a markdown-formatted string (if `as_markdown=True`).

### ⚠️ Design Notes / Gotchas
- Raises `ValueError` if required columns (`pval_col`, `fdr_sig_col`) are missing from the input DataFrame.
- If the input DataFrame is empty, all stats are computed gracefully (e.g., proportion as NaN).
- `groupby_col` can include missing values; `dropna=False` ensures they are retained in output.
- Markdown output is flat (no nested tables), optimized for logs or inline diagnostics.
- Does **not** perform the FDR correction itself — it assumes you’ve already flagged significance externally.
- Input DataFrame is not mutated.

### 🔗 Related Functions
- Depends internally on `_summarize()` (in-scope helper) and `_df_to_markdown()` (markdown formatter).
- Meant to follow FDR correction steps like `statsmodels.stats.multitest.multipletests`.

## 🧠 Function: validate_multiple_tests

### 📄 What It Does
- Computes empirical p-values for rule mining results by comparing observed statistics to a null distribution.
- Applies multiple hypothesis correction (e.g., FDR) and annotates the input DataFrame with corrected p-values and significance flags.

### 🚦 When to Use
- Use after generating rule mining results and constructing a null distribution (e.g., via permutation or bootstrapping).
- Useful when testing many rules for statistical significance while controlling the false discovery rate.
- Use when both directional and non-directional deviations from the null are of interest (via the `mode` argument).

### 🔢 Inputs and Outputs

**Inputs**
- `mining_res: pd.DataFrame`  
  → DataFrame of rule mining results, must contain the column specified by `early_stop_metric`.
- `null_df: pd.DataFrame`  
  → Null distribution DataFrame with the same `early_stop_metric` column.
- `early_stop_metric: str = "lift"`  
  → Metric used to evaluate rule quality; compared between actual and null.
- `mode: {"greater", "less", "two-sided"} = "greater"`  
  → Tail mode for p-value computation:
  - `"greater"`: looks for high values (right tail)
  - `"less"`: looks for low values (left tail)
  - `"two-sided"`: looks for any extreme deviation from center
- `correction_alpha: float = 0.05`  
  → Significance threshold for FDR correction.
- `correction_metric: str = "fdr_bh"`  
  → Correction method passed to `statsmodels.stats.multitest.multipletests`.
- `center: Optional[float] = None`  
  → For `"two-sided"` mode, specifies the center (e.g., null median) for deviation. If None, defaults to null median.

**Outputs**
- `Tuple[pd.DataFrame, pd.DataFrame]`
  - The first DataFrame is a copy of `mining_res` with added columns:
    - `"pval"` (empirical)
    - `"pval_<method>"` (corrected)
    - `"<method>_significant"` (boolean flag)
  - The second is a one-row or grouped summary DataFrame (from `summarize_fdr_results`).

### ⚠️ Design Notes / Gotchas
- Will raise `ValueError` if `early_stop_metric` is missing in either DataFrame.
- FDR correction failures (e.g., due to invalid method) raise explicit error with method name.
- `mining_res` is copied internally and not mutated in-place.
- If using `two-sided` mode, ensure null distribution is approximately symmetric if no `center` is provided.
- Assumes `compute_empirical_pvals` and `summarize_fdr_results` are defined and imported.

### 🔗 Related Functions
- `compute_empirical_pvals`: calculates one- or two-tailed empirical p-values.
- `summarize_fdr_results`: produces human-readable summary of significance outcomes.

### 🧪 Testing Status
- ✅ Unit tested with `pytest`, covering:
  - All three `mode` options
  - Empty and missing-column edge cases
  - Invalid correction methods
  - Manual `center` override for two-sided mode

### 🧪 Testing Status
- ✅ Unit tested via `pytest`
- Covers grouped vs. ungrouped input, markdown formatting, custom alpha values, missing columns, and edge cases (e.g., empty input)
- Parametrized tests check variations in input structure and content

## 🧠 Function: fdr_pipeline

### 📄 What It Does
- Runs False Discovery Rate (FDR) correction on rule mining results using parameters from a config object or override dictionary.
- Returns the results annotated with p-values and significance flags, along with a summary log for reporting.

### 🚦 When to Use
- Use this function as part of a modular validation pipeline when you need to apply FDR correction to rule mining outputs.
- Especially useful when working with many candidate rules and needing to control the false discovery rate.
- Call this function inside a larger workflow where config-driven execution is required.

### 🔢 Inputs and Outputs

**Inputs**
- `mining_res: pd.DataFrame`  
  → Rule mining results with test statistics (e.g. lift, confidence).
- `null_df: pd.DataFrame`  
  → Null distribution of the same test statistic (used for p-value estimation).
- `cfg: Any`  
  → Configuration object (e.g. Namespace or class) supporting attribute access for required FDR parameters.
- `logger: Optional[Any]`  
  → Optional structured logger with `.log_step()` support.
- `**overrides: Any`  
  → Keyword overrides for any config parameter (e.g. to experiment without modifying `cfg`).

**Outputs**
- `fdr_res: pd.DataFrame`  
  → `mining_res` with added columns:  
  - `"pval"` — empirical p-value  
  - `"pval_<method>"` — FDR-adjusted p-value  
  - `"<method>_significant"` — boolean flag after correction
- `fdr_log: pd.DataFrame`  
  → One-row summary DataFrame describing the FDR test results (from `summarize_fdr_results()`).

### ⚠️ Design Notes / Gotchas
- The function expects `cfg` to contain keys: `early_stop_metric`, `fdr_mode`, `correction_alpha`, `correction_metric`, and `log_max_rows`.
- `mining_res` and `null_df` must both contain the column specified in `early_stop_metric`.
- Internally relies on `validate_multiple_tests` for empirical p-values and multiple testing correction.
- Returns a shallow copy of `mining_res`; original input is not modified.
- Logs results via `logger.log_step()` if a logger is provided.

### 🔗 Related Functions
- `validate_multiple_tests`: performs core FDR correction and summary.
- `summarize_fdr_results`: generates markdown or dataframe summary of statistical outcomes.
- `compute_empirical_pvals`: computes one- or two-tailed empirical p-values.

### 🧪 Testing Status
- ✅ Unit tested with `pytest`, including:
  - Config + override merging
  - Logger integration
  - Invalid or missing columns
  - Edge cases (empty inputs, invalid modes)
