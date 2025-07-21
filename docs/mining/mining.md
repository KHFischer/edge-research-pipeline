## 🧠 Function: prepare_dataframe_for_mining

### 📄 What It Does
- Prepares a raw transactional dataframe for rule mining by dropping irrelevant columns, compressing feature columns to low-memory format, optionally deduplicating and sampling the rows, and generating a summary log of reduction steps.

### 🚦 When to Use
- Use this function before applying any rule mining algorithms that expect a binary feature matrix.
- Helpful when working with large transactional datasets where RAM usage or excessive duplicate rows could impact performance.
- Example use case: preparing engineered feature tables for Apriori, RuleFit, or other association rule mining algorithms.

- Do **not** use this function if:
  - Your features are not binary or convertible to binary.
  - You need continuous features preserved as-is (they will be forcibly cast to `uint8`).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Raw transactional dataframe including features and target.
- `date_col` (`str`): Name of the date column to drop.
- `id_cols` (`List[str]`): List of identifier columns to drop.
- `drop_cols` (`List[str]`): Any additional non-feature columns to drop.
- `target_col` (`str`): Name of the target column (must remain in the dataframe).
- `to_sample` (`bool`): Whether to apply stratified sampling based on target (default: True).
- `sample_size` (`int`): Maximum number of rows after sampling (default: 100,000).
- `drop_duplicates` (`bool`): Whether to drop exact duplicate rows (default: False).

**Outputs:**
- `processed_df` (`pd.DataFrame`): Ready-for-mining dataframe with binary features (`uint8`) and target.
- `log_df` (`pd.DataFrame`): Single-row dataframe recording:
  - Rows and columns dropped
  - Features retained
  - Duplicates dropped (if any)
  - Sampling status and row counts
  - RAM usage before and after

### ⚠️ Design Notes / Gotchas
- All non-target feature columns are forced to `uint8` — not optional.
- Target column is converted to `uint8` if it is boolean or strictly binary (2 unique classes).
- Sampling is stratified on the target column for class balance.
- Duplicate dropping removes entire identical rows (feature and target).
- Memory logging uses Pandas' `memory_usage(deep=True)` for approximate MB reporting.
- Does **not mutate** the input dataframe.
- No reliance on global state.

### 🔗 Related Functions
- Intended to prepare data for downstream mining functions like:
  - `mine_rules_apriori()`
  - `mine_rules_rulefit()`
  - `calculate_association_metrics()`

### 🧪 Testing Status
- Covered by unit tests:
  - Typical processing flow.
  - Missing target column (raises `ValueError`).
  - Deduplication behavior.
  - Sampling behavior (triggered / skipped).
  - Log dataframe integrity across options.
- Edge cases (e.g. small datasets, no-op sampling) are explicitly tested.

## 🧠 Function: parse_apriori_rules

### 📄 What It Does
- Converts Apriori-mined antecedent itemsets from a dataframe into a standardized, list-based rule format.
- Each rule represents a conjunction of features that define a mined pattern.

### 🚦 When to Use
- Use after mining association rules using Apriori to transform the raw frozenset-based antecedents into a format compatible with downstream rule processing and feature generation.
- Example: Preparing mined rules for conversion into boolean feature columns for model input or statistics calculation.
- Do **not** use if working with non-Apriori output formats (other miners need their own parsers).

### 🔢 Inputs and Outputs

**Inputs:**
- `apriori_df` (`pd.DataFrame`): 
  - Must include a column containing antecedents as `frozenset` objects.
- `column_name` (`str`):
  - Name of the dataframe column to parse (default: `'antecedents'`).

**Outputs:**
- `parsed_rules` (`List[List[Tuple[str, int]]]`):
  - Each rule is a list of (feature_name, expected_value) tuples.
  - All expected_values are set to `1` for Apriori (indicating feature presence).
  - Example Output:
    ```python
    [
        [('feature1', 1), ('feature2', 1)],
        [('feature3', 1)]
    ]
    ```

### ⚠️ Design Notes / Gotchas
- **Strict Format Assumption:** Input antecedents must be `frozenset` instances. Raises `ValueError` otherwise.
- **Expected Value Fixed at 1:** Assumes Apriori discovers patterns where features are "active" (feature == 1).
- **Non-Mutating:** Does not alter the input dataframe.
- No performance optimizations; input size is typically small after mining.

### 🔗 Related Functions
- `validate_parsed_rules()` — Use to verify output format conforms to rule standard.
- `generate_multivariate_feature_df()` — Consumes parsed rules to create feature columns.

### 🧪 Testing Status
- Unit tested with:
  - Typical rule sets.
  - Empty dataframe input.
  - Missing column detection.
  - Invalid type detection (non-frozenset).
  - Single-feature antecedents.
- All key edge cases covered.

## 🧠 Function: perform_rulefit

### 📄 What It Does
- Applies RuleFit to discover multivariate rules predicting a multiclass or binary target.
- Extracts interpretable, multi-feature rules as logical combinations, excluding linear terms.

### 🚦 When to Use
- Use when you want to mine rule-based patterns from a binary-feature dataset and a categorical target.
- Suitable when:
  - Features are strictly binary (0/1).
  - You need human-readable rules for explainability or downstream feature creation.
- Example: Mining rule combinations like "featA == 1 AND featB == 0" that predict a target class.

- Do **not** use:
  - On datasets with missing feature values (function raises error).
  - If working with continuous features (must pre-binarize).

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`):
  - Input dataframe with binary feature columns and a categorical/binary target column.
- `target_col` (`str`):
  - Name of the target column.
- `tree_size` (`int`):
  - Maximum depth of trees for rule generation.
- `min_rule_depth` (`int`):
  - Minimum number of conditions for rules to be retained (filters out trivial rules).

**Outputs:**
- `all_rules_df` (`pd.DataFrame`):
  - Columns include:
    - `rule` (string): Human-readable rule condition.
    - `consequents`: Predicted target class.
    - `support`: Fraction of samples matching the rule.
    - `depth`: Number of conditions in rule.
- `summary_df` (`pd.DataFrame`):
  - Per-class summary:
    - `target_class`
    - `total_extracted_rules`
    - `rules_retained_multivar`
    - `support_min`, `support_max`, `support_mean`

### ⚠️ Design Notes / Gotchas
- Multiclass handled via one-vs-rest (RuleFit models trained per class).
- Only **logical rules** extracted (linear terms excluded).
- Rules with fewer than `min_rule_depth` conditions are discarded.
- Sorting is by rule support (descending).
- Assumes all feature columns are pre-binarized to 0/1 integers.
- Raises `ValueError`:
  - If target column missing.
  - If any feature contains missing values.

### 🔗 Related Functions
- `parse_rulefit_rules()` – Converts rule strings into standardized rule tuple format.
- `generate_multivariate_feature_df()` – Consumes parsed rules to create boolean feature columns.

### 🧪 Testing Status
- Unit tested with:
  - Binary and multiclass targets.
  - Missing target detection.
  - NaN feature error detection.
  - Parameter variations (`tree_size`, `min_rule_depth`).
- Output structure and correctness validated via black-box tests.

## 🧠 Function: parse_rule_string_to_tuples

### 📄 What It Does
- Converts a single RuleFit rule string into a list of feature-condition pairs.
- Encodes each condition as (feature_name, expected_value), where expected_value is either 0 or 1.

### 🚦 When to Use
- Use after extracting rule strings from a RuleFit model to parse individual rule conditions into a standardized, programmatically usable format.
- Supports simple binary splits:
  - `"feature <= 0.5"` interpreted as feature == 0
  - `"feature > 0.5"` interpreted as feature == 1

- Do **not** use:
  - On rule strings containing other operators (e.g., `>=`, `<`, `==`).
  - On rules containing continuous thresholds (only supports 0.5 splits).
  - On non-standard formats (must use `' and '` to combine conditions).

---

## 🔢 Inputs and Outputs

**Inputs:**
- `rule_str` (`str`):
  - Rule string from RuleFit, using simple threshold logic.

**Outputs:**
- `List[Tuple[str, int]]`:
  - List of (feature_name, expected_value) tuples representing the rule’s conditions.

---

## ⚠️ Design Notes / Gotchas
- Rule strings must be strictly formatted:
  - Combined using `' and '`.
  - Conditions using either `<= 0.5` or `> 0.5`.
- Raises `ValueError` for:
  - Unsupported operators.
  - Unsupported thresholds.
  - Missing operators in a condition.
- Function assumes input rules are pre-validated as RuleFit outputs.
- Does **not** mutate input.
- No tolerance for extra whitespace beyond standard trimming.

---

## 🔗 Related Functions
- `parse_rulefit_rules()` – Processes a dataframe of rule strings using this parser.
- `validate_parsed_rules()` – Ensures parsed output conforms to expected structure.

---

## 🧪 Testing Status
- Unit tested with:
  - Typical rule strings (single and multi-condition).
  - Invalid formats (unsupported operators, bad thresholds).
  - Empty and malformed rules.
- Exception raising behavior verified.
- Tested via black-box input/output validation.

## 🧠 Function: parse_rulefit_rules

### 📄 What It Does
- Parses an entire dataframe column of RuleFit rule strings into standardized, tuple-based rule format.

### 🚦 When to Use
- Use when batch-processing RuleFit rules after mining to generate a uniform, downstream-compatible rule representation.
- Suitable for converting large sets of rules at once.

- Do **not** use:
  - On dataframes missing the rule column.
  - On datasets containing malformed or non-string rule entries.

---

## 🔢 Inputs and Outputs

**Inputs:**
- `rules_df` (`pd.DataFrame`):
  - DataFrame containing rule strings as output from RuleFit.
- `column_name` (`str`):
  - Name of the column containing rule strings (default: `'rule'`).

**Outputs:**
- `List[List[Tuple[str, int]]]`:
  - Each rule represented as a list of (feature_name, expected_value) tuples.

---

## ⚠️ Design Notes / Gotchas
- Assumes rule strings use simple binary splits and 'and' combination logic.
- Missing column or non-string entries raise `ValueError`.
- Automatically calls `validate_parsed_rules()` to enforce output format correctness.
- Function is non-mutating and returns a new parsed list.

---

## 🔗 Related Functions
- `parse_rule_string_to_tuples()` – Called internally to parse each rule.
- `validate_parsed_rules()` – Validates parsed output format.

---

## 🧪 Testing Status
- Unit tested with:
  - Typical dataframe inputs.
  - Malformed inputs (missing column, bad types).
  - Empty dataframe handling.
  - Exception handling for invalid cases.

## 🧠 Function: perform_subgroup_discovery

### 📄 What It Does
- Performs subgroup discovery using pysubgroup to mine multivariate, interpretable rules predicting classes of a multiclass (or binary) target.
- Returns discovered rules and per-class summary statistics.

### 🚦 When to Use
- Use when you need interpretable AND-based rules from a dataset with binary features and a categorical target.
- Useful for feature discovery, model explainability, or downstream rule-based modeling.
- Example: Mining patterns like "featA == True AND featB == False" that predict a given target class.

- Do **not** use:
  - On continuous or non-binary features (must binarize first).
  - On datasets where target column is missing.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`):
  - Binary-feature dataframe with a categorical target column.
- `target_col` (`str`):
  - Name of the target column.
- `top_n` (`int`):
  - Number of top rules to retain per target class (default: 50).
- `depth` (`int`):
  - Maximum number of AND-conditions allowed per rule (default: 3).
- `beam_width` (`int`):
  - Search beam width controlling exploration (default: 50).
- `qf` (optional pysubgroup quality function):
  - Quality function to optimize (default: WRAccQF).

**Outputs:**
- `all_rules_df` (`pd.DataFrame`):
  - Contains:
    - `rule`: Human-readable rule string.
    - `consequents`: Predicted target class.
    - `depth`: Number of conditions in the rule.
    - `quality`: Rule quality score (per qf).
- `summary_df` (`pd.DataFrame`):
  - Per-class summary of rule discovery including:
    - `total_raw_rules`, `rules_retained_multivar`, `quality_mean`, etc.

### ⚠️ Design Notes / Gotchas
- Internally one-hot encodes target for multiclass handling.
- Rules containing target conditions are ignored.
- Only multivariate rules (depth > 1) are retained in results.
- Raises `ValueError` if target column missing.
- Assumes features are strictly binary (internally coerced to boolean).
- Beam Search is used for subgroup discovery with adjustable breadth and depth.

### 🔗 Related Functions
- `parse_subgroup_rules()` — Parses discovered rules into standardized feature-value condition format.
- `validate_parsed_rules()` — Ensures structural correctness of parsed rules.

### 🧪 Testing Status
- Unit tested with:
  - Typical multiclass datasets.
  - Missing target handling.
  - Empty dataframe edge cases.
  - Beam width and parameter variations.
- Verified output dataframe structure and content.

## 🧠 Function: parse_subgroup_rule_to_tuples

### 📄 What It Does
- Converts a single subgroup rule string (from pysubgroup) into a standardized list of feature-condition pairs.
- Each rule condition is represented as (feature_name, expected_value), where expected_value is 0 or 1.

### 🚦 When to Use
- Use after subgroup discovery to parse individual rule strings for downstream processing, such as rule evaluation or feature generation.
- Do **not** use if your rule strings use unsupported operators (only '== True' and '== False' are handled).

### 🔢 Inputs and Outputs

**Inputs:**
- `rule_str` (`str`):
  - The rule string to parse (expected format: AND-combined equality conditions).
- `target_prefix` (`str`):
  - Feature name prefix used to ignore target-related conditions (default: `'target_'`).

**Outputs:**
- `List[Tuple[str, int]]`:
  - Parsed conditions as a list of (feature_name, expected_value) tuples.

### ⚠️ Design Notes / Gotchas
- Only parses:
  - `feature == True` → 1
  - `feature == False` → 0
- Multiple conditions must be joined with `'AND'`.
- Conditions with feature names starting with `target_prefix` are skipped.
- Raises `ValueError` if rule parts are improperly formatted.
- No tolerance for other comparison operators (e.g. `>=`, `!=`, etc.).

### 🔗 Related Functions
- `parse_subgroup_rules()` — Parses multiple rules from a dataframe column.
- `validate_parsed_rules()` — Enforces parsed output structure correctness.

### 🧪 Testing Status
- Unit tested with:
  - Single and multi-condition rules.
  - Empty strings.
  - Skipped target-prefixed conditions.
  - Malformed input triggers `ValueError`.

## 🧠 Function: normalize_and_dedup_rules

### 📄 What It Does
- Normalizes, deduplicates, and aggregates multivariate rules mined from different algorithms.
- Tracks provenance of each unique rule and counts unique rules per algorithm.

### 🚦 When to Use
- Use after mining rule candidates from multiple algorithms (e.g. Apriori, RuleFit, Subgroup Discovery).
- Ensures a clean, non-redundant rule set with algorithm source tracking.
- Enables accurate counting of unique contributions from each algorithm.

- Do **not** use:
  - If input rules are not in parsed, feature-condition pair format.
  - If provenance tracking is unnecessary.

### 🔢 Inputs and Outputs

**Inputs:**
- `rule_sources` (`List[Tuple[str, List[List[Tuple[str, int]]]]]`):
  - List of (`algorithm_name`, `rules`) pairs.
  - Each rule is a list of (`feature_name`, `expected_value`) tuples.

**Outputs:**
- `deduplicated_rules` (`List[Tuple[List[Tuple[str, int]], Set[str]]]`):
  - Unique, normalized rules paired with a set of contributing algorithms.
- `rule_count_df` (`pd.DataFrame`):
  - Dataframe summarizing how many unique rules each algorithm contributed.
  - Columns: `algorithm`, `unique_rule_count`.

### ⚠️ Design Notes / Gotchas
- Assumes input rules are pre-parsed into feature-condition tuples.
- Normalization guarantees that feature order does not affect rule equality.
- Rules appearing in multiple algorithms retain all provenance sources.
- Returns a pandas dataframe for easy visualization and downstream reporting.
- No side effects or global state dependencies.
- Processing is in-memory and dictionary-backed — scales well for typical rule mining tasks.

### 🔗 Related Functions
- `normalize_rule()` – Produces canonical form of a rule via sorting.
- `deduplicate_rules_with_provenance()` – Deduplicates rules while retaining provenance.
- `count_rules_per_algorithm()` – Tallies unique rules per contributing algorithm.

### 🧪 Testing Status
- Unit tested with:
  - Overlapping rules across algorithms.
  - Correct provenance attribution.
  - Empty input handling.
  - Output dataframe correctness.

## 🧠 Function: generate_rule_activation_dataframe

### 📄 What It Does
- Converts a list of multivariate rules into a boolean feature dataframe, where each column indicates whether a rule is satisfied for each row of the input dataframe.
- Generates a human-readable mapping of each rule to aid interpretability.

### 🚦 When to Use
- Use after mining and deduplicating multivariate rules when you need to:
  - Transform rules into actionable features for machine learning pipelines.
  - Generate rule activation masks for further analysis or statistical testing.
  - Retain the original target variable for supervised learning workflows.

- Do **not** use:
  - If your dataframe columns are not strictly binary or integer-encoded.
  - If your rules are not pre-normalized as (feature_name, expected_value) tuples.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`):
  - Dataframe of binary features and a target column.
- `unique_rules` (`List[Tuple[List[Tuple[str, int]], Set[str]]]`):
  - List of deduplicated, normalized rules. Each rule is a list of (feature_name, expected_value) conditions, paired with the set of algorithms that generated it.
- `target_col` (`str`):
  - Name of the target column to retain in the output dataframe.
- `prefix` (`str`):
  - Prefix string for naming the rule columns (default: `'rule'`).

**Outputs:**
- `rule_df` (`pd.DataFrame`):
  - DataFrame where:
    - Each rule column contains boolean indicators (True/False) for whether the rule is satisfied in each row.
    - The original target column is retained.
- `mapping_df` (`pd.DataFrame`):
  - DataFrame mapping rule column names to human-readable string representations of the rule logic.

### ⚠️ Design Notes / Gotchas
- Requires that all rule features exist as columns in the input dataframe — raises `KeyError` otherwise.
- Raises `ValueError` if target column is missing.
- Handles multi-condition rules using vectorized numpy operations for efficiency.
- Produces interpretable rule expressions using standard AND logic.
- Does **not** support missing data in feature columns.
- Preserves row order and index from the input dataframe.

### 🔗 Related Functions
- `normalize_and_dedup_rules()` – Deduplicates and normalizes rules before this step.
- `parse_apriori_rules()`, `parse_rulefit_rules()`, `parse_subgroup_rules()` – Functions that generate rule condition lists.

### 🧪 Testing Status
- Unit tested with:
  - Typical multiclass and binary datasets.
  - Missing feature or target columns triggering exceptions.
  - Edge case of no rules (empty input).
  - Custom rule column naming.
