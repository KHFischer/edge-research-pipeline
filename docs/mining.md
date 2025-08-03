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

## 🧠 Function: compute_rule_depth

### 📄 What It Does
- Computes how many base feature conditions a rule contains, based on its human-readable representation.

### 🚦 When to Use
- Use after generating human-readable rules to:
  - Quantify rule complexity.
  - Filter rules by depth (e.g. select only 2-condition rules).
- Not useful if rules aren't consistently formatted with `' AND '` separators.

### 🔢 Inputs and Outputs

**Inputs:**
- `rule_str` (`str`):
  - Human-readable rule expression, e.g. `"('featureA' == 1) AND ('featureB' == 0)"`.

**Outputs:**
- `int`:
  - Rule depth (number of base conditions).
  - Returns 0 for empty or malformed strings.

### ⚠️ Design Notes / Gotchas
- Rule string must use `' AND '` as separator.
- Case-sensitive match for `' AND '` (does not match lowercase).
- Gracefully handles nulls and empty strings.
- Pure function with no side effects.

### 🔗 Related Functions
- `merge_multivar_map_into_stats()` – Generates human-readable rules.

### 🧪 Testing Status
- Unit tested:
  - Single- and multi-condition rules.
  - Empty/null/whitespace handling.
  - Edge cases like empty rules returning zero.

## 🧠 Function: `perform_elcs`

### 📄 What It Does
- Trains a one-vs-rest eLCS (Learning Classifier System) model for each class in a multiclass target column.
- Extracts all learned rules from each model and returns them in a standardized format, along with per-class summary statistics.

### 🚦 When to Use
- Use this function when you have a prepared feature dataframe (binary or categorical features) and need to mine multivariate classification rules using eLCS.
- Useful as part of a larger rule mining or feature generation pipeline, especially where interpretability is desired.
- Not suitable if your features are continuous (requires pre-binarization) or your target column is missing.

Example use case:
- Mining binary activation rules from discretized financial signals to create new features for downstream modeling.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - DataFrame containing input features and a target column.
  - Features must be binary or categorical (already preprocessed).
- `target_col` (`str`):
  - Name of the target column in `df`.
  - Can be multiclass; function handles binary decomposition internally.

**Outputs:**

- `all_rules` (`List[List[Tuple[str, int]]]`):
  - Flat list of all discovered rules across all classes.
  - Each rule is represented as a list of `(feature_name, expected_value)` conditions.
- `log_df` (`pd.DataFrame`):
  - Summary DataFrame with one row per target class.
  - Columns:
    - `target_class`: Class label.
    - `n_rules`: Number of rules found for that class.
    - `avg_depth`: Average number of antecedents per rule.
    - `avg_fitness`: Mean fitness score of rules.
    - `avg_accuracy`: Mean empirical accuracy of rules.

### ⚠️ Design Notes / Gotchas
- Requires features to be binary or binarized before use.
- Target column must be categorical or discrete.
- Function assumes eLCS is properly installed and imported (`skeLCS` dependency).
- Returns all raw rules without filtering — downstream steps should handle pruning, scoring, or deduplication.
- Does not track or return predicted class labels (`phenotype`) — rules are mined as logical conditions only.
- Performance can degrade on very large datasets due to eLCS evolutionary mechanics.
- Does not mutate input DataFrame.

### 🔗 Related Functions
- `validate_parsed_rules()` — downstream rule validation.
- `normalize_and_dedup_rules()` — useful post-processing step after rule extraction.
- Other mining functions: `perform_apriori()`, `perform_rulefit()`, `perform_subgroup_discovery()`.

### 🧪 Testing Status
- Covered by unit tests in `test_perform_elcs_typical_case()`, `test_perform_elcs_single_class()`, and others.
- Edge cases tested:
  - Missing target column.
  - Single-class targets.
  - Variable dataset shapes.
- No mocks required; function is self-contained.

## 🧠 Function: `df_to_orange_table`

### 📄 What It Does
- Converts a pandas DataFrame of binary feature columns and a categorical target column into an Orange Table.
- This prepares the dataset for use with Orange3 algorithms such as CN2 rule induction.

### 🚦 When to Use
- Use when you need to pass binarized tabular data into Orange3’s rule induction learners, especially `CN2Learner`.
- Input must be preprocessed to have binary (0/1) features and a discrete target.
- Do not use if your dataset contains continuous or non-binary feature columns.

Example: Converting one-hot encoded stock sector indicators and forward-return class labels for rule mining.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - Input dataframe containing:
    - Binary feature columns (0/1 or bool).
    - Categorical target column.
- `target_col` (`str`):
  - Name of the target column (default: `"forward_return"`).
  - Must exist in `df` and contain discrete values.

**Outputs:**

- `Orange.data.Table`:
  - Features as discrete variables with values `["0", "1"]`.
  - Class variable with values matching the unique, sorted class labels from the target column.

### ⚠️ Design Notes / Gotchas
- Raises:
  - `KeyError` if the target column is missing.
  - `ValueError` if any feature column contains non-binary data.
- Target labels retain their original class names (string labels).
- Feature columns are strictly treated as discrete categorical variables, not numeric.
- Does not mutate input dataframe.
- Assumes Orange3 is installed and available at runtime.

### 🔗 Related Functions
- `perform_cn2()` – downstream rule mining function using Orange CN2Learner.
- `perform_elcs()` – alternative rule mining pipeline.
- `validate_parsed_rules()` – useful downstream for checking parsed rule outputs.

### 🧪 Testing Status
- Unit tested using pytest:
  - Normal inputs.
  - Single-row input.
  - Missing target column (exception).
  - Non-binary features (exception).
  - Empty dataframe (edge case).

## 🧠 Function: `perform_cn2`

### 📄 What It Does
- Runs CN2 rule induction using Orange3 to discover logical rules from a tabular dataframe.
- Extracts rules as standardized feature-activation conditions, ignoring predicted class outputs.

### 🚦 When to Use
- Use when you have a dataframe of binarized or categorical features and need to generate multivariate rules via CN2.
- Suitable for feature discovery, interpretable model construction, or rule-based system generation.
- Not suitable for continuous features without prior discretization.

Example: Mining explainable classification rules from a discretized financial dataset.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - Feature dataframe containing all predictors and the target column.
  - Features must be pre-binarized or categorical (no continuous columns).

- `target_col` (`str`):
  - Name of the discrete target column to predict.

**Outputs:**

- `parsed_rules` (`List[List[Tuple[str, int]]]`):
  - Flat list of discovered rules.
  - Each rule is a list of conditions: `(feature_name, expected_value)` pairs.

- `log_df` (`pd.DataFrame`):
  - Single-row dataframe summarizing:
    - `'n_rules'`: Number of discovered rules.
    - `'avg_depth'`: Average number of conditions per rule.

### ⚠️ Design Notes / Gotchas
- Requires your dataframe to be preprocessed into discrete features.
- Rules with unsupported operators (e.g., inequalities) are silently skipped.
- Predicted class (consequent) is ignored—rules describe only feature activations.
- Relies on external `df_to_orange_table()` utility to convert pandas dataframe to Orange Table format.
- Does not mutate inputs.
- Assumes Orange3 is installed and available at runtime.

### 🔗 Related Functions
- `df_to_orange_table()` – pandas-to-Orange conversion utility (must be provided separately).
- `perform_elcs()` – for rule mining via Learning Classifier Systems.
- `validate_parsed_rules()` – optional downstream rule format validator.
- `generate_rule_activation_dataframe()` – optional feature generation from rules.

### 🧪 Testing Status
- Unit tested under varied dataset shapes, single-class targets, and missing target error handling.
- Edge cases tested:
  - Empty dataframe.
  - Single-class datasets.
  - Missing target column.

## 🧠 Function: `perform_cart`

### 📄 What It Does
- Trains a CART (Classification and Regression Tree) using scikit-learn’s `DecisionTreeClassifier`.
- Extracts each root-to-leaf path as a rule in standardized feature-activation format.
- Returns both the extracted rules and summary statistics about the trained tree.

### 🚦 When to Use
- Use when you need to generate multivariate rules from binary or numeric features using a decision tree.
- Ideal for building interpretable rule-based models, feature generators, or edge detection logic.
- Not suitable for datasets requiring multi-condition splits beyond binary or numerical encoding.

Example: Converting financial signal features into logical trading rules using a decision tree.

### 🔢 Inputs and Outputs

**Inputs:**

- `df` (`pd.DataFrame`):
  - Input dataframe with feature columns (binary or numeric) and a discrete target column.
- `target_col` (`str`):
  - Name of the target column (multiclass or binary).
- `max_depth` (`Optional[int]`, default `5`):
  - Maximum allowed depth of the decision tree.
- `criterion` (`str`, default `'gini'`):
  - Splitting criterion, either `'gini'` or `'entropy'`.
- `random_state` (`Optional[int]`, default `42`):
  - Random seed for reproducibility.
- `min_samples_split` (`int`, default `2`):
  - Minimum number of samples required to split a node.
- `min_samples_leaf` (`int`, default `1`):
  - Minimum samples required at a leaf node.

**Outputs:**

- `parsed_rules` (`List[List[Tuple[str, int]]]`):
  - List of extracted rules.
  - Each rule is represented as a list of `(feature_name, expected_value)` tuples.
- `log_df` (`pd.DataFrame`):
  - Single-row summary dataframe:
    - `'n_rules'`: Total number of extracted rules.
    - `'avg_depth'`: Average number of conditions per rule.
    - `'tree_depth'`: Maximum depth of the trained tree.

### ⚠️ Design Notes / Gotchas
- Returns only antecedents (feature conditions). Predicted classes per rule are ignored.
- Features must be pre-binarized or numeric.
- Rules are extracted as simple paths from root to leaf—no pruning, scoring, or post-filtering is applied.
- Validates output rule format using `validate_parsed_rules()` before returning.
- Function does not mutate input dataframes.
- Multiclass targets are handled natively by the decision tree.

### 🔗 Related Functions
- `validate_parsed_rules()` – downstream rule format validation.
- `generate_rule_activation_dataframe()` – to convert rules into usable feature columns.
- `perform_cn2()`, `perform_elcs()` – other rule mining algorithms in this pipeline.

### 🧪 Testing Status
- Unit tested via pytest:
  - Typical multiclass datasets.
  - Single-row edge cases.
  - Missing target column (raises ValueError).
  - Parameterized max_depth control.
  - Empty dataframe handling.

## 🧠 Function: generate_synthetic_data_sdv

### 📄 What It Does
- Generates synthetic tabular data from a real dataframe using SDV's synthesizers (Gaussian Copula, CTGAN, or TVAE).
- Returns both the generated synthetic dataset and a metadata summary with per-column quality diagnostics.
- Silences verbose internal logs and warnings from SDV libraries by default.

---

### 🚦 When to Use
- Use when you need to generate realistic synthetic data for testing, augmentation, or privacy-preserving analysis.
- Appropriate for structured, tabular datasets without missing values.
- Supports small and medium datasets typically used in finance, research, or ML pipelines.
- Do **not** use if your dataframe contains missing values (must be cleaned first).
- Not optimized for sequential, relational, or time-series data.

---

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Clean input dataframe (must have no missing values).
- `num_rows` (`int`): Number of synthetic rows to generate.
- `model` (`str`, optional): Choice of synthesizer backend:
  - `'gaussian_copula'` (default)
  - `'ctgan'`
  - `'tvae'`
- `verbose` (`bool`, optional): If `True`, prints an overall quality score. Defaults to `False`.

**Outputs:**
- `synthetic_data` (`pd.DataFrame`): Generated synthetic dataset with same columns as input.
- `metadata_df` (`pd.DataFrame`): Summary dataframe with:
  - `column_name`
  - `sdtype` (SDV-inferred semantic data type)
  - Per-column quality scores (e.g., shape similarity).

---

### ⚠️ Design Notes / Gotchas

- Raises `ValueError` if input dataframe contains missing values or invalid model name.
- Does **not modify** the input dataframe.
- Internally detects column data types using SDV's metadata detection (based on column content).
- Silences:
  - stdout / stderr
  - warnings
  - SDV / Copulas / RDT loggers (unless `verbose=True`)
- Fitting and sampling are done within a logging suppression context.
- Quality score evaluation uses SDV's built-in comparison metrics (`evaluate_quality()`).
- Metadata summary merges column data types and individual column quality scores.
- Performance: Training time depends on synthesizer type and dataset size (CTGAN and TVAE will be slower than Gaussian Copula).
- Warning suppression inside SDV internals may hide non-critical library warnings—use caution in debug scenarios.

---

### 🔗 Related Functions

- `force_silence()` — suppresses global stdout, stderr, warnings, and logs.
- `_suppress_sdv_logs()` — specifically targets SDV-related loggers.
- `evaluate_quality()` — SDV function used internally for synthetic data evaluation.

---

### 🧪 Testing Status

- Unit tests implemented:
  - Basic generation across all model types.
  - Invalid model exception handling.
  - Missing values exception handling.
  - Edge cases: single-row and large-row generation.
  - Verbose mode output check.
- All outputs validated for shape and structure.
- No known edge cases untested at this stage.

## 🧠 Function: generate_synthetic_data_synthcity

### 📄 What It Does
- Trains a selected Synthcity tabular generative model on a real dataset and returns synthetic samples.
- Evaluates the quality of the synthetic data using SDV’s `evaluate_quality()` function for consistency across generators.

---

### 🚦 When to Use
- Use this function to quickly generate realistic synthetic tabular data from any clean DataFrame, optionally conditioned on a target column.
- Ideal for augmenting training data, benchmarking robustness, or validating the statistical similarity of generated data.
- Avoid using on datasets with missing values or highly nested column structures (multi-index, JSON-like columns), as neither Synthcity nor SDV handle those well.

---

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): Clean input dataset with consistent datatypes and no missing values.
- `target_col` (`Optional[str]`): Name of target column to condition on (or `None` for unconditional generation).
- `n_rows` (`int`): Number of synthetic rows to generate.
- `model` (`str`): Name of the Synthcity plugin to use. Must be one of `'ctgan'`, `'tvae'`, `'rtvae'`, `'adsgan'`, `'pategan'`.
- `n_iter` (`int`): Number of training iterations (default: `1000`).
- `batch_size` (`int`): Training batch size (default: `128`).
- `lr` (`float`): Learning rate (default: `1e-4`).
- `device` (`str`): Compute device (`'cpu'` or `'cuda'`; default: `'cpu'`).

**Outputs:**
- `synthetic_df` (`pd.DataFrame`): Generated synthetic data, with same schema as input `df`.
- `metadata_df` (`pd.DataFrame`): SDV-inferred column types and per-column shape similarity scores.

---

### ⚠️ Design Notes / Gotchas

- The function wraps Synthcity’s training and generation pipeline, but uses SDV for quality scoring due to the complexity of Synthcity's native benchmarking API.
- It assumes `df` is already cleaned — any missing values will cause downstream model or metadata generation failures.
- The Synthcity plugin must be listed in `Plugins().list()`, or a `ValueError` will be raised.
- The `target_col`, if provided, must be present in `df` and treated as a column to condition generation on.
- Performance: GAN-based plugins like `ctgan` and `adsgan` are slower than VAE-based models like `tvae` or `rtvae`, especially with high `n_iter`.

---

### 🔗 Related Functions
- `evaluate_quality` (SDV): Used internally to generate quality scores.
- `GenericDataLoader` (Synthcity): Wraps real and synthetic data for training.
- `Plugins().get(...)` (Synthcity): Fetches model plugin with custom hyperparameters.

---

### 🧪 Testing Status

- ✅ Unit tested with:
  - Multiple valid model types
  - Inclusion/exclusion of target column
  - Validation of output structure and expected column names
  - Proper exception handling for unsupported models and empty DataFrames
- Edge cases like one-row datasets and very small batch sizes are also covered.

## 🧠 Function: apply_class_imbalance

### 📄 What It Does
- Rebalances a dataset according to a target class distribution.
- If no distribution is provided, it generates a skewed one favoring minority classes.
- Uses the Badgers library to resample tabular data with the desired class proportions.

### 🚦 When to Use
- Use this function to simulate class imbalance for robustness testing or stress-testing classification logic.
- Helpful in training pipelines that must handle noisy or imbalanced real-world label distributions.
- Not appropriate when exact replication of the original distribution is needed (i.e., no distortion).

### 🔢 Inputs and Outputs

#### Inputs:
- `df` (`pd.DataFrame`): The input dataset, must include the `target_col`.
- `target_col` (`str`): The name of the column to rebalance based on class values.
- `proportions` (`dict`, optional): Mapping of class → probability (must sum to 1.0). If omitted, one is auto-generated.
- `random_state` (`int`, optional): Seed for deterministic resampling.

#### Outputs:
- `pd.DataFrame`: A new dataframe with rows sampled to match the given (or generated) class distribution.

### ⚠️ Design Notes / Gotchas
- Input `target_col` must exist in the dataframe or a `ValueError` is raised.
- If `proportions` is not passed, the function uses `generate_skewed_proportions` to overweight minority classes.
- The Badgers resampler works by oversampling or undersampling — downstream consumers should not assume row-wise identity with the input.
- Currently only supports classification-style categorical target columns.

### 🔗 Related Functions
- [`generate_skewed_proportions`](#function-generate_skewed_proportions): Utility used internally to construct default class weights.
- Badgers: [Fraunhofer IESE Badgers GitHub](https://github.com/Fraunhofer-IESE/badgers)

### 🧪 Testing Status
- Unit tested with Pytest for:
  - Typical 2-class and 3-class rebalancing
  - Custom vs. auto-generated proportions
  - Error handling (missing column, invalid inputs)
  - Distribution shape verification

## 🧠 Function: flip_boolean_values

### 📄 What It Does
- Randomly flips boolean values in specified columns of a DataFrame.
- Useful for introducing controlled corruption into binary features for stress testing or robustness evaluation.

### 🚦 When to Use
- Use this to simulate noisy or imperfect binary inputs (e.g. mislabeled indicators, faulty sensors).
- Works best when the target columns are strictly boolean (`dtype=bool`).
- Avoid using on integer or float columns — it explicitly raises an error on incompatible types.

### 🔢 Inputs and Outputs

#### Inputs:
- `df` (`pd.DataFrame`): The input dataframe with boolean columns.
- `columns` (`list[str]`, optional): List of column names to flip. If `None`, all boolean columns are flipped.
- `flip_fraction` (`float`): Proportion of rows to flip per column (must be between 0.0 and 1.0).
- `seed` (`int`): Seed for reproducibility.

#### Output:
- `pd.DataFrame`: A copy of the original dataframe with selected boolean values flipped.

### ⚠️ Design Notes / Gotchas
- Only works on columns with `bool` dtype. Will raise `TypeError` for non-bool columns.
- Will raise `ValueError` if `flip_fraction` is outside the [0, 1] range.
- Flipping is done independently per column; rows may be flipped in more than one column.

### 🔗 Related Functions
- [`flip_labels`](#function-flip_labels): Flips label values in a categorical target column.
- Used as part of the synthetic data corruption module.

### 🧪 Testing Status
- Unit tested with pytest:
  - Verifies correctness for selected columns and full-bool input
  - Includes failure cases (bad dtype, invalid fraction)
  - Confirms column integrity and value flips

## 🧠 Function: `generate_combined_synthetic_data`

### 📄 What It Does
- Generates synthetic tabular data using one or both of two engines: SDV and Synthcity.
- Returns a single concatenated dataframe along with per-generator metadata logs.

### 🚦 When to Use
- Use this function when you want to expand a dataset using realistic synthetic samples for modeling, validation, or robustness testing.
- Useful in cases where:
  - Your dataset is too small or imbalanced.
  - You want to benchmark model performance on synthetic data.
  - You want to simulate edge cases or hypothetical feature-target relationships.

**Example use case**: augmenting financial time series with generated samples before training a rule mining model.

**Avoid if**:
- Your data contains missing values (they must be preprocessed first).
- You don’t want any form of artificial data in your pipeline.

### 🔢 Inputs and Outputs

#### Inputs
- `df: pd.DataFrame`  
  The clean, real dataset to base synthetic generation on. No NaNs allowed.
- `target_col: Optional[str]`  
  Optional target column for conditional generation (used only by Synthcity).
- `to_sdv: bool`  
  Whether to run SDV-based generation.
- `to_synthcity: bool`  
  Whether to run Synthcity-based generation.
- `sdv_model: str`  
  SDV model name (`"gaussian_copula"`, `"ctgan"`, or `"tvae"`).
- `sdv_rows: int`  
  Number of rows to generate via SDV.
- `sdv_verbose: bool`  
  Whether to display SDV quality metrics during generation.
- `sc_model: str`  
  Synthcity model/plugin name (`"ctgan"`, `"tvae"`, etc.).
- `sc_rows: int`  
  Number of rows to generate via Synthcity.
- `sc_n_iter: int`  
  Number of training iterations for Synthcity.
- `sc_batch_size: int`  
  Training batch size for Synthcity.
- `sc_lr: float`  
  Learning rate for Synthcity training.
- `sc_device: str`  
  Device to train Synthcity model on (`"cpu"` or `"cuda"`).
- `silence: bool`  
  If True, suppresses all logging, stdout/stderr, and warnings.

#### Outputs
- `Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]`  
  - A single dataframe with all generated rows (from SDV and/or Synthcity).
  - A dictionary of per-generator metadata logs (keys: `"sdv"`, `"synthcity"`).

### ⚠️ Design Notes / Gotchas
- Raises `RuntimeError` if both `to_sdv` and `to_synthcity` are set to False.
- Input `df` must be non-empty and fully preprocessed — no NaNs, no unsupported dtypes.
- Logging suppression (`silence=True`) disables not just print/logs, but also library warnings.
- Target column is only used by Synthcity and will be ignored by SDV even if provided.
- SDV and Synthcity each return their own log dataframe — format varies by backend.
- Resulting synthetic dataframes are concatenated row-wise, with no provenance column unless manually added later.

### 🔗 Related Functions
- `generate_synthetic_data_sdv`
- `generate_synthetic_data_synthcity`
- `force_silence` (log suppression context manager)

### 🧪 Testing Status
- ✅ Covered by `pytest` unit tests with generator patching and parameterized cases
- Tests include:
  - Valid calls for each backend individually and together
  - Exception raised when both generators are off
  - Shape and structure validation of the output
- Edge cases like empty input or malformed schema are tested for expected failure

## 🧠 Function: `augment_dataset`

### 📄 What It Does
- Applies one or more data augmentation techniques to a DataFrame: class imbalance adjustment, boolean feature flipping, and label flipping.
- Returns a new dataset with the requested transformations applied in-place.

### 🚦 When to Use
- Use this when you want to:
  - Simulate label noise or feature corruption for robustness testing.
  - Adjust class distributions to test model sensitivity or mimic real-world imbalance.
  - Enrich synthetic datasets with controlled perturbations.

**Example**: After generating synthetic data, apply label flipping and imbalance to simulate noisy deployment environments.

**Avoid if**:
- Your dataset has missing values or unsupported dtypes (the function assumes pre-cleaned input).
- You need per-step output separation — this returns a single, final DataFrame.

### 🔢 Inputs and Outputs

#### Inputs
- `df: pd.DataFrame`  
  The input dataset to augment.
- `target_col: str`  
  Column name to treat as the classification target.
- `to_aug_imbalance: bool`  
  Whether to resample classes using real or synthetic proportions.
- `to_aug_flip_feats: bool`  
  Whether to randomly flip values in boolean features.
- `to_aug_flip_targets: bool`  
  Whether to randomly flip labels in the target column.
- `flip_feats_frac: float`  
  Fraction of values to flip per boolean feature (if enabled).
- `flip_targs_frac: float`  
  Fraction of target labels to flip (if enabled).
- `imbalance_proportions: Optional[Dict]`  
  Optional dictionary specifying class label → desired proportion.
- `random_state: int`  
  Seed for reproducibility in all stochastic operations.

#### Output
- `pd.DataFrame`  
  A new DataFrame with all enabled augmentation steps applied sequentially.

### ⚠️ Design Notes / Gotchas
- Does **not** modify the input `df`; always returns a copy.
- Flip fractions must be between `0.0` and `1.0`; enforced via `ValueError`.
- Assumes the target column exists — fails early if not.
- Boolean feature flipping only applies to columns with dtype `bool`.
- Resampling logic uses external `apply_class_imbalance()` which may shrink or expand the row count.

### 🔗 Related Functions
- `apply_class_imbalance` — controls class-level sampling
- `flip_boolean_values` — random boolean noise injection
- `flip_labels` — label flipping for robustness tests

### 🧪 Testing Status
- ✅ Unit tested via `pytest` with:
  - Normal, partial, and full augment scenarios
  - Edge case validation (missing target column, bad fractions)
  - Structure and shape checks on outputs

## 🧠 Function: `mine_stats`

### 📄 What It Does
- Runs one or more rule mining algorithms on a given dataframe and returns combined rule statistics, per-miner logs, and rule provenance.
- Supports both univariate statistics and multivariate rule mining across several configurable algorithms.

### 🚦 When to Use
- Use this function when you want to apply multiple rule mining techniques in a unified pipeline and compare their outputs.
- Ideal for batch mining workflows, dashboard generation, and edge discovery pipelines.
- **Avoid if** you only want to run a single miner with custom handling — use the individual `mine_*` functions instead.

### 🔢 Inputs and Outputs

#### Inputs
- `df: pd.DataFrame`  
  Cleaned and preprocessed dataset to mine.
- `target_col: str`  
  Name of the target variable column.
- `miners: List[str]`  
  List of miners to execute. Supported: `'univar'`, `'apriori'`, `'rulefit'`, `'subgroup'`, `'elcs'`, `'cn2'`, `'cart'`.
- `cfg: Any`  
  Configuration object passed to the univariate stat function and stat calculator.
- `apriori_min_support: float`  
  Minimum support for Apriori mining.
- `apriori_metric: str`  
  Rule quality metric for Apriori (e.g., `'lift'`, `'confidence'`).
- `apriori_min_metric: float`  
  Minimum threshold for Apriori metric.
- `rulefit_tree_size: int`  
  Tree depth for RuleFit.
- `rulefit_min_depth: int`  
  Minimum rule length to retain from RuleFit.
- `subgroup_top_n: int`  
  Max number of subgroup rules per class.
- `subgroup_depth: int`  
  Maximum rule depth for subgroup mining.
- `subgroup_beam_width: int`  
  Beam width for subgroup search.
- `cart_max_depth: int`  
  Max tree depth for CART miner.
- `cart_criterion: str`  
  CART split criterion (`"gini"` or `"entropy"`).
- `cart_random_state: int`  
  Random seed for reproducibility.
- `cart_min_samples_split: int`  
  Minimum samples required to split a node.
- `cart_min_samples_leaf: int`  
  Minimum samples required at a leaf node.

#### Outputs
- `final_stats_df: pd.DataFrame`  
  Combined statistics for all mined rules, with metrics and provenance.
- `logs: Dict[str, pd.DataFrame]`  
  Dictionary of miner-specific logs, keyed by miner name.
- `rules_df: pd.DataFrame`  
  Rule provenance dataframe showing unique rule counts per miner.

### ⚠️ Design Notes / Gotchas
- Input dataframe must be fully cleaned — no missing values or unexpected dtypes.
- Unrecognized miner names will raise a `ValueError`.
- If no miners are selected, the result will be an empty dataframe.
- Deduplication of multivariate rules happens after all miners run.
- Univariate and multivariate statistics are combined if both are run.
- The `cfg` object is assumed to be compatible with `generate_statistics` and `mine_univar` — it is not validated inside this function.
- `rules_df` includes a row for univariate rules if present.

### 🔗 Related Functions
- `mine_univar` — runs univariate statistics.
- `mine_multivar` — combines, deduplicates, and scores rules.
- `mine_apriori`, `mine_rulefit`, `mine_cart`, etc. — individual miners called internally.
- `generate_statistics`, `generate_rule_activation_dataframe` — part of the stats pipeline.

### 🧪 Testing Status
- ✅ Covered by unit tests using `pytest`
- Tests include:
  - No miners
  - Only univariate
  - Only multivariate
  - Mixed miner combinations
  - Invalid miner names
- All outputs are validated for structure and content

## 🧠 Function: `coalesce_data`

### 📄 What It Does
- Combines real, synthetic, and optionally augmented datasets into a single dataframe for downstream analysis.
- Prioritizes augmented datasets over their original versions when available.

### 🚦 When to Use
- Use this function when you need to merge any combination of:
  - a cleaned real dataset,
  - its augmented variant (e.g., noisy labels or corrupted features),
  - a synthetic dataset,
  - and/or its augmented variant.
- It's helpful in pipelines where synthetic data and corruption are optional but the downstream miner expects a unified input.

### 🔢 Inputs and Outputs
- **Inputs:**
  - `real_df` (`pd.DataFrame`): Required original dataset after preprocessing.
  - `synth_df` (`Optional[pd.DataFrame]`): Optional synthetic data to include.
  - `augmented_real_df` (`Optional[pd.DataFrame]`): Optional corrupted/augmented version of the real data.
  - `augmented_synth_df` (`Optional[pd.DataFrame]`): Optional corrupted/augmented version of the synthetic data.
- **Output:**
  - `pd.DataFrame`: A single dataframe combining the available inputs. Will include:
    - `augmented_real_df` if provided, else `real_df`
    - `augmented_synth_df` if provided, else `synth_df` (if either exists)

### ⚠️ Design Notes / Gotchas
- Assumes all inputs have compatible schemas (same columns). If not, `pd.concat` will raise or introduce NaNs.
- Always returns a dataframe — never returns `None`.
- At least `real_df` must be provided (non-null); others are optional.
- Preserves order: real/augmented-real rows come before synth/augmented-synth rows.

### 🔗 Related Functions
- `augment_dataset`: Generates the optional augmented inputs.
- `generate_combined_synthetic_data`: Produces `synth_df` input.
- `mine_stats`: Downstream consumer of the combined dataframe.

### 🧪 Testing Status
- Fully unit tested with `pytest`.
- Covers combinations including: real only, real + synth, augmented only, mixed, edge cases, and schema mismatch.

## 🧠 Function: `data_prep_pipeline`

### 📄 What It Does
- Prepares a dataset for rule mining by optionally sampling, synthesizing, corrupting, and merging real and synthetic data into a final dataframe.
- Returns both the processed dataset and associated metadata/logs for transparency and auditing.

### 🚦 When to Use
- Use this function when you want to standardize and automate the preprocessing pipeline prior to applying rule mining algorithms.
- Supports flexible configuration via a config object, with override options for easy API or UI integration.
- Especially useful in workflows where you want to test robustness via synthetic data or introduce controlled noise to test signal stability.

### 🔢 Inputs and Outputs

**Inputs:**
- `df` (`pd.DataFrame`): The raw dataset to be preprocessed.
- `cfg` (`Any`): Config object containing default parameters (must support `getattr()` access).
- `logger` (`Optional[Any]`): Optional logger for structured logging, should implement `.log_step(...)`.
- `**overrides` (`dict`): Optional keyword overrides for config values.

**Outputs:**
- `mining_input_df` (`pd.DataFrame`): The final dataframe (real + synthetic + augmented as applicable) ready for mining.
- `logs` (`Dict[str, Any]`):
  - `"prep_log"` (`pd.DataFrame`): Details about how the dataframe was prepared.
  - `"synth_logs"` (`dict`): Metadata from synthetic data generation, empty if not run.

### ⚠️ Design Notes / Gotchas
- Uses helper `param(name)` to resolve values from overrides first, falling back to `cfg`.
- The `logger` is optional but recommended in production to track each stage of the pipeline.
- `coalesce_data(...)` prefers augmented data over raw when both exist.
- `cfg` must include all expected attributes or raise `AttributeError`.
- If neither synthetic data nor augmentation is enabled, this reduces to a clean and fast prep-only step.

### 🔗 Related Functions
- `prepare_dataframe_for_mining`
- `generate_combined_synthetic_data`
- `augment_dataset`
- `coalesce_data`

### 🧪 Testing Status
- Unit tested via `pytest` fixtures covering:
  - Basic usage with real data only
  - Empty data edge case
  - Overrides for config parameters
  - Logging integration
  - Missing attribute error handling

## 🧠 Function: `mining_pipeline`

### 📄 What It Does
- Executes one or more rule mining algorithms on a preprocessed dataset.
- Collects statistics and logs from each miner and returns the combined results.

### 🚦 When to Use
- Use when you have a cleaned and feature-engineered dataset and want to extract interpretable rules using a configurable set of mining algorithms.
- Ideal in research or production settings where multiple miners need to be run in parallel with unified logging.
- Avoid calling this on raw or unencoded data — upstream preparation is expected.

### 🔢 Inputs and Outputs

#### Inputs
- `df` (`pd.DataFrame`): Preprocessed dataframe with encoded features and target column.
- `cfg` (`Any`): Configuration object containing default parameters; must support attribute-style access.
- `logger` (`Optional[Any]`): Optional logger with a `.log_step()` method for structured pipeline logging.
- `**overrides` (`dict`): Optional keyword arguments to override any config attribute.

#### Outputs
- `final_stats_df` (`pd.DataFrame`): Dataframe of rule statistics across all selected miners.
- `{"mining_logs": List[pd.DataFrame]}`: Dictionary containing miner-specific log dataframes.

### ⚠️ Design Notes / Gotchas
- Expects all inputs to be validated and transformed prior to this step — no additional preprocessing or error handling is done.
- If an unknown miner is passed in `miners`, the underlying `mine_stats()` function will raise.
- Assumes that all miner functions return a consistent log format for aggregation.
- Logging is optional but structured; each step logs with relevant config and results.

### 🔗 Related Functions
- `mine_stats`: Core function that dispatches and merges results from individual miners.
- `data_prep_pipeline`: Use this prior to `mining_pipeline` to prepare the input dataframe.
- Individual miner functions: `mine_apriori`, `mine_rulefit`, `mine_cart`, etc.

### 🧪 Testing Status
- Unit tested with valid, empty, and invalid inputs.
- Includes coverage for logger-enabled runs, override behavior, and edge-case miners.
