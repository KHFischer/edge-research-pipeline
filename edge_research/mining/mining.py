import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Set, Dict
from mlxtend.frequent_patterns import apriori, association_rules
from imodels import RuleFitClassifier
import pysubgroup as ps
import re
from collections import Counter

# --- Prep ---
def prepare_dataframe_for_mining(
    df: pd.DataFrame,
    date_col: str,
    id_cols: List[str],
    drop_cols: List[str],
    target_col: str = "forward_return",
    to_sample: bool = True,
    sample_size: int = 100_000,
    drop_duplicates: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares a transactional dataframe for rule mining, including column pruning, memory optimization,
    optional deduplication, stratified sampling, and processing log generation.

    Args:
        df (pd.DataFrame): Input transactional dataframe.
        date_col (str): Name of the date column to drop.
        id_cols (List[str]): List of ID columns to drop.
        drop_cols (List[str]): List of additional columns to drop.
        target_col (str): Name of the target column.
        to_sample (bool): Whether to apply stratified sampling (default True).
        sample_size (int): Maximum number of rows after sampling (default 100_000).
        drop_duplicates (bool): Whether to drop exact duplicate rows (default False).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Processed dataframe ready for mining.
            - Single-row dataframe logging reduction and memory usage stats.

    Raises:
        ValueError: If target column is missing after column removal.
    """
    log = {}
    df_working = df.copy()

    log['initial_rows'] = len(df_working)
    log['initial_columns'] = df_working.shape[1]
    log['initial_ram_mb'] = df_working.memory_usage(deep=True).sum() / (1024 ** 2)

    # Drop non-feature columns
    non_feature_cols = id_cols + [date_col] + drop_cols
    df_working.drop(columns=non_feature_cols, errors='ignore', inplace=True)
    log['columns_dropped'] = log['initial_columns'] - df_working.shape[1]

    # Validate target column presence
    if target_col not in df_working.columns:
        raise ValueError(f"Target column '{target_col}' missing after dropping non-feature columns.")

    # Encode target column if binary
    target_unique = df_working[target_col].dropna().nunique()
    if pd.api.types.is_bool_dtype(df_working[target_col]) or target_unique == 2:
        df_working[target_col] = df_working[target_col].astype('uint8')

    # Encode all remaining features as uint8
    feature_cols = [c for c in df_working.columns if c != target_col]
    df_working[feature_cols] = df_working[feature_cols].astype('uint8')
    log['features_retained'] = len(feature_cols)

    # Optional deduplication
    if drop_duplicates:
        before_dedup = len(df_working)
        df_working.drop_duplicates(inplace=True)
        log['duplicates_dropped'] = before_dedup - len(df_working)
    else:
        log['duplicates_dropped'] = 0

    log['rows_after_drop_duplicates'] = len(df_working)

    # Optional stratified sampling
    if to_sample:
        log['sampling_applied'] = True
        if len(df_working) > sample_size:
            sample_frac = sample_size / len(df_working)
            df_working = (
                df_working
                .groupby(target_col, group_keys=False)
                .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
                .reset_index(drop=True)
            )
    else:
        log['sampling_applied'] = False

    log['rows_after_sampling'] = len(df_working)
    log['final_rows'] = len(df_working)
    log['final_ram_mb'] = df_working.memory_usage(deep=True).sum() / (1024 ** 2)

    log_df = pd.DataFrame([log])

    return df_working, log_df

def validate_parsed_rules(rules: List[List[Tuple[str, int]]]) -> None:
    """
    Validates that a list of parsed rules conforms to the expected format.

    Each parsed rule must be:
        - A list or tuple of (feature_name, expected_value) conditions.
        - Each feature_name must be a string.
        - Each expected_value must be an integer.

    This function raises ValueError immediately upon detecting any format inconsistency.

    Args:
        rules (List[List[Tuple[str, int]]]): 
            Parsed rules to validate.

    Raises:
        ValueError:
            If rules do not conform to the expected format.

    Example:
        >>> rules = [
        ...     [("feature1", 1), ("feature2", 1)],
        ...     [("feature3", 0)]
        ... ]
        >>> validate_parsed_rules(rules)  # passes silently
    """
    if not isinstance(rules, list):
        raise ValueError("Rules must be a list of parsed rules.")

    for rule_idx, rule in enumerate(rules):
        if not isinstance(rule, (list, tuple)):
            raise ValueError(f"Rule {rule_idx} must be a list or tuple of conditions.")
        for cond_idx, condition in enumerate(rule):
            if not (isinstance(condition, (list, tuple)) and len(condition) == 2):
                raise ValueError(
                    f"Condition {cond_idx} in rule {rule_idx} must be a (feature_name, expected_value) tuple."
                )
            feature, value = condition
            if not isinstance(feature, str):
                raise ValueError(
                    f"Feature name in condition {cond_idx} of rule {rule_idx} must be a string."
                )
            if not isinstance(value, int):
                raise ValueError(
                    f"Expected value in condition {cond_idx} of rule {rule_idx} must be an integer."
                )

# --- Apriori ---
def perform_apriori(
    df: pd.DataFrame,
    target_col: str = 'forward_return',
    min_support: float = 0.01,
    metric: str = "lift",
    min_threshold: float = 0.0,
    sort_rules: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mines multivariate association rules predicting the target column using Apriori.

    Args:
        df (pd.DataFrame): Input dataframe with binary features and a target column.
        target_col (str): Name of the target column.
        min_support (float): Minimum support threshold for itemset mining.
        metric (str): Metric to evaluate rule quality (passed to mlxtend).
        min_threshold (float): Minimum metric threshold.
        sort_rules (bool): Whether to sort output rules by [metric, confidence].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Rules dataframe after all filtering.
            - Single-row dataframe logging mining summary stats.

    Raises:
        ValueError: If target column is missing or multi-label consequents detected.
    """
    log = {}
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    target_dummies = pd.get_dummies(df[target_col], prefix='target')
    df_encoded = pd.concat([df.drop(columns=target_col), target_dummies], axis=1).astype(bool)

    log['initial_features'] = df_encoded.shape[1]
    log['target_levels'] = target_dummies.shape[1]
    log['min_support'] = min_support
    log['metric'] = metric
    log['min_threshold'] = min_threshold
    log['sort_applied'] = sort_rules

    itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    log['initial_itemsets'] = len(itemsets)

    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    log['initial_rules'] = len(rules)

    # Target filtering
    rules = rules[rules['consequents'].apply(
        lambda x: any(str(i).startswith('target_') for i in x)
    )]
    log['rules_after_target_filter'] = len(rules)

    # Single-label consequents
    rules = rules[rules['consequents'].apply(lambda x: len(x) == 1)]
    log['rules_after_single_consequent'] = len(rules)

    # Multivariate antecedents only
    rules = rules[rules['antecedents'].apply(lambda x: len(x) > 1)]
    log['rules_after_multivar_filter'] = len(rules)

    # Standardize consequents
    def extract_consequent(x):
        if len(x) != 1:
            raise ValueError(f"Unexpected multi-label consequent: {x}")
        return next(iter(x)).replace('target_', '')

    rules['consequents'] = rules['consequents'].apply(extract_consequent)

    # Optional sorting
    if sort_rules:
        rules = rules.sort_values([metric, 'confidence'], ascending=False).reset_index(drop=True)

    log_df = pd.DataFrame([log])

    return rules, log_df

def parse_apriori_rules(
    apriori_df: pd.DataFrame,
    column_name: str = 'antecedents'
) -> List[List[Tuple[str, int]]]:
    """
    Parses Apriori antecedents from a dataframe column into a standardized rule format.

    Each rule is represented as a list of (feature_name, expected_value) tuples,
    where expected_value is fixed at 1 for Apriori-generated antecedents.

    Args:
        apriori_df (pd.DataFrame):
            Dataframe containing mined Apriori rules.
        column_name (str):
            Name of the column containing antecedents as frozensets.

    Returns:
        List[List[Tuple[str, int]]]:
            List of parsed rules, where each rule is a list of (feature_name, 1) conditions.

    Raises:
        ValueError:
            If the column is missing or contains non-frozenset entries.

    Example:
        >>> df = pd.DataFrame({'antecedents': [frozenset({'featA', 'featB'})]})
        >>> parse_apriori_rules(df)
        [[('featA', 1), ('featB', 1)]]
    """
    if column_name not in apriori_df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    parsed_rules: List[List[Tuple[str, int]]] = []

    for idx, itemset in enumerate(apriori_df[column_name]):
        if not isinstance(itemset, frozenset):
            raise ValueError(
                f"Row {idx}: Expected frozenset in column '{column_name}', "
                f"got {type(itemset).__name__}."
            )
        rule = [(str(feature), 1) for feature in itemset]
        parsed_rules.append(rule)

    validate_parsed_rules(parsed_rules)
    return parsed_rules

# --- Rulefit ---
def perform_rulefit(
    df: pd.DataFrame,
    target_col: str = "forward_return",
    tree_size: int = 3,
    min_rule_depth: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs RuleFit mining to extract multivariate rule combinations predicting a multiclass target.

    This function trains separate binary RuleFit models for each one-hot encoded class of the target,
    extracts logical rules (excluding linear terms), and returns only multivariate rules.

    Args:
        df (pd.DataFrame): Input dataframe with binary features and a categorical or binary target column.
        target_col (str): Name of the target column.
        tree_size (int): Maximum depth of trees used for rule generation.
        min_rule_depth (int): Minimum depth (number of conditions) to consider a rule multivariate.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - all_rules_df: Combined dataframe of mined rules across all target classes.
            - summary_df: Per-target-class summary dataframe of rule counts and support statistics.

    Raises:
        ValueError: If target column missing or features contain NaNs.

    Notes:
        - Linear terms (single-feature coefficients) are excluded from the output.
        - Only rules with depth >= min_rule_depth are retained.
        - Feature columns must be strictly binary (0/1) before calling.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    prefix = "target"
    target_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    target_cols = target_dummies.columns.tolist()

    all_rules = []
    summary_records = []

    for col in target_cols:
        X = df.drop(columns=[target_col])
        y = target_dummies[col]

        if X.isnull().any().any():
            raise ValueError(f"Feature matrix contains missing values. Impute before using RuleFit.")

        X_bin = X.astype(bool).astype('uint8')

        model = RuleFitClassifier(tree_size=tree_size)
        model.fit(X_bin, y)

        # Extract only logical rules (exclude linear terms)
        rules_list = [r for r in model.rules_ if r.rule and r.rule.strip() != ""]
        total_extracted_rules = len(rules_list)

        # Parse to dataframe
        rules_dicts = [r.__dict__ for r in rules_list]
        rules_df = pd.DataFrame(rules_dicts)

        # Add target class label
        class_label = col.replace(f"{prefix}_", "")
        rules_df["consequents"] = class_label

        # Compute rule depth
        def get_depth(rule_str):
            if not rule_str or rule_str.strip() == "":
                return 0
            return len(rule_str.split(" and "))

        rules_df["depth"] = rules_df["rule"].apply(get_depth)

        # Keep only multivariate rules
        rules_df = rules_df[rules_df["depth"] >= min_rule_depth].reset_index(drop=True)

        # Log summary
        summary_records.append({
            "target_class": class_label,
            "total_extracted_rules": total_extracted_rules,
            "rules_retained_multivar": len(rules_df),
            "support_min": rules_df["support"].min() if not rules_df.empty else None,
            "support_max": rules_df["support"].max() if not rules_df.empty else None,
            "support_mean": rules_df["support"].mean() if not rules_df.empty else None
        })

        all_rules.append(rules_df)

    all_rules_df = pd.concat(all_rules, ignore_index=True)
    all_rules_df = all_rules_df.sort_values(["support"], ascending=False).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_records)

    return all_rules_df, summary_df

def parse_rule_string_to_tuples(rule_str: str) -> List[Tuple[str, int]]:
    """
    Parses a single RuleFit rule string into a list of (feature_name, expected_value) conditions.

    Supports:
        - 'feature <= 0.5' → ('feature', 0)
        - 'feature > 0.5'  → ('feature', 1)
        - Multi-condition rules split by 'and'.

    Args:
        rule_str (str):
            RuleFit rule as a string.

    Returns:
        List[Tuple[str, int]]:
            Parsed rule conditions as (feature_name, expected_value) tuples.

    Raises:
        ValueError:
            If the rule string contains unsupported formats or missing operators.
    """
    parts = [p.strip() for p in rule_str.split('and')]
    rule: List[Tuple[str, int]] = []

    for part in parts:
        if "<=" in part:
            op = "<="
            col, val = part.split("<=")
        elif ">" in part:
            op = ">"
            col, val = part.split(">")
        else:
            raise ValueError(f"Cannot parse rule part (no operator found): '{part}'")

        col = col.strip()
        val = val.strip()

        if op == "<=" and val == "0.5":
            rule.append((col, 0))
        elif op == ">" and val == "0.5":
            rule.append((col, 1))
        else:
            raise ValueError(f"Unhandled rule format: '{part}' (operator {op}, value {val})")

    return rule

def parse_rulefit_rules(
    rules_df: pd.DataFrame,
    column_name: str = 'rule'
) -> List[List[Tuple[str, int]]]:
    """
    Parses RuleFit rules from a dataframe into standardized parsed rule format.

    Each rule is represented as a list of (feature_name, expected_value) tuples.

    Args:
        rules_df (pd.DataFrame):
            DataFrame containing a column of RuleFit rule strings.
        column_name (str):
            Name of the column containing rule strings.

    Returns:
        List[List[Tuple[str, int]]]:
            Parsed rules in standardized format.

    Raises:
        ValueError:
            If column is missing or rule strings are malformed.

    Example:
        >>> df = pd.DataFrame({'rule': ['feature1 <= 0.5 and feature2 > 0.5']})
        >>> parse_rulefit_rules(df)
        [[('feature1', 0), ('feature2', 1)]]
    """
    if column_name not in rules_df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    parsed_rules: List[List[Tuple[str, int]]] = []

    for idx, rule_str in enumerate(rules_df[column_name]):
        if not isinstance(rule_str, str):
            raise ValueError(
                f"Row {idx}: Expected rule string in column '{column_name}', "
                f"got {type(rule_str).__name__}."
            )
        rule = parse_rule_string_to_tuples(rule_str)
        parsed_rules.append(rule)

    validate_parsed_rules(parsed_rules)

    return parsed_rules

# --- Subgroup Discovery ---
def perform_subgroup_discovery(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 50,
    depth: int = 3,
    beam_width: int = 50,
    qf: Optional[Any] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Subgroup Discovery using pysubgroup to identify multivariate rule combinations
    predicting each class of a multiclass target.

    Mines interpretable AND-based rules per class using Beam Search and a quality function.

    Args:
        df (pd.DataFrame):
            Input dataframe of binary features and a categorical target.
        target_col (str):
            Name of the target column.
        top_n (int):
            Maximum number of rules to retain per class.
        depth (int):
            Maximum number of conditions in any rule.
        beam_width (int):
            Beam search width (controls exploration breadth).
        qf (Optional[Any]):
            pysubgroup quality function (default: WRAccQF).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - all_rules_df: DataFrame of mined rules across all target classes.
            - summary_df: DataFrame of per-class mining summary statistics.

    Raises:
        ValueError:
            If target column missing from input dataframe.

    Notes:
        - Only multivariate rules (depth > 1) are returned.
        - Target conditions within rules are ignored.
        - Feature columns must be binary (converted to boolean internally).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    qf = qf or ps.WRAccQF()

    prefix = "target"
    target_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    target_cols = target_dummies.columns.tolist()

    all_rules = []
    summary_records = []

    for col in target_cols:
        class_label = col.replace(f"{prefix}_", "")
        df_bin = pd.concat([df.drop(columns=[target_col]), target_dummies[col]], axis=1).astype(bool)

        target = ps.BinaryTarget(col, True)
        feature_cols = [c for c in df_bin.columns if c != col and df_bin[c].dtype == bool]
        search_space = [ps.EqualitySelector(c, True) for c in feature_cols]

        task = ps.SubgroupDiscoveryTask(
            df_bin,
            target,
            search_space,
            result_set_size=top_n,
            depth=depth,
            qf=qf
        )

        result = ps.BeamSearch(beam_width=beam_width).execute(task)
        rules_df = result.to_dataframe()
        total_raw_rules = len(rules_df)

        # Process and clean rules
        rules_df = rules_df.rename(columns={"subgroup": "rule"})
        rules_df["rule"] = rules_df["rule"].astype(str)

        rules_df["depth"] = rules_df["rule"].apply(lambda s: len(s.split(" AND ")) if s.strip() else 0)
        rules_df = rules_df[rules_df["depth"] > 1].reset_index(drop=True)
        multivar_rules_count = len(rules_df)

        rules_df["consequents"] = class_label

        summary_records.append({
            "target_class": class_label,
            "empty_rule_set": total_raw_rules == 0,
            "total_raw_rules": total_raw_rules,
            "rules_retained_multivar": multivar_rules_count,
            "rules_filtered_out": total_raw_rules - multivar_rules_count,
            "avg_rule_depth": rules_df["depth"].mean() if not rules_df.empty else None,
            "quality_min": rules_df["quality"].min() if not rules_df.empty else None,
            "quality_max": rules_df["quality"].max() if not rules_df.empty else None,
            "quality_mean": rules_df["quality"].mean() if not rules_df.empty else None
        })

        all_rules.append(rules_df)

    all_rules_df = pd.concat(all_rules, ignore_index=True)
    all_rules_df = all_rules_df.sort_values("quality", ascending=False).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_records)

    return all_rules_df, summary_df

def parse_subgroup_rule_to_tuples(
    rule_str: str,
    target_prefix: str = "target_"
) -> List[Tuple[str, int]]:
    """
    Parses a single subgroup rule string into a list of (feature_name, expected_value) tuples.

    Supports:
        - 'feature == True'  → ('feature', 1)
        - 'feature == False' → ('feature', 0)
        - Multiple conditions joined with 'AND'.

    Args:
        rule_str (str):
            Subgroup rule string as returned by pysubgroup.
        target_prefix (str):
            Prefix used to identify and ignore target conditions (default: 'target_').

    Returns:
        List[Tuple[str, int]]:
            List of parsed conditions.

    Raises:
        ValueError:
            If rule part cannot be parsed or malformed.
    """
    rule_str = str(rule_str).strip()

    if not rule_str:
        return []

    if rule_str.startswith("(") and rule_str.endswith(")"):
        rule_str = rule_str[1:-1]

    parsed_rule: List[Tuple[str, int]] = []

    for part in [p.strip() for p in rule_str.split("AND")]:
        match = re.match(r"(.+?)\s*==\s*(True|False)", part)
        if not match:
            raise ValueError(f"Cannot parse rule part: '{part}'")

        feature, value_str = match.groups()
        feature = feature.strip().strip("()")

        if feature.startswith(target_prefix):
            continue  # Skip target condition

        parsed_rule.append((feature, 1 if value_str == "True" else 0))

    return parsed_rule

def parse_subgroup_rules(
    subgroup_rules_df: pd.DataFrame,
    column_name: str = "rule",
    target_prefix: str = "target_"
) -> List[List[Tuple[str, int]]]:
    """
    Parses subgroup rules from a dataframe column into standardized rule format.

    Args:
        subgroup_rules_df (pd.DataFrame):
            Dataframe containing a column of subgroup rule strings.
        column_name (str):
            Column name containing the rule strings (default: 'rule').
        target_prefix (str):
            Prefix used to ignore target conditions within the rules.

    Returns:
        List[List[Tuple[str, int]]]:
            List of parsed rules, where each rule is a list of (feature_name, expected_value) tuples.

    Raises:
        ValueError:
            If the specified column is missing or if parsing fails for any rule.
    """
    if column_name not in subgroup_rules_df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    parsed_rules: List[List[Tuple[str, int]]] = []

    for idx, rule_str in enumerate(subgroup_rules_df[column_name]):
        try:
            parsed_rule = parse_subgroup_rule_to_tuples(rule_str, target_prefix=target_prefix)
            parsed_rules.append(parsed_rule)
        except ValueError as e:
            raise ValueError(f"Error parsing rule at row {idx}: {e}")

    validate_parsed_rules(parsed_rules)

    return parsed_rules

# --- Rule normalize & Dedup ---
def normalize_rule(rule: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Returns a canonical, sorted version of a rule for consistent comparison and deduplication.

    Each rule is a list of (feature_name, expected_value) tuples. Sorting is performed
    first by feature name, then by expected value.

    Args:
        rule (List[Tuple[str, int]]): Rule as list of (feature, value) pairs.

    Returns:
        List[Tuple[str, int]]: Sorted rule in canonical form.
    """
    return sorted(rule, key=lambda x: (x[0], x[1]))


def deduplicate_rules_with_provenance(
    rule_sources: List[Tuple[str, List[List[Tuple[str, int]]]]]
) -> List[Tuple[List[Tuple[str, int]], Set[str]]]:
    """
    Deduplicates rules across multiple algorithms and tracks provenance for each unique rule.

    Args:
        rule_sources (List[Tuple[str, List[List[Tuple[str, int]]]]]):
            A list of (algorithm_name, rules) pairs. Each rule is a list of (feature, value) pairs.

    Returns:
        List[Tuple[List[Tuple[str, int]], Set[str]]]:
            List of (unique_rule, set_of_algorithms) pairs, where each unique_rule is represented
            as a sorted list of (feature, value) pairs.
    """
    rule_dict: Dict[Tuple[Tuple[str, int], ...], Set[str]] = {}

    for source_name, rules in rule_sources:
        for rule in rules:
            rule_key = tuple(rule)  # Must already be sorted for consistency.
            if rule_key not in rule_dict:
                rule_dict[rule_key] = set()
            rule_dict[rule_key].add(source_name)

    return [
        (list(rule_key), algorithms)
        for rule_key, algorithms in rule_dict.items()
    ]


def count_rules_per_algorithm(
    deduplicated_rules: List[Tuple[List[Tuple[str, int]], Set[str]]]
) -> pd.DataFrame:
    """
    Counts how many unique rules are attributed to each algorithm.

    Args:
        deduplicated_rules: List of (rule, set_of_algorithms) pairs.

    Returns:
        pd.DataFrame: Dataframe with 'algorithm' and 'unique_rule_count' columns.
    """
    algo_counter = Counter()

    for _, algorithms in deduplicated_rules:
        for algo in algorithms:
            algo_counter[algo] += 1

    df = pd.DataFrame.from_records(
        list(algo_counter.items()),
        columns=['algorithm', 'unique_rule_count']
    ).sort_values('algorithm').reset_index(drop=True)

    return df
    
def normalize_and_dedup_rules(
    rule_sources: List[Tuple[str, List[List[Tuple[str, int]]]]]
) -> List[Tuple[List[Tuple[str, int]], Set[str]]]:
    """
    Combines normalization and deduplication for a collection of rules from multiple algorithms.

    Args:
        rule_sources (List[Tuple[str, List[List[Tuple[str, int]]]]]):
            Input list of (algorithm_name, rules) pairs.

    Returns:
        List[Tuple[List[Tuple[str, int]], Set[str]]]:
            List of unique rules paired with set of source algorithms.
    """
    normalized_rule_sources = [
        (source_name, [normalize_rule(rule) for rule in rules])
        for source_name, rules in rule_sources
    ]
    
    deduplicated_rules = deduplicate_rules_with_provenance(normalized_rule_sources)
    rule_count_df = count_rules_per_algorithm(deduplicated_rules)
    
    return deduplicated_rules, rule_count_df

def generate_rule_activation_dataframe(
    df: pd.DataFrame,
    unique_rules: List[Tuple[List[Tuple[str, int]], Set[str]]],
    target_col: str,
    prefix: str = "rule"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts mined multivariate rules into a boolean feature dataframe where each column
    represents the activation (satisfaction) of a rule across all rows.

    Also generates a human-readable mapping from rule columns to their logical expressions.

    Args:
        df (pd.DataFrame):
            Input dataframe with binary features and target column.
        unique_rules (List[Tuple[List[Tuple[str, int]], Set[str]]]):
            Unique, normalized rules as a list of (rule_conditions, provenance) pairs.
            Each rule is a list of (feature_name, expected_value) tuples.
        target_col (str):
            Name of the target column to retain in the output dataframe.
        prefix (str):
            Prefix for generated rule columns (default: 'rule').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - rule_df (pd.DataFrame):
                One boolean column per rule, plus the original target column.
            - mapping_df (pd.DataFrame):
                Mapping from rule column names to human-readable rule descriptions.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    rule_columns: Dict[str, np.ndarray] = {}
    rule_descriptions: List[Dict[str, str]] = []

    for idx, (rule_conditions, _) in enumerate(unique_rules):
        mask = np.ones(len(df), dtype=bool)
        human_readable_parts = []

        for feature_name, expected_value in rule_conditions:
            if feature_name not in df.columns:
                raise KeyError(f"Feature '{feature_name}' not found in dataframe columns.")
            mask &= (df[feature_name].values == expected_value)
            human_readable_parts.append(f"('{feature_name}' == {expected_value})")

        rule_column_name = f"{prefix}_{idx:04d}"
        rule_columns[rule_column_name] = mask

        human_readable_rule = " AND ".join(human_readable_parts)
        rule_descriptions.append({
            "rule_column": rule_column_name,
            "human_readable_rule": human_readable_rule
        })

    rule_df = pd.DataFrame(rule_columns, index=df.index)
    rule_df[target_col] = df[target_col].values

    mapping_df = pd.DataFrame(rule_descriptions)

    return rule_df, mapping_df
