import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Set, Dict, Union
from mlxtend.frequent_patterns import apriori, association_rules
from imodels import RuleFitClassifier
import pysubgroup as ps
import re
from collections import Counter
from skeLCS import eLCS
from sklearn.tree import DecisionTreeClassifier
import os
import sys
import logging
import warnings
import contextlib
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

from numpy.random import default_rng
from badgers.generators.tabular_data.imbalance import RandomSamplingClassesGenerator

from scripts.statistics.calculator import generate_statistics

# Orange and Synthcity are incompatible together. Please see README.md for more information
try:
    import Orange
except ImportError:
    Orange = None

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
except ImportError:
    Plugins = None
    dataloader = None


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

# --- Premium Funcs ---
def merge_multivar_map_into_stats(
    multivar_stats: pd.DataFrame,
    multivar_map: pd.DataFrame,
    antecedents_col: str = "antecedents"
) -> pd.DataFrame:
    """
    Merges human-readable rule descriptions into a multivariate statistics dataframe.

    The function extracts rule column names from the 'antecedents' column (assumed format: 'rule_0000 == 1'),
    removes the ' == 1' suffix for consistency, and merges human-readable descriptions from a mapping dataframe.

    Args:
        multivar_stats (pd.DataFrame):
            DataFrame output from the statistics calculator, containing a rule column reference in
            the specified 'antecedents_col' (e.g. 'rule_0001 == 1').
        multivar_map (pd.DataFrame):
            DataFrame mapping 'rule_column' names to 'human_readable_rule' descriptions.
            Must contain columns:
                - 'rule_column'
                - 'human_readable_rule'
        antecedents_col (str):
            Name of the column in multivar_stats containing the rule column references (default: 'antecedents').

    Returns:
        pd.DataFrame:
            The multivar_stats dataframe augmented with:
                - 'rule_column': cleaned rule column name (without ' == 1' suffix).
                - 'human_readable_rule': human-readable string representation of the rule.
    """

    if antecedents_col not in multivar_stats.columns:
        raise ValueError(f"Column '{antecedents_col}' not found in multivar_stats dataframe.")

    if not {'rule_column', 'human_readable_rule'}.issubset(multivar_map.columns):
        raise ValueError("multivar_map must contain 'rule_column' and 'human_readable_rule' columns.")

    stats_df = multivar_stats.copy()
    stats_df['rule_column'] = stats_df[antecedents_col].str.replace(' == 1', '', regex=False)

    merged_df = stats_df.merge(
        multivar_map,
        how='left',
        on='rule_column'
    )

    return merged_df

def compute_rule_depth(rule_str: str) -> int:
    """
    Computes the depth of a rule based on its human-readable string.

    A rule's depth is defined as the number of conditions combined using 'AND'.
    For example:
        - A single-condition rule returns 1.
        - A two-condition rule returns 2, etc.

    Args:
        rule_str (str):
            Human-readable rule string (e.g. "('featureA' == 1) AND ('featureB' == 0)").

    Returns:
        int:
            Rule depth (number of base feature conditions).
            Returns 0 for empty or invalid rule strings.
    """
    if not isinstance(rule_str, str) or not rule_str.strip():
        return 0

    return rule_str.count(" AND ") + 1

def perform_elcs(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[List[List[Tuple[str, int]]], pd.DataFrame]:
    """
    Perform eLCS (Learning Classifier System) rule mining on a dataframe.

    Args:
        df (pd.DataFrame):
            Input dataframe containing binary or categorical features and a target column.
        target_col (str):
            Name of the target column for rule mining.

    Returns:
        Tuple:
            - all_rules (List[List[Tuple[str, int]]]):
                Flat list of discovered rules across all classes.
                Each rule is a list of (feature_name, expected_value) tuples.
            - log_df (pd.DataFrame):
                Summary dataframe with per-class statistics:
                    - 'target_class'
                    - 'n_rules'
                    - 'avg_depth'
                    - 'avg_fitness'
                    - 'avg_accuracy'

    Raises:
        ValueError:
            If the target column is missing from the dataframe.

    Notes:
        This function trains a one-vs-rest eLCS model for each class in the target,
        extracts all discovered rules, and returns both the rules and per-class summary stats.
        Phenotype prediction is ignored, as downstream processes evaluate consequents independently.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    prefix = "target"
    target_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    target_cols = target_dummies.columns.tolist()

    all_rules = []
    log_records = []

    feature_names = df.drop(columns=[target_col]).columns.tolist()

    for col in target_cols:
        X = df.drop(columns=[target_col])
        y = target_dummies[col]

        X_bin = X.astype(bool).astype('uint8').values.astype(float)
        y_bin = y.astype(bool).astype('uint8').values.astype(float)

        model = eLCS()
        model.fit(X_bin, y_bin)

        class_rules = []

        for classifier in model.population.popSet:
            condition_vector = classifier.condition
            specified_indices = classifier.specifiedAttList

            parsed_rule = [
                (feature_names[idx], int(val))
                for idx, val in zip(specified_indices, condition_vector)
            ]

            class_rules.append(parsed_rule)
            all_rules.append(parsed_rule)

        log_records.append({
            "target_class": col.replace(f"{prefix}_", ""),
            "n_rules": len(class_rules),
            "avg_depth": (
                sum(len(r) for r in class_rules) / len(class_rules)
                if class_rules else 0
            ),
            "avg_fitness": (
                sum(c.fitness for c in model.population.popSet) / len(model.population.popSet)
                if model.population.popSet else None
            ),
            "avg_accuracy": (
                sum(c.accuracy for c in model.population.popSet) / len(model.population.popSet)
                if model.population.popSet else None
            ),
        })

    log_df = pd.DataFrame(log_records)
    validate_parsed_rules(all_rules)
    
    return all_rules, log_df

def df_to_orange_table(df: pd.DataFrame, target_col: str = "forward_return"):
    """
    Convert a pandas DataFrame of binary indicator features and a categorical target
    into an Orange Table suitable for CN2 rule induction.

    Args:
        df (pd.DataFrame):
            Input dataframe where all feature columns must be binary (0/1 or bool).
        target_col (str, optional):
            Name of the target column to use as the class variable (default: "forward_return").

    Returns:
        Orange.data.Table:
            Orange Table with discrete feature variables and a discrete class variable.

    Raises:
        ImportError:
            If Orange is not installed.
        KeyError:
            If the specified target column is missing from the dataframe.
        ValueError:
            If any feature columns are not binary indicators.

    Notes:
        - Feature values are coerced to integer 0/1 before conversion.
        - Class variable values retain their original categorical labels.
        - All features are treated as discrete variables with values ["0", "1"].
    """
    try:
        import Orange
    except ImportError:
        raise ImportError(
            "❌ The `orange3` package is required for CN2 rule mining but is not installed.\n"
            "To enable this feature, install the optional dependency:\n\n"
            "    pip install edge-research-pipeline[orange]\n\n"
            "Alternatively, disable the CN2 miner in your config."
        )

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    feature_cols = [col for col in df.columns if col != target_col]

    # Validate all features are binary (0/1, True/False)
    invalid_features = [
        col for col in feature_cols
        if not set(df[col].unique()) <= {0, 1, True, False}
    ]
    if invalid_features:
        raise ValueError(f"Non-binary feature columns detected: {invalid_features}")

    # Prepare feature matrix (X)
    X = df[feature_cols].astype("int8").to_numpy()
    feature_vars = [
        Orange.data.DiscreteVariable.make(col, values=["0", "1"])
        for col in feature_cols
    ]

    # Prepare class variable (Y)
    y_series = df[target_col].astype(str)
    class_values = sorted(y_series.unique().tolist())
    class_var = Orange.data.DiscreteVariable.make(target_col, values=class_values)
    label_to_index = {label: idx for idx, label in enumerate(class_values)}
    Y = y_series.map(label_to_index).to_numpy().reshape(-1, 1)

    domain = Orange.data.Domain(feature_vars, class_var)
    table = Orange.data.Table.from_numpy(domain, X, Y)

    return table

def perform_cn2(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[List[List[Tuple[str, int]]], pd.DataFrame]:
    """
    Perform CN2 rule induction using Orange3 and extract rules in standardized format.

    Args:
        df (pd.DataFrame):
            Input dataframe containing feature columns and a discrete target column.
            Features must be pre-binarized or categorical.
        target_col (str):
            Name of the target column in the dataframe.

    Returns:
        Tuple:
            parsed_rules (List[List[Tuple[str, int]]]):
                Flat list of discovered rules, where each rule is a list of 
                (feature_name, expected_value) conditions.
            log_df (pd.DataFrame):
                Summary dataframe with:
                    - 'n_rules': Total number of rules extracted.
                    - 'avg_depth': Average number of antecedents per rule.

    Raises:
        ValueError:
            If the target column is missing from the dataframe.

    Notes:
        - The predicted class (consequent) of each rule is ignored.
        - Unsupported operators (non-binary conditions) are skipped.
        - Requires pre-binarized or discretized features.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    data_table = df_to_orange_table(df, target_col)
    learner = Orange.classification.rules.CN2Learner()
    classifier = learner(data_table)

    parsed_rules: List[List[Tuple[str, int]]] = []
    feature_names = [attr.name for attr in data_table.domain.attributes]

    for rule in classifier.rule_list:
        parsed_rule: List[Tuple[str, int]] = []
        for selector in rule.selectors:
            feature_name = feature_names[selector.column]

            # Support only basic equality / non-equality operators
            if selector.op in ("=", "==", "!="):
                try:
                    value = int(selector.value)
                except (ValueError, TypeError):
                    continue  # Skip non-integer or malformed values

                # Interpret '!=' as negation of equality
                if selector.op == "!=":
                    value = 0 if value != 0 else 1

                parsed_rule.append((feature_name, value))

        if parsed_rule:
            parsed_rules.append(parsed_rule)

    n_rules = len(parsed_rules)
    avg_depth = (sum(len(rule) for rule in parsed_rules) / n_rules) if n_rules else 0

    log_df = pd.DataFrame([{
        "n_rules": n_rules,
        "avg_depth": avg_depth
    }])

    validate_parsed_rules(parsed_rules)

    return parsed_rules, log_df

def perform_cart(
    df: pd.DataFrame,
    target_col: str,
    max_depth: Optional[int] = 5,
    criterion: str = "gini",
    random_state: Optional[int] = 42,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Tuple[List[List[Tuple[str, int]]], pd.DataFrame]:
    """
    Train a CART (Classification and Regression Tree) using scikit-learn and extract rules
    in standardized format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing binary or numeric features and a discrete target column.
    target_col : str
        Name of the target column.
    max_depth : int or None, optional
        Maximum depth of the tree (default is 5).
    criterion : str, optional
        Splitting criterion: 'gini' (default) or 'entropy'.
    random_state : int or None, optional
        Seed for reproducibility (default is 42).
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node (default is 2).
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node (default is 1).

    Returns
    -------
    parsed_rules : List[List[Tuple[str, int]]]
        Extracted rules as a list of (feature_name, expected_value) conditions.
    log_df : pd.DataFrame
        Summary dataframe containing:
            - 'n_rules': number of rules (root-to-leaf paths)
            - 'avg_depth': average number of conditions per rule
            - 'tree_depth': maximum depth of the tree

    Raises
    ------
    ValueError
        If target column is missing from the dataframe.

    Notes
    -----
    - Rules are extracted as root-to-leaf paths, ignoring class predictions.
    - Multiclass targets are natively supported.
    - Assumes features are pre-binarized or numeric.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    model.fit(X, y)

    tree = model.tree_
    parsed_rules: List[List[Tuple[str, int]]] = []

    def traverse(node: int, path: List[Tuple[str, int]]):
        if tree.feature[node] != -2:  # Not a leaf
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            if threshold < 0.5:
                condition = (feature, 0)
            else:
                condition = (feature, 1)

            traverse(tree.children_left[node], path + [condition])
            traverse(tree.children_right[node], path + [(feature, 1 - condition[1])])
        else:
            if path:
                parsed_rules.append(path)

    traverse(0, [])

    n_rules = len(parsed_rules)
    avg_depth = sum(len(rule) for rule in parsed_rules) / n_rules if n_rules else 0
    tree_depth = model.get_depth()

    log_df = pd.DataFrame([{
        "n_rules": n_rules,
        "avg_depth": avg_depth,
        "tree_depth": tree_depth
    }])

    validate_parsed_rules(parsed_rules)

    return parsed_rules, log_df

# -- Synthetic Data --
@contextlib.contextmanager
def force_silence():
    """Globally suppress stdout, stderr, warnings, and logging during critical operations."""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.disable(logging.CRITICAL)
                try:
                    yield
                finally:
                    logging.disable(logging.NOTSET)


@contextlib.contextmanager
def _suppress_sdv_logs(level: int = logging.ERROR):
    """
    Temporarily suppress SDV, Copulas, and RDT library loggers.

    Args:
        level (int): Logging level to apply (default: logging.ERROR).
    """
    noisy_loggers = ["sdv", "copulas", "rdt"]
    saved_levels = {
        name: (logger.level, logger.propagate)
        for name in noisy_loggers
        if (logger := logging.getLogger(name))
    }
    try:
        for name in noisy_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
        yield
    finally:
        for name, (lvl, prop) in saved_levels.items():
            logger = logging.getLogger(name)
            logger.setLevel(lvl)
            logger.propagate = prop


def generate_synthetic_data_sdv(
    df: pd.DataFrame,
    num_rows: int,
    model: str = "gaussian_copula",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic tabular data using SDV's synthesizers (GaussianCopula, CTGAN, or TVAE),
    while suppressing noisy library logs by default.

    Args:
        df (pd.DataFrame): Preprocessed dataframe (no missing values).
        num_rows (int): Number of synthetic rows to generate.
        model (str, optional): Synthesizer type. Options: 'gaussian_copula', 'ctgan', 'tvae'.
        verbose (bool, optional): If True, print quality score after generation. Defaults to False.

    Raises:
        ValueError: If dataframe contains missing values or unsupported model type.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - synthetic_data: Generated synthetic dataframe.
            - metadata_df: DataFrame summarizing column sdtypes and per-column quality scores.
    """

    if df.isnull().any().any():
        raise ValueError("Input dataframe contains missing values. Please impute or drop before generating synthetic data.")

    model_registry = {
        "gaussian_copula": GaussianCopulaSynthesizer,
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }

    if model not in model_registry:
        raise ValueError(f"Unsupported model: '{model}'. Choose from {list(model_registry)}.")

    # Detect metadata (suppress SDV internals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

    synthesizer_cls = model_registry[model]
    synthesizer = synthesizer_cls(metadata)

    # Fit and sample while suppressing library logs unless verbose
    with _suppress_sdv_logs(level=logging.CRITICAL if not verbose else logging.INFO):
        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_rows=num_rows)

    # Evaluate quality of synthetic data
    quality_score = evaluate_quality(real_data=df, synthetic_data=synthetic_data, metadata=metadata)

    if verbose:
        print(f"[SDV] Synthetic data quality score: {quality_score.get_score():.3f}")

    # Build metadata summary and merge per-column quality scores
    metadata_dict = metadata.to_dict()
    metadata_df = pd.DataFrame({
        "column_name": list(metadata_dict["columns"].keys()),
        "sdtype": [col_info["sdtype"] for col_info in metadata_dict["columns"].values()]
    })

    column_scores = quality_score.get_details("Column Shapes").rename(columns={"Column": "column_name"})
    metadata_df = metadata_df.merge(column_scores, on="column_name", how="left")

    return synthetic_data, metadata_df

def generate_synthetic_data_synthcity(
    df: pd.DataFrame,
    target_col: Optional[str],
    n_rows: int,
    model: str = "ctgan",
    n_iter: int = 1000,
    batch_size: int = 128,
    lr: float = 1e-4,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic tabular data using Synthcity and evaluate quality using SDV.

    Parameters
    ----------
    df : pd.DataFrame
        Real dataset. Must be pre-cleaned (no missing values) and column types defined.
    target_col : str or None
        Optional target column used for conditional generation.
    n_rows : int
        Number of synthetic rows to generate.
    model : str, default='ctgan'
        Synthcity plugin to use. Supported: 'ctgan', 'tvae', 'rtvae', 'adsgan', 'pategan'.
    n_iter : int, default=1000
        Number of training iterations for the Synthcity plugin.
    batch_size : int, default=128
        Mini-batch size during training.
    lr : float, default=1e-4
        Learning rate for model optimization.
    device : str, default='cpu'
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    synthetic_df : pd.DataFrame
        Synthetic dataset generated by the plugin, including target column if provided.
    metadata_df : pd.DataFrame
        Summary of column types and SDV-evaluated shape similarity scores.

    Raises
    ------
    ImportError
        If Synthcity is not installed or its submodules are unavailable.
    ValueError
        If the specified model is not supported by Synthcity.
    """
    try:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
    except ImportError:
        raise ImportError(
            "❌ The `synthcity` package is required for synthetic data generation but is not installed.\n"
            "To enable this feature, install the optional dependency:\n\n"
            "    pip install edge-research-pipeline[synth]\n\n"
            "Alternatively, disable synthetic data generation in your config or workflow."
        )

    if model not in Plugins().list():
        raise ValueError(f"Model '{model}' is not a supported Synthcity plugin.")

    # Build Synthcity dataloader
    dataloader = GenericDataLoader(df, target_column=target_col)

    # Initialize and train Synthcity generator
    plugin = Plugins().get(
        model,
        n_iter=n_iter,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    plugin.fit(dataloader)

    # Generate synthetic data and unwrap it from Synthcity dataloader
    synthetic_df = plugin.generate(count=n_rows).raw()

    # Create SDV metadata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

    # Evaluate synthetic data quality using SDV
    quality_report = evaluate_quality(
        real_data=df,
        synthetic_data=synthetic_df,
        metadata=metadata
    )

    # Build per-column metadata summary with SDV scores
    metadata_dict = metadata.to_dict()
    metadata_df = pd.DataFrame({
        "column_name": metadata_dict["columns"].keys(),
        "sdtype": [col["sdtype"] for col in metadata_dict["columns"].values()]
    })

    column_scores = quality_report.get_details("Column Shapes").rename(columns={"Column": "column_name"})
    metadata_df = metadata_df.merge(column_scores, on="column_name", how="left")

    return synthetic_df, metadata_df

def generate_skewed_proportions(y: pd.Series, power: float = 1.5) -> Dict[Union[str, int], float]:
    """
    Generate synthetic class proportions by inverting and sharpening the
    current class distribution.

    This method increases the presence of minority classes by assigning
    them disproportionately higher weights.

    Parameters
    ----------
    y : pd.Series
        Target column with categorical labels.
    power : float, default=1.5
        Skew exponent. Higher values create more imbalance.

    Returns
    -------
    dict
        Dictionary of class label -> new class probability.
    """
    class_dist = y.value_counts(normalize=True)
    inverted = 1.0 / (class_dist + 1e-8)  # prevent division by zero
    skewed = inverted ** power
    normalized = skewed / skewed.sum()
    return normalized.to_dict()


def apply_class_imbalance(
    df: pd.DataFrame,
    target_col: str,
    proportions: Optional[Dict[Union[str, int], float]] = None,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Apply class imbalance to a tabular dataset using Badgers.

    If no class proportions are provided, generates a default skewed distribution
    that increases the frequency of minority classes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and a target column.
    target_col : str
        Name of the column to treat as the supervised class label.
    proportions : dict, optional
        Mapping from class label to target proportion (must sum to 1.0).
        If None, a skewed distribution is auto-generated.
    random_state : int, optional
        Random seed for deterministic sampling.

    Returns
    -------
    pd.DataFrame
        Resampled dataframe matching the desired class imbalance.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    if proportions is None:
        proportions = generate_skewed_proportions(df[target_col])

    rng = default_rng(random_state)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    transformer = RandomSamplingClassesGenerator(random_generator=rng)
    X_balanced, y_balanced = transformer.generate(X=X, y=y, proportion_classes=proportions)

    return pd.concat([X_balanced, y_balanced.rename(target_col)], axis=1)

def flip_boolean_values(
    df: pd.DataFrame,
    columns: Optional[Union[List[str], None]] = None,
    flip_fraction: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Randomly flips boolean values in specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with boolean columns to perturb.
    columns : list of str or None, optional
        List of boolean column names to flip. If None, all bool-type columns are used.
    flip_fraction : float, default=0.1
        Fraction of rows to flip per column (0.0–1.0).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        New dataframe with boolean values flipped in selected columns.
    """

    if not 0 <= flip_fraction <= 1:
        raise ValueError("flip_fraction must be between 0.0 and 1.0")

    df = df.copy()
    rng = np.random.default_rng(seed)

    if columns is None:
        columns = df.select_dtypes(include=["bool"]).columns.tolist()

    for col in columns:
        if df[col].dtype != "bool":
            raise TypeError(f"Column '{col}' is not of type bool")
        mask = rng.random(len(df)) < flip_fraction
        df.loc[mask, col] = ~df.loc[mask, col]

    return df


def flip_labels(
    df: pd.DataFrame,
    target_col: str,
    flip_fraction: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Randomly flips a fraction of labels in a target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a categorical target column.
    target_col : str
        Name of the target column to corrupt.
    flip_fraction : float, default=0.1
        Fraction of labels to flip (0.0–1.0).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        New dataframe with corrupted labels.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    if not 0 <= flip_fraction <= 1:
        raise ValueError("flip_fraction must be between 0.0 and 1.0")

    df = df.copy()
    rng = np.random.default_rng(seed)

    unique_labels = df[target_col].unique()
    if len(unique_labels) < 2:
        raise ValueError("Label flipping requires at least 2 unique classes")

    mask = rng.random(len(df)) < flip_fraction
    flip_indices = df.index[mask]
    new_labels = rng.choice(unique_labels, size=len(flip_indices))
    df.loc[flip_indices, target_col] = new_labels

    return df

def generate_combined_synthetic_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    to_sdv: bool = True,
    to_synthcity: bool = True,
    sdv_model: str = "gaussian_copula",
    sdv_rows: int = 1000,
    sdv_verbose: bool = False,
    sc_model: str = "ctgan",
    sc_rows: int = 1000,
    sc_n_iter: int = 1000,
    sc_batch_size: int = 128,
    sc_lr: float = 1e-4,
    sc_device: str = "cpu",
    silence: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic tabular data using SDV and/or Synthcity, and return a combined dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input real dataset. Must be preprocessed and free of missing values.
    target_col : Optional[str]
        Target column to use for conditional generation in Synthcity. Ignored by SDV.
    to_sdv : bool, default=True
        If True, generate data using the selected SDV model.
    to_synthcity : bool, default=True
        If True, generate data using the selected Synthcity model.
    sdv_model : str, default='gaussian_copula'
        SDV synthesizer to use. One of: {'gaussian_copula', 'ctgan', 'tvae'}.
    sdv_rows : int, default=1000
        Number of synthetic samples to generate using SDV.
    sdv_verbose : bool, default=False
        Whether to print SDV quality score.
    sc_model : str, default='ctgan'
        Synthcity plugin to use. Examples: 'ctgan', 'tvae', etc.
    sc_rows : int, default=1000
        Number of synthetic samples to generate using Synthcity.
    sc_n_iter : int, default=1000
        Number of training iterations for the Synthcity plugin.
    sc_batch_size : int, default=128
        Mini-batch size during training.
    sc_lr : float, default=1e-4
        Learning rate for the Synthcity generator.
    sc_device : str, default='cpu'
        Device to use for Synthcity ('cpu' or 'cuda').
    silence : bool, default=True
        If True, suppress stdout, stderr, warnings, and logging during generation.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        - Combined synthetic dataframe.
        - Dictionary of metadata logs keyed by generator name ("sdv", "synthcity").

    Raises
    ------
    RuntimeError
        If both `to_sdv` and `to_synthcity` are False or generation fails.
    """

    synthetic_data = []
    logs: Dict[str, pd.DataFrame] = {}

    context = force_silence() if silence else contextlib.nullcontext()
    with context:
        if to_sdv:
            sdv_data, sdv_log = generate_synthetic_data_sdv(
                df=df,
                num_rows=sdv_rows,
                model=sdv_model,
                verbose=sdv_verbose,
            )
            synthetic_data.append(sdv_data)
            logs["sdv"] = sdv_log

        if to_synthcity:
            sc_data, sc_log = generate_synthetic_data_synthcity(
                df=df,
                target_col=target_col,
                n_rows=sc_rows,
                model=sc_model,
                n_iter=sc_n_iter,
                batch_size=sc_batch_size,
                lr=sc_lr,
                device=sc_device,
            )
            synthetic_data.append(sc_data)
            logs["synthcity"] = sc_log

    if not synthetic_data:
        raise RuntimeError("No synthetic data was generated. At least one of `to_sdv` or `to_synthcity` must be True.")

    combined_df = pd.concat(synthetic_data, ignore_index=True)
    return combined_df, logs

def augment_dataset(
    df: pd.DataFrame,
    target_col: str,
    to_aug_imbalance: bool = False,
    to_aug_flip_feats: bool = False,
    to_aug_flip_targets: bool = False,
    flip_feats_frac: float = 0.1,
    flip_targs_frac: float = 0.1,
    imbalance_proportions: Optional[Dict] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply optional data augmentation techniques to a dataset: class imbalance, feature corruption, and label flipping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, assumed to be preprocessed and free of missing values.
    target_col : str
        Name of the target column for supervised tasks.
    to_aug_imbalance : bool, default=False
        Whether to apply class imbalance resampling using predefined or auto-generated proportions.
    to_aug_flip_feats : bool, default=False
        Whether to randomly flip values in boolean feature columns.
    to_aug_flip_targets : bool, default=False
        Whether to randomly flip class labels in the target column.
    flip_feats_frac : float, default=0.1
        Fraction of boolean feature values to flip (if enabled). Must be between 0.0 and 1.0.
    flip_targs_frac : float, default=0.1
        Fraction of target labels to flip (if enabled). Must be between 0.0 and 1.0.
    imbalance_proportions : dict, optional
        Optional dictionary mapping class labels to desired sampling proportions.
        If not provided, a skewed distribution is generated automatically.
    random_state : int, default=42
        Random seed for all stochastic operations.

    Returns
    -------
    pd.DataFrame
        Augmented dataframe with all requested transformations applied in sequence.

    Raises
    ------
    ValueError
        If flip fractions are outside [0.0, 1.0] or if target column is missing.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if not (0.0 <= flip_feats_frac <= 1.0):
        raise ValueError("flip_feats_frac must be between 0.0 and 1.0")

    if not (0.0 <= flip_targs_frac <= 1.0):
        raise ValueError("flip_targs_frac must be between 0.0 and 1.0")

    augmented_df = df.copy()

    if to_aug_imbalance:
        augmented_df = apply_class_imbalance(
            df=augmented_df,
            target_col=target_col,
            proportions=imbalance_proportions,
            random_state=random_state
        )

    if to_aug_flip_feats:
        augmented_df = flip_boolean_values(
            df=augmented_df,
            columns=None,  # auto-selects all boolean columns
            flip_fraction=flip_feats_frac,
            seed=random_state
        )

    if to_aug_flip_targets:
        augmented_df = flip_labels(
            df=augmented_df,
            target_col=target_col,
            flip_fraction=flip_targs_frac,
            seed=random_state
        )

    return augmented_df

### --- Thin wrappers --- ###
def mine_apriori(df, target_col, apriori_min_support, apriori_metric, apriori_min_metric):
    """
    Thin wrapper that runs Apriori + parses the rules. Used by the miner dispatcher.
    """

    apriori_rules, apriori_log = perform_apriori(df, target_col, apriori_min_support, apriori_metric, apriori_min_metric)
    apriori_parsed = parse_apriori_rules(apriori_rules)
    return apriori_parsed, apriori_log

def mine_rulefit(df, target_col, rulefit_tree_size, rulefit_min_depth):
    """
    Thin wrapper that runs Rulefit + parses the rules. Used by the miner dispatcher.
    """

    rulefit_rules, rulefit_log = perform_rulefit(df, target_col, rulefit_tree_size, rulefit_min_depth)
    rulefit_parsed = parse_rulefit_rules(rulefit_rules)
    return rulefit_parsed, rulefit_log

def mine_subgroup(df, target_col, subgroup_top_n, subgroup_depth, subgroup_beam_width):
    """
    Thin wrapper that runs Subgroup Discovery + parses the rules. Used by the miner dispatcher.
    """

    subgroup_rules, subgroup_log = perform_subgroup_discovery(df, target_col, subgroup_top_n, subgroup_depth, subgroup_beam_width)
    subgroup_parsed = parse_subgroup_rules(subgroup_rules)
    return subgroup_parsed, subgroup_log

def mine_elcs(df, target_col):
    """
    Thin wrapper that runs ELCS Association Rule Mining + parses the rules. Used by the miner dispatcher.
    """

    elcs_parsed, elcs_log = perform_elcs(df, target_col)
    return elcs_parsed, elcs_log

def mine_cn2(df, target_col):
    """
    Thin wrapper that runs CN2 Rule Induction + parses the rules. Used by the miner dispatcher.
    """

    cn2_parsed, cn2_log = perform_cn2(df, target_col)
    return cn2_parsed, cn2_log

def mine_cart(df, target_col, cart_max_depth, cart_criterion, cart_random_state, cart_min_samples_split, cart_min_samples_leaf):
    """
    Thin wrapper that runs CART Association Rule Mining + parses the rules. Used by the miner dispatcher.
    """

    cart_parsed, cart_log = perform_cart(
        df, target_col, cart_max_depth, cart_criterion, cart_random_state, cart_min_samples_split, cart_min_samples_leaf
    )
    return cart_parsed, cart_log

def mine_univar(df, cfg):
    """
    Thin wrapper that runs calculates statistics for all univariate features.
    """

    univar_stats, univar_stats_log = generate_statistics(df, cfg)
    univar_stats['rule_depth'] = 1
    return univar_stats, univar_stats_log

def mine_multivar(df, cfg, rule_sources, target_col):
    """
    Thin wrapper that runs calculates statistics for all mined rules.
    """

    unique_rules, rules_df = normalize_and_dedup_rules(rule_sources)
    multivar_df, multivar_map = generate_rule_activation_dataframe(df, unique_rules, target_col)

    # Compute stats for multivar df
    multivar_stats, multivar_stats_log = generate_statistics(multivar_df, cfg)
    multivar_map['rule_depth'] = multivar_map['human_readable_rule'].apply(compute_rule_depth)
    multivar_stats = merge_multivar_map_into_stats(multivar_stats, multivar_map)
    multivar_stats = multivar_stats.drop(['antecedents', 'rule_column'], axis=1).rename(columns={"human_readable_rule": "antecedents"})
    return multivar_stats, multivar_stats_log, rules_df
### --- End of thin wrappers --- ###

def mine_stats(
    df: pd.DataFrame,
    target_col: str,
    miners: List[str],
    cfg,
    apriori_min_support: float = 0.01,
    apriori_metric: str = "lift",
    apriori_min_metric: float = 0.0,
    rulefit_tree_size: int = 3,
    rulefit_min_depth: int = 2,
    subgroup_top_n: int = 50,
    subgroup_depth: int = 3,
    subgroup_beam_width: int = 50,
    cart_max_depth: int = 5,
    cart_criterion: str = "gini",
    cart_random_state: int = 42,
    cart_min_samples_split: int = 2,
    cart_min_samples_leaf: int = 1
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Executes selected rule miners and returns combined statistics, logs, and rule provenance.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed input dataframe.
    target_col : str
        Name of the target column used by all miners.
    miners : List[str]
        List of miner names to run. Supported: 'univar', 'apriori', 'rulefit', 'subgroup', 'elcs', 'cn2', 'cart'.
    cfg : Any
        Configuration object to be passed to statistics calculator and univariate miner.
    apriori_min_support : float, default=0.01
        Minimum support for Apriori rule mining.
    apriori_metric : str, default="lift"
        Metric used by Apriori for rule evaluation.
    apriori_min_metric : float, default=0.0
        Minimum threshold for Apriori metric.
    rulefit_tree_size : int, default=3
        Tree depth for RuleFit.
    rulefit_min_depth : int, default=2
        Minimum rule depth to include from RuleFit.
    subgroup_top_n : int, default=50
        Number of top subgroups to return from Subgroup Discovery.
    subgroup_depth : int, default=3
        Maximum rule depth for Subgroup Discovery.
    subgroup_beam_width : int, default=50
        Beam width for Subgroup Discovery.
    cart_max_depth : int, default=5
        Maximum depth of CART decision tree.
    cart_criterion : str, default="gini"
        CART splitting criterion.
    cart_random_state : int, default=42
        Random seed for CART.
    cart_min_samples_split : int, default=2
        Minimum samples required to split a node in CART.
    cart_min_samples_leaf : int, default=1
        Minimum samples required at a leaf node in CART.

    Returns
    -------
    final_stats_df : pd.DataFrame
        Combined dataframe of rule statistics.
    logs : Dict[str, pd.DataFrame]
        Dictionary of miner logs keyed by miner name.
    rules_df : pd.DataFrame
        Provenance dataframe indicating rule origin per algorithm.
    """

    MINER_FUNCS = {
        "apriori": mine_apriori,
        "rulefit": mine_rulefit,
        "subgroup": mine_subgroup,
        "elcs": mine_elcs,
        "cn2": mine_cn2,
        "cart": mine_cart,
    }

    MINER_ARGS = {
        "apriori": [df, target_col, apriori_min_support, apriori_metric, apriori_min_metric],
        "rulefit": [df, target_col, rulefit_tree_size, rulefit_min_depth],
        "subgroup": [df, target_col, subgroup_top_n, subgroup_depth, subgroup_beam_width],
        "elcs": [df, target_col],
        "cn2": [df, target_col],
        "cart": [df, target_col, cart_max_depth, cart_criterion, cart_random_state, cart_min_samples_split, cart_min_samples_leaf],
    }

    unknown = set(miners) - set(MINER_FUNCS) - {"univar"}
    if unknown:
        raise ValueError(f"Unrecognized miners: {unknown}")

    stats: List[pd.DataFrame] = []
    rule_sources = []
    logs: Dict[str, pd.DataFrame] = {}
    rules_df = pd.DataFrame()

    for name in miners:
        if name == "univar":
            stats_df, log = mine_univar(df, cfg)
            stats.append(stats_df)
            logs[name] = log
        else:
            parsed_rules, log = MINER_FUNCS[name](*MINER_ARGS[name])
            rule_sources.append((name, parsed_rules))
            logs[name] = log

    if rule_sources:
        multivar_stats, multivar_log, rules_df = mine_multivar(df, cfg, rule_sources, target_col)
        stats.append(multivar_stats)
        logs["multivar"] = multivar_log

    if not stats:
        final_stats_df = pd.DataFrame()
    elif len(stats) == 1:
        final_stats_df = stats[0]
    else:
        final_stats_df = pd.concat(stats, ignore_index=True)

    # If univariate rules were found, add a row to rules_df for provenance tracking
    if "univar" in miners and not final_stats_df.empty:
        univar_count = final_stats_df[final_stats_df['rule_depth'] == 1]['antecedents'].nunique()
        if univar_count > 0:
            rules_df = pd.concat([
                rules_df,
                pd.DataFrame([{"algorithm": "univar", "unique_rule_count": univar_count}])
            ], ignore_index=True)

    return final_stats_df, logs, rules_df

def coalesce_data(
    real_df: pd.DataFrame,
    synth_df: Optional[pd.DataFrame],
    augmented_real_df: Optional[pd.DataFrame],
    augmented_synth_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine real, synthetic, and optionally augmented datasets into a single dataframe
    for downstream use, depending on which components are available.

    Parameters
    ----------
    real_df : pd.DataFrame
        Original preprocessed real dataset. Always required.
    synth_df : Optional[pd.DataFrame]
        Synthetic data generated by a model. May be None if not used.
    augmented_real_df : Optional[pd.DataFrame]
        Augmented version of real_df (e.g., with label or feature noise). May be None.
    augmented_synth_df : Optional[pd.DataFrame]
        Augmented version of synth_df. May be None.

    Returns
    -------
    pd.DataFrame
        A combined dataframe constructed from the available real, synthetic, and/or
        augmented datasets. If no synthetic or augmented data is provided, returns real_df.
    """

    combined = []

    # Prefer augmented real over raw real
    if augmented_real_df is not None:
        combined.append(augmented_real_df)
    else:
        combined.append(real_df)

    # Optionally include synthetic (augmented or not)
    if synth_df is not None:
        combined.append(augmented_synth_df if augmented_synth_df is not None else synth_df)

    return pd.concat(combined, ignore_index=True)

def data_prep_pipeline(
    df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full data preparation pipeline for mining, including optional synthetic data generation
    and data corruption. Combines all relevant inputs into a final dataset for downstream use.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset to be prepared.
    cfg : Any
        Configuration object providing default parameters.
    logger : Optional[Any], default=None
        Optional logger for structured step logging. Should implement a `.log_step()` method.
    overrides : dict
        Optional keyword arguments to override config values.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Prepared dataframe ready for rule mining.
        - Dictionary of logs and metadata including:
            * "prep_log": pd.DataFrame
            * "synth_logs": dict (if applicable)
    """

    def param(name: str):
        return overrides.get(name, getattr(cfg, name))

    id_cols = param("id_cols")
    target_col = param("target_col")
    log_max_rows = param("log_max_rows")

    prep_df_kwargs = {
        "date_col": param("date_col"),
        "id_cols": id_cols,
        "drop_cols": param("drop_cols"),
        "target_col": target_col,
        "to_sample": param("to_sample"),
        "sample_size": param("sample_size"),
        "drop_duplicates": param("drop_duplicates"),
    }

    synth_kwargs = {
        "target_col": target_col,
        "to_sdv": param("to_sdv"),
        "to_synthcity": param("to_synthcity"),
        "sdv_model": param("sdv_model"),
        "sdv_rows": param("sdv_rows"),
        "sdv_verbose": param("sdv_verbose"),
        "sc_model": param("sc_model"),
        "sc_rows": param("sc_rows"),
        "sc_n_iter": param("sc_n_iter"),
        "sc_batch_size": param("sc_batch_size"),
        "sc_lr": param("sc_lr"),
        "sc_device": param("sc_device"),
        "silence": param("synth_silence"),
    }

    augment_kwargs = {
        "target_col": target_col,
        "to_aug_imbalance": param("to_aug_imbalance"),
        "to_aug_flip_feats": param("to_aug_flip_feats"),
        "to_aug_flip_targets": param("to_aug_flip_targets"),
        "flip_feats_frac": param("flip_feats_frac"),
        "flip_targs_frac": param("flip_targs_frac"),
    }

    # Step 1: Prepare real data
    real_df, prep_log = prepare_dataframe_for_mining(df, **prep_df_kwargs)
    if logger:
        logger.log_step(
            step_name="Prepping dataframe for mining",
            info=prep_df_kwargs,
            df=prep_log,
            max_rows=log_max_rows,
        )

    # Step 2: Optionally generate synthetic data
    synth_df, synth_logs = None, {}
    if param("to_sdv") or param("to_synthcity"):
        synth_df, synth_logs = generate_combined_synthetic_data(df=real_df, **synth_kwargs)
        if logger:
            logger.log_step(
                step_name="Creating Synthetic Data",
                info=synth_kwargs,
                df=synth_logs,
                max_rows=log_max_rows,
            )

    # Step 3: Optionally augment real/synth/both
    augmented_real_df = None
    augmented_synth_df = None
    if param("corrupt_data"):
        target = param("corrupt_target")
        if target in {"real", "both"}:
            augmented_real_df = augment_dataset(df=real_df, **augment_kwargs)
        if target in {"synthetic", "both"} and synth_df is not None:
            augmented_synth_df = augment_dataset(df=synth_df, **augment_kwargs)
        if logger:
            logger.log_step(
                step_name="Corrupting Data",
                info=augment_kwargs,
                df=pd.DataFrame(),
                max_rows=log_max_rows,
            )

    # Step 4: Merge final data for mining
    mining_input_df = coalesce_data(
        real_df=real_df,
        synth_df=synth_df,
        augmented_real_df=augmented_real_df,
        augmented_synth_df=augmented_synth_df,
    )

    return mining_input_df, {
        "prep_log": prep_log,
        "synth_logs": synth_logs,
    }

def mining_pipeline(
    df: pd.DataFrame,
    cfg: Any,
    logger: Optional[Any] = None,
    **overrides: Any
) -> Tuple[pd.DataFrame, Dict[str, List[pd.DataFrame]]]:
    """
    Run rule mining algorithms on a prepared dataframe and log relevant metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, expected to be preprocessed and feature-engineered.
    cfg : Any
        Configuration object with default mining parameters. Must support attribute access.
    logger : Optional[Any], default=None
        Optional logger instance for structured pipeline step logging. Should support `.log_step(...)`.
    overrides : dict
        Keyword overrides for config values. Any mining param in `cfg` can be overridden.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[pd.DataFrame]]]
        - The combined statistics dataframe from rule mining.
        - A dictionary containing a list of miner log dataframes under the key "mining_logs".
    """

    def param(name: str) -> Any:
        return overrides.get(name, getattr(cfg, name))

    target_col = param("target_col")
    log_max_rows = param("log_max_rows")

    mine_kwargs = {
        "target_col": target_col,
        "miners": param("miners"),
        "apriori_min_support": param("apriori_min_support"),
        "apriori_metric": param("apriori_metric"),
        "apriori_min_metric": param("apriori_min_metric"),
        "rulefit_tree_size": param("rulefit_tree_size"),
        "rulefit_min_depth": param("rulefit_min_depth"),
        "subgroup_top_n": param("subgroup_top_n"),
        "subgroup_depth": param("subgroup_depth"),
        "subgroup_beam_width": param("subgroup_beam_width"),
        "cart_max_depth": param("cart_max_depth"),
        "cart_criterion": param("cart_criterion"),
        "cart_random_state": param("cart_random_state"),
        "cart_min_samples_split": param("cart_min_samples_split"),
        "cart_min_samples_leaf": param("cart_min_samples_leaf"),
    }

    # Execute mining
    final_stats_df, logs, rules_df = mine_stats(df=df, cfg=cfg, **mine_kwargs)

    # Optional logging
    if logger:
        logger.log_step(
            step_name="Mining Rules",
            info=mine_kwargs,
            df=list(logs.values()),
            max_rows=log_max_rows,
        )

    return final_stats_df, rules_df, {"mining_logs": list(logs.values())}
