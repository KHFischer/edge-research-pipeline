import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Dict, Union, Any
from collections import defaultdict
import logging
import re
from mlxtend.frequent_patterns import apriori, association_rules
from imodels import RuleFitClassifier
import pysubgroup as ps


def stratified_sample(
    df: pd.DataFrame,
    target_cols: Union[str, List[str]] = "return",
    sample_size: int = 5000,
    random_state: int = 42,
    logger: Optional = None
) -> pd.DataFrame:
    """
    Return a stratified sample of df with up to `sample_size` rows.
    
    If df has fewer rows, returns a copy unchanged.
    
    Supports either a single target column or multiple one-hot columns.
    
    Args:
        df: Input dataframe.
        target_cols: Target column name (str) or list of one-hot column names.
        sample_size: Desired number of samples.
        random_state: Random seed.
        logger: Optional logger.
    
    Returns:
        Sampled dataframe.
    """
    n_rows = len(df)
    if n_rows <= sample_size:
        if logger:
            logger.info(f"Dataset has {n_rows} rows, <= sample size {sample_size}. Returning full dataset.")
        return df.copy()

    # Determine stratify labels
    if isinstance(target_cols, str):
        if target_cols not in df.columns:
            raise ValueError(f"Target column '{target_cols}' not found in dataframe.")
        stratify_labels = df[target_cols]
    else:
        missing = set(target_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Target columns {missing} not found in dataframe.")
        stratify_labels = df[target_cols].idxmax(axis=1)

    # Sample
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    if logger:
        logger.info(f"Sampled {sample_size} rows (from {n_rows}) stratified by target.")

    return sampled_df.reset_index(drop=True)

def perform_apriori(
    df: pd.DataFrame,
    min_support: float,
    min_observations: int,
    lift_min: float,
    lift_max: float,
    target_col: str = 'return',
    logger: Optional = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Apriori mining and return both the full rules dataframe and a summary dataframe.
    
    Args:
        df: Input dataframe.
        min_support: Minimum support threshold.
        min_observations: Minimum observation count.
        lift_min: Lower bound for lift (inclusive).
        lift_max: Upper bound for lift (inclusive).
        target_col: Name of the target column.
        logger: Optional logger.
    
    Returns:
        Tuple of (rules_df, summary_df)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # One-hot encode target
    target_dummies = pd.get_dummies(df[target_col], prefix='target')
    df_encoded = pd.concat([df.drop(target_col, axis=1), target_dummies], axis=1)

    # Ensure boolean dtype
    df_encoded = df_encoded.astype(bool)

    if logger:
        logger.info(f"Running Apriori with min_support={min_support}")

    # Mine frequent itemsets
    itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # Mine rules
    rules = association_rules(itemsets, metric="lift", min_threshold=0)

    initial_count = len(rules)
    if logger:
        logger.info(f"Generated {initial_count} initial rules.")

    # Filter rules with consequents that are target labels
    rules = rules[rules['consequents'].apply(lambda x: any(str(i).startswith('target_') for i in x))]
    after_target_filter = len(rules)

    # Keep only single-label consequents
    rules = rules[rules['consequents'].apply(lambda x: len(x) == 1)]
    after_single_label_filter = len(rules)

    # Keep only multi-feature antecedents
    rules = rules[rules['antecedents'].apply(lambda x: len(x) > 1)]
    after_multi_antecedent_filter = len(rules)

    # Compute additional columns
    rules = rules.copy()
    rules['obs'] = rules['support'] * len(df)
    rules['depth'] = rules['antecedents'].apply(len)

    # Apply lift and observation filters
    lift_mask = (rules['lift'] >= lift_min) & (rules['lift'] <= lift_max)
    obs_mask = (rules['obs'] >= min_observations)
    rules['selected'] = lift_mask & obs_mask

    selected_count = rules['selected'].sum()

    if logger:
        logger.info(
            f"Rules after filtering: {after_target_filter} (target filter), "
            f"{after_single_label_filter} (single label), "
            f"{after_multi_antecedent_filter} (multi antecedent), "
            f"{selected_count} selected."
        )

    # Standardize consequent naming
    rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)).replace('target_', ''))

    # Sort for readability
    rules = rules.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)

    # Build summary dataframe
    summary_data = {
        "total_rules_generated": [initial_count],
        "rules_after_target_filter": [after_target_filter],
        "rules_after_single_label_filter": [after_single_label_filter],
        "rules_after_multi_antecedent_filter": [after_multi_antecedent_filter],
        "rules_selected": [selected_count],
        "lift_min_value": [rules["lift"].min()],
        "lift_max_value": [rules["lift"].max()],
        "lift_mean_value": [rules["lift"].mean()],
        "support_min": [rules["support"].min()],
        "support_max": [rules["support"].max()],
        "support_mean": [rules["support"].mean()],
        "obs_min": [rules["obs"].min()],
        "obs_max": [rules["obs"].max()],
        "obs_mean": [rules["obs"].mean()],
    }
    summary_df = pd.DataFrame(summary_data)

    return rules, summary_df

def parse_apriori_rules(
    apriori_df: pd.DataFrame,
    column_name: str = 'antecedents'
) -> List[Dict[str, int]]:
    """
    Convert a dataframe column of Apriori antecedents into a list of dictionaries
    mapping feature names to 1.
    
    Args:
        apriori_df: DataFrame containing a column with frozenset antecedents.
        column_name: Name of the column containing frozensets.
    
    Returns:
        List of dictionaries like [{'feature1': 1, 'feature2': 1}, ...]
    """
    if column_name not in apriori_df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    rule_dicts: List[Dict[str, int]] = []

    for idx, itemset in enumerate(apriori_df[column_name]):
        if not isinstance(itemset, frozenset):
            raise ValueError(
                f"Row {idx}: Expected frozenset in column '{column_name}', got {type(itemset)}"
            )
        rule_dict = {str(feature): 1 for feature in itemset}
        rule_dicts.append(rule_dict)

    return rule_dicts

def perform_rulefit(
    df: pd.DataFrame,
    target_col: str,
    logger: Optional = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform RuleFit mining for multi-feature combinations predicting a multiclass target.

    Since RuleFit does not natively support multiclass classification, this function
    loops over each one-hot encoded target class separately and extracts rules for each.

    Args:
        df: Input dataframe.
        target_col: Name of the target column.
        logger: Optional logger.

    Returns:
        Tuple:
            - DataFrame of all rules including target class (consequent) and rule depth.
            - Summary DataFrame with counts and descriptive statistics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    prefix = "target"
    target_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    target_classes = df[target_col].unique()
    target_cols = target_dummies.columns.tolist()

    if logger:
        logger.info(f"Running RuleFit separately for {len(target_cols)} target classes.")

    all_rules = []
    summary_records = []

    for col in target_cols:
        # Extract X and y
        X = df.drop(columns=[target_col])
        y = target_dummies[col]

        if X.isnull().any().any():
            raise ValueError(f"Feature matrix contains missing values. Impute before using RuleFit.")
        
        # Binary encoding
        X_bin = X.astype(bool).astype('uint8')

        # Fit RuleFit model
        model = RuleFitClassifier()
        model.fit(X_bin, y)

        # Extract rules
        rules_list = model.rules_
        rules_dicts = [r.__dict__ for r in rules_list]
        rules_df = pd.DataFrame(rules_dicts)

        # Clean consequent naming
        class_label = col.replace(f"{prefix}_", "")
        rules_df["consequents"] = class_label

        # Compute depth safely
        def get_depth(rule_str):
            if not rule_str or rule_str.strip() == "":
                return 0
            return len(rule_str.split(" and "))

        rules_df["depth"] = rules_df["rule"].apply(get_depth)

        # Only multivariate rules
        rules_df = rules_df[rules_df["depth"] > 1].reset_index(drop=True)

        # Append per-class summary
        summary_records.append({
            "target_class": class_label,
            "total_rules": len(rules_dicts),
            "multivariate_rules": len(rules_df),
            "support_min": rules_df["support"].min() if not rules_df.empty else None,
            "support_max": rules_df["support"].max() if not rules_df.empty else None,
            "support_mean": rules_df["support"].mean() if not rules_df.empty else None
        })

        all_rules.append(rules_df)

    # Combine all target classes
    all_rules_df = pd.concat(all_rules, ignore_index=True)

    # Sort
    all_rules_df = all_rules_df.sort_values(
        ["support"], ascending=False
    ).reset_index(drop=True)

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_records)

    if logger:
        logger.info(
            f"Generated {len(all_rules_df)} multivariate rules across {len(target_cols)} target classes."
        )

    return all_rules_df, summary_df

def parse_rule_string_to_dict(rule_str: str) -> dict:
    """
    Parse a RuleFit rule string into {column: 0/1} conditions.
    
    Supports rules like:
        'col <= 0.5' -> col=0
        'col > 0.5'  -> col=1
        Combined with 'and'.
    """
    parts = [p.strip() for p in rule_str.split('and')]
    rule_dict = {}

    for part in parts:
        # Check which operator is present
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
            rule_dict[col] = 0
        elif op == ">" and val == "0.5":
            rule_dict[col] = 1
        else:
            raise ValueError(f"Unhandled rule format: '{part}' (operator {op}, value {val})")

    return rule_dict

def parse_rulefit_rules(all_rules_df):
    parsed_rules = []
    for rule in all_rules_df['rule']:
        rule_dict = parse_rule_string_to_dict(rule)
        parsed_rules.append(rule_dict)
    return parsed_rules

def perform_subgroup_discovery(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 50,
    depth: int = 3,
    logger: Optional = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Subgroup Discovery using pysubgroup on a multiclass target.

    Args:
        df: Input dataframe.
        target_col: Name of the target column.
        top_n: Number of top rules per target class to retrieve.
        depth: Max number of conditions (ANDs) in a rule.
        logger: Optional logger.

    Returns:
        Tuple:
            - DataFrame of rules.
            - Summary DataFrame with counts and descriptive statistics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    prefix = "target"
    target_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    target_classes = df[target_col].unique()
    target_cols = target_dummies.columns.tolist()

    if logger:
        logger.info(f"Running SubgroupDiscovery for {len(target_cols)} target classes.")

    all_rules = []
    summary_records = []

    for col in target_cols:
        class_label = col.replace(f"{prefix}_", "")

        # Prepare dataset
        df_bin = pd.concat([df.drop(columns=[target_col]), target_dummies[col]], axis=1)
        df_bin = df_bin.astype(bool)

        # Define target
        target = ps.BinaryTarget(col, True)

        # Define search space
        boolean_cols = [c for c in df_bin.columns if df_bin[c].dtype == bool and c != col]
        search_space = [ps.EqualitySelector(c, True) for c in boolean_cols]

        # Define search task
        task = ps.SubgroupDiscoveryTask(
            df_bin,
            target,
            search_space,
            result_set_size=top_n,
            depth=depth,
            qf=ps.WRAccQF()
        )

        # Execute
        result = ps.BeamSearch(beam_width=top_n).execute(task)
        rules_df = result.to_dataframe()
        total_rules = len(rules_df)
        rules_df["consequents"] = class_label

        # Compute depth
        def get_depth(subgroup):
            subgroup_str = str(subgroup)
            if not subgroup_str or subgroup_str.strip() == "":
                return 0
            return len(subgroup_str.split(" AND "))
        
        rules_df["depth"] = rules_df["subgroup"].apply(get_depth)

        # Keep only multivariate rules
        rules_df = rules_df[rules_df["depth"] > 1].reset_index(drop=True)

        # Append per-class summary
        summary_records.append({
            "target_class": class_label,
            "total_rules": total_rules,
            "multivariate_rules": len(rules_df),
            "quality_min": rules_df["quality"].min() if not rules_df.empty else None,
            "quality_max": rules_df["quality"].max() if not rules_df.empty else None,
            "quality_mean": rules_df["quality"].mean() if not rules_df.empty else None,
        })

        all_rules.append(rules_df)

    # Combine all
    all_rules_df = pd.concat(all_rules, ignore_index=True)

    # Rename columns for consistency
    all_rules_df = all_rules_df.rename(columns={"subgroup": "rule"})

    # Sort by quality
    all_rules_df = all_rules_df.sort_values("quality", ascending=False).reset_index(drop=True)

    # Summary
    summary_df = pd.DataFrame(summary_records)

    if logger:
        logger.info(
            f"Generated {len(all_rules_df)} multivariate rules across {len(target_cols)} target classes."
        )

    return all_rules_df, summary_df

def parse_subgroup_rule_to_dict(
    rule_str,
    target_prefix: str = "target_"
) -> Dict[str, int]:
    """
    Parse a subgroup rule into {feature: 0 or 1}.
    Converts any object to string before parsing.
    """
    # Always convert to string FIRST
    rule_str = str(rule_str)

    if not rule_str or rule_str.strip() == "":
        return {}

    # Clean enclosing parentheses
    rule_str = rule_str.strip()
    if rule_str.startswith("(") and rule_str.endswith(")"):
        rule_str = rule_str[1:-1]

    # Split by 'AND'
    parts = [p.strip() for p in rule_str.split("AND")]
    rule_dict = {}

    for part in parts:
        m = re.match(r"(.+?)\s*==\s*(True|False)", part)
        if not m:
            raise ValueError(f"Cannot parse rule part: '{part}'")
        col, val = m.groups()

        col = col.strip().strip("()")

        if col.startswith(target_prefix):
            continue

        rule_dict[col] = 1 if val == "True" else 0

    return rule_dict

def parse_subgroup_rules(
    subgroup_rules_df: pd.DataFrame,
    target_prefix: str = "target_"
) -> List[Dict[str, int]]:
    """
    Parse the 'rule' column of a DataFrame of subgroup rules
    into a list of {feature: 0/1} dictionaries.

    Args:
        subgroup_rules_df: DataFrame containing a 'rule' column.
        target_prefix: Prefix to ignore target conditions.

    Returns:
        List of dictionaries representing each rule.
    """
    if "rule" not in subgroup_rules_df.columns:
        raise ValueError("Input DataFrame must have a 'rule' column.")

    parsed_rules = []
    for i, rule in enumerate(subgroup_rules_df["rule"]):
        try:
            rule_dict = parse_subgroup_rule_to_dict(rule, target_prefix)
            parsed_rules.append(rule_dict)
        except ValueError as e:
            raise ValueError(f"Error parsing rule in row {i}: {e}")

    return parsed_rules

def dict_to_sorted_tuple(d: dict) -> tuple:
    """
    Converts a dict to a sorted tuple of items.
    This is the canonical representation for deduplication.
    """
    return tuple(sorted(d.items()))

def combine_and_dedup_rules(
    algorithm_rule_pairs: List[Tuple[str, List[Dict[str, Any]]]],
    dict_to_sorted_tuple: callable
) -> Tuple[List[Tuple], pd.DataFrame]:
    """
    Combines rule dictionaries from multiple mining algorithms,
    deduplicates them, and logs which algorithm(s) produced each rule.

    Parameters
    ----------
    algorithm_rule_pairs : List of (algorithm_name, rule_list) pairs
        Each rule_list is a list of dicts representing rules.
    dict_to_sorted_tuple : callable
        Function to convert a rule dict into a sorted tuple representation.

    Returns
    -------
    unique_rule_tuples : List of unique rule tuples.
    log_df : pd.DataFrame
        DataFrame with columns ['rule_tuple', 'algorithms'].
    """
    if not isinstance(algorithm_rule_pairs, list):
        raise ValueError("algorithm_rule_pairs must be a list of (algorithm_name, rule_list) pairs")

    rule_sources = defaultdict(set)

    # Process each algorithm's rules
    for algo_name, rules in algorithm_rule_pairs:
        if not isinstance(rules, list):
            raise ValueError(f"Rules for algorithm '{algo_name}' must be a list")
        for d in rules:
            t = dict_to_sorted_tuple(d)
            rule_sources[t].add(algo_name)

    # Final unique tuples
    unique_rule_tuples = list(rule_sources.keys())

    # Build log DataFrame
    log_records = []
    for rule_tuple, sources in rule_sources.items():
        log_records.append({
            "rule_tuple": rule_tuple,
            "algorithms": sorted(list(sources))
        })

    log_df = pd.DataFrame(log_records).sort_values("algorithms").reset_index(drop=True)

    return unique_rule_tuples, log_df

def apply_rule_tuple(rule_tuple, df):
    """
    Evaluates a rule tuple (('feature1', 0), ...) on the dataframe and returns a dense boolean mask.
    """
    # Start with a dense Series of True values (not sparse)
    mask = pd.Series([True] * len(df), index=df.index)

    for feature, expected_val in rule_tuple:
        # Cast the feature column to dense before comparison to avoid SparseArray issues
        mask &= (df[feature].to_numpy() == expected_val)

    return mask

def generate_multivariate_feature_df(
    rule_tuples: List[Tuple[Tuple[str, int], ...]],
    df: pd.DataFrame,
    target_col: str = "return",
    include_target: bool = True,
    prefix: str = "rule",
    sparse: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies a list of multivariate rule tuples to a dataframe to produce one-hot encoded rule columns.

    Parameters
    ----------
    rule_tuples : List of rule tuples
        Each rule is a tuple of (feature, expected_value) pairs.
    df : pd.DataFrame
        The dataframe containing all features and the target.
    target_col : str, default "return"
        Name of the target column to optionally include in output.
    include_target : bool, default True
        Whether to include the target column in the output dataframe.
    prefix : str, default "rule"
        Prefix for rule column names.
    sparse : bool, default False
        Whether to store rule columns as pandas SparseArray to save memory.
    logger : logging.Logger or None
        Logger instance. If None, uses a default logger.

    Returns
    -------
    result_df : pd.DataFrame
        Dataframe containing one column per rule (boolean), optionally with the target.
    rule_map_df : pd.DataFrame
        Dataframe mapping each rule column to the corresponding rule tuple.
    log_df : pd.DataFrame
        Dataframe with per-rule processing logs (success/failure and any error message).
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting generate_multivariate_feature_df with %d rules", len(rule_tuples))

    # Input validation
    if not isinstance(rule_tuples, list):
        raise ValueError("rule_tuples must be a list of rule tuples")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if include_target and target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    new_features = {}
    rule_records = []
    log_records = []

    for i, rule in enumerate(rule_tuples):
        col_name = f"{prefix}_{i:04d}"
        log_entry = {
            "rule_index": i,
            "rule_column": col_name,
            "rule": str(rule),
            "status": "success",
            "error": None,
        }

        try:
            # Validate individual rule
            if not isinstance(rule, (tuple, list)):
                raise TypeError("Rule is not a tuple or list")

            # Check all features exist
            for feature, expected_val in rule:
                if feature not in df.columns:
                    raise KeyError(f"Feature '{feature}' not found in dataframe")

            # Evaluate mask
            mask = pd.Series(True, index=df.index)
            for feature, expected_val in rule:
                mask &= (df[feature].to_numpy() == expected_val)

            # Optionally store as sparse
            if sparse:
                new_features[col_name] = pd.arrays.SparseArray(mask.values)
            else:
                new_features[col_name] = mask

            rule_records.append({
                "rule_column": col_name,
                "rule": str(rule)
            })

        except Exception as e:
            logger.warning("Rule %d skipped due to error: %s", i, str(e))
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)

        log_records.append(log_entry)

    result_df = pd.DataFrame(new_features, index=df.index)

    if include_target:
        result_df[target_col] = df[target_col]

    rule_map_df = pd.DataFrame(rule_records)
    log_df = pd.DataFrame(log_records)

    logger.info("Generated multivariate feature dataframe: %d columns created, %d failures",
                log_df['status'].value_counts().get('success', 0),
                log_df['status'].value_counts().get('failed', 0))

    return result_df, log_df

def rule_tuple_to_readable_string(rule_tuple, sep=" AND "):
    """
    Converts a rule tuple into a human-readable logical expression.
    e.g. (('feature1', 1), ('feature2', 0)) → 'feature1 == 1 AND feature2 == 0'
    """
    return sep.join([f"{feature} == {val}" for feature, val in rule_tuple])
    
def map_antecedents_to_readable(
    stats_df: pd.DataFrame,
    rule_log_df: pd.DataFrame,
    sep: str = " AND "
) -> pd.DataFrame:
    """
    Maps the antecedents in stats_df to human-readable rule strings,
    retaining the '== 0/1' meaning.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing a column 'antecedents' with e.g., 'rule_0001 == 0'
    rule_log_df : pd.DataFrame
        DataFrame mapping rule_column to rule tuples
    sep : str, default " AND "
        Separator for human-readable rule string

    Returns
    -------
    mapped_df : pd.DataFrame
        stats_df with 'antecedents' replaced by human-readable strings
    """
    # Create mapping: rule_column -> rule (string repr)
    rule_map = dict(
        zip(rule_log_df["rule_column"], rule_log_df["rule"])
    )

    # Helper to parse string repr back to tuple
    def parse_rule_str(rule_str):
        return eval(rule_str)

    # Build mapping: rule_column -> human readable string
    readable_map = {}
    for rule_col, rule_str in rule_map.items():
        rule_tuple = parse_rule_str(rule_str)
        readable_str = rule_tuple_to_readable_string(rule_tuple, sep=sep)
        readable_map[rule_col] = readable_str

    # Extract rule column and 0/1
    def parse_antecedent(antecedent_str):
        """
        'rule_0331 == 0' -> ('rule_0331', 0)
        """
        parts = antecedent_str.split("==")
        rule_col = parts[0].strip()
        value = int(parts[1].strip())
        return rule_col, value

    # Create final readable antecedents with NOT if needed
    def build_readable_antecedent(antecedent_str):
        rule_col, value = parse_antecedent(antecedent_str)
        readable = readable_map.get(rule_col, f"[Unknown rule: {rule_col}]")
        if value == 1:
            return readable
        else:
            return f"NOT ({readable})"

    # Apply mapping
    mapped_antecedents = stats_df["antecedents"].apply(build_readable_antecedent)

    # Return new DataFrame with replaced antecedents
    mapped_df = stats_df.copy()
    mapped_df["antecedents"] = mapped_antecedents

    return mapped_df

def add_readable_rule_column(merged_df: pd.DataFrame, sep: str = " AND ") -> pd.DataFrame:
    """
    Adds a 'readable_rule' column to merged_df by converting rule_tuple_multivar
    into a human-readable logical expression.

    Parameters
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame containing 'rule_tuple_multivar' as parsed tuples.
    sep : str, default " AND "
        Separator between conditions.

    Returns
    -------
    pd.DataFrame
        A copy of merged_df with an additional 'readable_rule' column.
    """
    # Defensive copy to avoid modifying input
    df = merged_df.copy()

    # Apply conversion using your utility
    df["readable_rule"] = df["rule_tuple_norm"].apply(
        lambda tup: rule_tuple_to_readable_string(tup, sep=sep)
    )

    return df
    
def get_rule_depth(antecedent_str: str, sep: str = " AND ") -> int:
    """
    Counts the number of clauses in a readable antecedent string.
    Handles NOT(...) wrapping gracefully.
    """
    s = antecedent_str.strip()
    # Handle NOT wrapping
    if s.startswith("NOT (") and s.endswith(")"):
        # Remove 'NOT (' prefix and trailing ')'
        s = s[5:-1].strip()
    # Split by separator
    clauses = s.split(sep)
    return len(clauses)