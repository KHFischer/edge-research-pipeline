import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Any
import ast

from scripts.utils.central_logger import MarkdownLogger
from scripts.statistics.calculator import generate_statistics
from scripts.rules_mining.mining_parsing import (
    stratified_sample, 
    perform_apriori, 
    parse_apriori_rules, 
    perform_rulefit, 
    parse_rulefit_rules, 
    perform_subgroup_discovery, 
    parse_subgroup_rules, 
    dict_to_sorted_tuple,
    combine_and_dedup_rules, 
    generate_multivariate_feature_df,
    map_antecedents_to_readable,
    add_readable_rule_column
)

def run_univar(
    sampled_df: pd.DataFrame, 
    min_support: float, 
    min_observations: int, 
    lift_upper_bound: float, 
    lift_lower_bound: float,
    logger: MarkdownLogger,
    min_antecedent_support: float = -1,
    min_consequent_support: float = -1,
    min_confidence: float = -1,
    min_representativity: float = -1,
    min_leverage: float = -1,
    min_conviction: float = -1,
    min_zhangs_metric: float = -1,
    min_jaccard: float = -1,
    min_certainty: float = -1,
    min_kulczynski: float = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
    """
    Runs univariate analysis on all features against the target.

    Returns:
        - univar_stats: DataFrame with univariate statistics and selection flag
        - univar_log: DataFrame with detailed logs
        - raw_rules: None (univariate does not produce rules)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    univar_stats, univar_log = generate_statistics(
        sampled_df,
        min_support,
        min_observations,
        lift_upper_bound,
        lift_lower_bound,
        min_antecedent_support,
        min_consequent_support,
        min_confidence,
        min_representativity,
        min_leverage,
        min_conviction,
        min_zhangs_metric,
        min_jaccard,
        min_certainty,
        min_kulczynski,
    )
    univar_stats["depth"] = 1

    logger.log_step(
        step_name="Univariate All Features log",
        info={
            "min_support": min_support,
            "min_observations": min_observations,
            "lift_upper_bound": lift_upper_bound,
            "lift_lower_bound": lift_lower_bound,
            "min_antecedent_support": min_antecedent_support,
            "min_consequent_support": min_consequent_support,
            "min_confidence": min_confidence,
            "min_representativity": min_representativity,
            "min_leverage": min_leverage,
            "min_conviction": min_conviction,
            "min_zhangs_metric": min_zhangs_metric,
            "min_jaccard": min_jaccard,
            "min_certainty": min_certainty,
            "min_kulczynski": min_kulczynski,
        },
        df=univar_log
    )

    return univar_stats, univar_log, None

def run_apriori(
    df: pd.DataFrame,
    target_col: str,
    min_support: float,
    min_observations: int,
    lift_lower_bound: float,
    lift_upper_bound: float,
    logger: MarkdownLogger,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Runs the Apriori algorithm on the input dataframe and parses the discovered rules.

    Returns
    -------
    apriori_stats : pd.DataFrame
        DataFrame with Apriori statistics and selection flags.
    apriori_log : pd.DataFrame
        DataFrame with detailed Apriori logs.
    apriori_rules : list
        List of parsed rule tuples (for further processing).
    """
    # Run Apriori algorithm
    apriori_stats, apriori_log = perform_apriori(
        df,
        min_support,
        min_observations,
        lift_lower_bound,
        lift_upper_bound,
        target_col
    )

    # Parse rules into consistent format
    apriori_rules = parse_apriori_rules(apriori_stats)

    # Add depth column
    apriori_stats["depth"] = apriori_stats["antecedents"].apply(
        lambda x: len(x) if isinstance(x, (frozenset, set, list, tuple)) else 1
    )

    # Log results
    logger.log_step(
        step_name="Apriori Results log",
        info={
            "target_col": target_col,
            "min_support": min_support,
            "min_observations": min_observations,
            "lift_upper_bound": lift_upper_bound,
            "lift_lower_bound": lift_lower_bound,
        },
        df=apriori_log
    )

    return apriori_stats, apriori_log, apriori_rules

def run_rulefit(
    df: pd.DataFrame,
    target_col: str,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Runs the RuleFit algorithm on the input dataframe and parses discovered rules.

    Returns
    -------
    rulefit_stats : pd.DataFrame
        DataFrame with RuleFit statistics and selection flags.
    rulefit_log : pd.DataFrame
        DataFrame with detailed RuleFit logs.
    rulefit_rules : list
        List of parsed rule tuples (for further processing).
    """
    # Run RuleFit
    rulefit_stats, rulefit_log = perform_rulefit(
        df,
        target_col
    )

    # Parse rules into consistent format
    rulefit_rules = parse_rulefit_rules(rulefit_stats)

    # Add depth column
    rulefit_stats["depth"] = rulefit_stats["rule"].apply(
        lambda x: x.lower().count(" and ") + 1 if isinstance(x, str) else 1
    )

    # Log
    logger.log_step(
        step_name="Rulefit Results log",
        info={
            "target_col": target_col
        },
        df=rulefit_log
    )

    return rulefit_stats, rulefit_log, rulefit_rules

def run_subgroup(
    df: pd.DataFrame,
    target_col: str,
    top_n: int,
    depth: int,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Runs Subgroup Discovery on the input dataframe and parses discovered rules.

    Returns
    -------
    subgroup_stats : pd.DataFrame
        DataFrame with Subgroup statistics and selection flags.
    subgroup_log : pd.DataFrame
        DataFrame with detailed Subgroup logs.
    subgroup_rules : list
        List of parsed rule tuples (for further processing).
    """
    # Run Subgroup Discovery
    subgroup_stats, subgroup_log = perform_subgroup_discovery(
        df,
        target_col,
        top_n,
        depth
    )

    # Parse rules into consistent format
    subgroup_rules = parse_subgroup_rules(subgroup_stats)

    # Add depth column
    subgroup_stats["depth"] = subgroup_stats["rule"].apply(
        lambda x: x.lower().count(" and ") + 1 if isinstance(x, str) else 1
    )

    # Log
    logger.log_step(
        step_name="Subgroup Discovery Results log",
        info={
            "target_col": target_col,
            "top_n": top_n,
            "depth": depth
        },
        df=subgroup_log
    )

    return subgroup_stats, subgroup_log, subgroup_rules

def run_multivariate_testing(
    unique_tuples: list,
    df: pd.DataFrame,
    target_col: str,
    min_support: float,
    min_observations: int,
    lift_upper_bound: float,
    lift_lower_bound: float,
    logger: MarkdownLogger,
    min_antecedent_support: float = -1,
    min_consequent_support: float = -1,
    min_confidence: float = -1,
    min_representativity: float = -1,
    min_leverage: float = -1,
    min_conviction: float = -1,
    min_zhangs_metric: float = -1,
    min_jaccard: float = -1,
    min_certainty: float = -1,
    min_kulczynski: float = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates multivariate rule features, calculates statistics, and maps to readable names.

    Returns
    -------
    readable_stats_df : pd.DataFrame
        DataFrame of multivariate statistics with readable antecedents.
    multivar_log : pd.DataFrame
        Log DataFrame from statistics calculation.
    multivar_rule_log : pd.DataFrame
        Log DataFrame from rule application (one-hot encoding).
    """
    # Create one-hot encoded rule columns
    multivar_rule_df, multivar_rule_log = generate_multivariate_feature_df(
        unique_tuples,
        df,
        target_col=target_col
    )

    logger.log_step(
        step_name="Creating one hot encoded multivar df",
        info={},
        df=multivar_rule_log
    )

    # Calculate statistics
    multivar_stats, multivar_log = generate_statistics(
        multivar_rule_df,
        min_support,
        min_observations,
        lift_upper_bound,
        lift_lower_bound,
        min_antecedent_support,
        min_consequent_support,
        min_confidence,
        min_representativity,
        min_leverage,
        min_conviction,
        min_zhangs_metric,
        min_jaccard,
        min_certainty,
        min_kulczynski,
    )

    # Map rule IDs to readable names
    readable_stats_df = map_antecedents_to_readable(multivar_stats, multivar_rule_log)

    # Compute depth from readable antecedents
    readable_stats_df["depth"] = readable_stats_df["antecedents"].apply(
        lambda x: x.lower().count(" and ") + 1 if isinstance(x, str) else 1
    )

    logger.log_step(
        step_name="Multivariate All Features log",
        info={
            "min_support": min_support,
            "min_observations": min_observations,
            "lift_upper_bound": lift_upper_bound,
            "lift_lower_bound": lift_lower_bound,
            "min_antecedent_support": min_antecedent_support,
            "min_consequent_support": min_consequent_support,
            "min_confidence": min_confidence,
            "min_representativity": min_representativity,
            "min_leverage": min_leverage,
            "min_conviction": min_conviction,
            "min_zhangs_metric": min_zhangs_metric,
            "min_jaccard": min_jaccard,
            "min_certainty": min_certainty,
            "min_kulczynski": min_kulczynski,
        },
        df=multivar_log
    )

    return readable_stats_df, multivar_log, multivar_rule_log

def create_summary_df(combined_results: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary DataFrame with counts of selected/unselected rules by depth.

    Parameters
    ----------
    combined_results : pd.DataFrame
        The combined results DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with counts grouped by 'selected' and 'depth'.
    """
    if combined_results.empty:
        return pd.DataFrame(columns=["selected", "depth", "n_rules"])
    
    summary_df = (
        combined_results
        .groupby(["selected", "depth"])
        .size()
        .reset_index(name="n_rules")
        .sort_values(["selected", "depth"])
        .reset_index(drop=True)
    )
    return summary_df

def normalize_and_merge_rules(rules_log_df: pd.DataFrame, multivar_rule_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and merge rules_log_df and multivar_rule_log_df on rule_tuple.

    Args:
        rules_log_df (pd.DataFrame): DataFrame with rule_tuple (tuple) and algorithms.
        multivar_rule_log_df (pd.DataFrame): DataFrame with rule_tuple (string) and metadata.

    Returns:
        pd.DataFrame: Merged DataFrame on normalized rule_tuple.
    """
    # Ensure a copy to avoid mutating input
    rules_df = rules_log_df.copy()
    multi_df = multivar_rule_log_df.copy()

    # Normalize multivar_rule_log_df 'rule_tuple' from string to tuple
    multi_df["rule_tuple_norm"] = multi_df["rule_tuple"].apply(ast.literal_eval)

    # For safety: also normalize rules_log_df to ensure no string types
    rules_df["rule_tuple_norm"] = rules_df["rule_tuple"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Merge on the normalized column
    merged_df = pd.merge(
        multi_df,
        rules_df,
        how="left",
        left_on="rule_tuple_norm",
        right_on="rule_tuple_norm",
        suffixes=("_multivar", "_rules")
    )

    return merged_df

def run_edge_discovery_pipeline(
    prepared_df: pd.DataFrame,
    cfg: Any,
    logger: MarkdownLogger,
    target_col: str = 'return',
    mine_rules: bool = True
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Runs the full edge discovery pipeline with univariate and multivariate testing.

    Returns
    -------
    results : Dict[str, Optional[pd.DataFrame]]
        Dictionary containing:
            - 'combined_results': All discovered rules (univariate + multivariate)
            - 'univariate_results': Univariate results or None
            - 'multivariate_results': Multivariate results or None
            - 'rules_log': DataFrame of deduplicated rules or None
            - 'combined_summary': Summary DataFrame of combined results
    """
    # Setup
    min_support = cfg.min_support
    min_observations = cfg.min_observations
    lift_upper_bound = cfg.lift_upper_bound
    lift_lower_bound = cfg.lift_lower_bound
    min_antecedent_support = cfg.min_antecedent_support
    min_consequent_support = cfg.min_consequent_support
    min_confidence = cfg.min_confidence
    min_representativity = cfg.min_representativity
    min_leverage = cfg.min_leverage
    min_conviction = cfg.min_conviction
    min_zhangs_metric = cfg.min_zhangs_metric
    min_jaccard = cfg.min_jaccard
    min_certainty = cfg.min_certainty
    min_kulczynski = cfg.min_kulczynski
    top_n = cfg.top_n
    depth = cfg.depth

    # Create dataframe with only features and target
    xy_df = prepared_df.drop(cfg.id_cols + [cfg.date_col], axis=1)
    rule_sources = []

    # Sampling
    if cfg.sample_data:
        sampled_df = stratified_sample(
            xy_df,
            sample_size=cfg.sample_size
        )
    else:
        sampled_df = prepared_df.copy()

    # Univariate testing
    if cfg.run_univariate:
        univar_stats, univar_log, _ = run_univar(
            sampled_df,
            min_support,
            min_observations,
            lift_upper_bound,
            lift_lower_bound,
            logger,
            min_antecedent_support,
            min_consequent_support,
            min_confidence,
            min_representativity,
            min_leverage,
            min_conviction,
            min_zhangs_metric,
            min_jaccard,
            min_certainty,
            min_kulczynski
        )
        univar_passed = univar_stats[univar_stats["selected"]]
        passed_feats = list(
            univar_passed["antecedents"].apply(lambda x: x.split(" ==")[0]).unique()
        )
        test_df = xy_df[passed_feats + [target_col]]
    else:
        univar_stats = None
        test_df = xy_df.copy()

    # Multivar testing needs at least 2 features + target column
    features = [col for col in test_df.columns if col != target_col]

    if (len(features) > 2) & (mine_rules):
        if cfg.run_apriori:
            apriori_stats, apriori_log, apriori_rules = run_apriori(
                test_df,
                target_col,
                min_support,
                min_observations,
                lift_lower_bound,
                lift_upper_bound,
                logger
            )
            rule_sources.append(("apriori", apriori_rules))
            
        if cfg.run_rulefit:
            rulefit_stats, rulefit_log, rulefit_rules = run_rulefit(
                test_df,
                target_col,
                logger
            )
            rule_sources.append(("rulefit", rulefit_rules))

        if cfg.run_subgroup:
            subgroup_stats, subgroup_log, subgroup_rules = run_subgroup(
                test_df,
                target_col,
                top_n,
                depth,
                logger
            )
            rule_sources.append(("subgroup", subgroup_rules))

        if rule_sources:
            unique_tuples, rules_log_df = combine_and_dedup_rules(
                rule_sources,
                dict_to_sorted_tuple=dict_to_sorted_tuple
            )
            logger.log_step(
                step_name="All Rules before Dedup log",
                info={},
                df=rules_log_df
            )
        else:
            unique_tuples, rules_log_df = [], pd.DataFrame()
    else:
        unique_tuples, rules_log_df = [], None
        logger.log_step(
            step_name="Multivariate Skipped",
            info={"reason": "Not enough features after univariate filtering or disabled"},
            df=pd.DataFrame()
        )

    # Multivariate testing
    if unique_tuples:
        multivar_stats_df, multivar_log_df, multivar_rule_log_df = run_multivariate_testing(
            unique_tuples,
            test_df,
            target_col,
            min_support,
            min_observations,
            lift_upper_bound,
            lift_lower_bound,
            logger,
            min_antecedent_support,
            min_consequent_support,
            min_confidence,
            min_representativity,
            min_leverage,
            min_conviction,
            min_zhangs_metric,
            min_jaccard,
            min_certainty,
            min_kulczynski,
        )
    else:
        multivar_stats_df = None

    # Merge rule logs for full log
    multivar_rule_log_df = multivar_rule_log_df.rename(columns={'rule': 'rule_tuple'})
    rule_df = normalize_and_merge_rules(rules_log_df, multivar_rule_log_df)
    rule_df = add_readable_rule_column(rule_df)

    # Expected columns for consistency
    expected_cols = [
        "antecedents", "consequents", "antecedent support",
        "consequent support", "support", "confidence", "lift",
        "representativity", "leverage", "conviction", "zhangs_metric",
        "jaccard", "certainty", "kulczynski", "obs", "selected", "depth"
    ]

    # Collect DataFrames to combine
    dfs_to_combine = []
    if univar_stats is not None:
        dfs_to_combine.append(univar_stats[expected_cols])
    if multivar_stats_df is not None:
        dfs_to_combine.append(multivar_stats_df[expected_cols])

    if dfs_to_combine:
        combined_results = pd.concat(dfs_to_combine, ignore_index=True)
    else:
        combined_results = pd.DataFrame(columns=expected_cols)

    combined_summary = create_summary_df(combined_results)
    logger.log_step(
        step_name="Combined Results Summary",
        info={},
        df=combined_summary
    )

    return combined_results, rule_df, combined_summary