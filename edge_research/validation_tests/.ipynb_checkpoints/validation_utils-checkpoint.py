import numpy as np
import pandas as pd
from typing import List, Tuple, Callable
from scripts.rules_mining.mining_parsing import generate_multivariate_feature_df, map_antecedents_to_readable
from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline
from scripts.utils.utils import parse_human_readable_rules

def extract_fulldata_rules(combined_results: pd.DataFrame):
    # Extract univariate rules
    def parse_single_clause(s):
        parts = s.strip().split("==")
        if len(parts) != 2:
            raise ValueError(f"Unexpected univariate clause: {s}")
        return parts[0].strip()

    univar_rules = combined_results[
        (combined_results["depth"] == 1) &
        (combined_results["selected"] == True)
    ]
    parsed_univar_rules = [parse_single_clause(s) for s in univar_rules["antecedents"]]

    # 3. Extract multivariate rules
    multivar_rules = combined_results[
        (combined_results["depth"] >= 2) &
        ~(combined_results["antecedents"].str.contains("NOT")) &
        (combined_results["selected"] == True)
    ]
    parsed_multivar_rules = parse_human_readable_rules(multivar_rules["antecedents"])
    return parsed_univar_rules, parsed_multivar_rules

def parse_rules_readable(merged_results: pd.DataFrame, multivar_rule_log: pd.DataFrame):
    # Split readable/unreadable rules
    unreadable_rules = merged_results[merged_results["antecedents"].str.match(r"^rule_")]
    readable_rules = merged_results[~merged_results["antecedents"].str.match(r"^rule_")]

    # Map back to readable
    if not unreadable_rules.empty:
        readable_stats_df = map_antecedents_to_readable(unreadable_rules, multivar_rule_log)
        res = pd.concat([readable_rules, readable_stats_df], ignore_index=True)
    else:
        res = readable_rules.copy()

    # 9. Compute depth
    res["depth"] = res["antecedents"].apply(
        lambda x: x.lower().count(" and ") + 1 if isinstance(x, str) else 1
    )
    return res
    
def evaluator_toggled(
    prepared_df: pd.DataFrame,
    cfg: Config,
    func: Callable,
    func_args: tuple,
    discovery_mode: str,
    logger: MarkdownLogger,
    target_col: str = 'return'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if discovery_mode == "discover_on_full_data":
        # Full-data mining
        combined_results, rules_log_df, combined_summary = run_edge_discovery_pipeline(
            prepared_df=prepared_df,
            cfg=cfg,
            logger=logger,
            mine_rules=True
        )
        # Extract mined rules
        parsed_univar_rules, parsed_multivar_rules = extract_fulldata_rules(combined_results)
        
        # Encode multivariate rules
        if parsed_multivar_rules:
            multivar_rule_df, multivar_rule_log = generate_multivariate_feature_df(parsed_multivar_rules, prepared_df)
            prepared_no_target = prepared_df.drop(target_col, axis=1)
            all_rules_df = pd.concat([prepared_no_target, multivar_rule_df], axis=1)
        else:
            all_rules_df = prepared_df.copy()
    
        # Run test without new mining and translate encoded rules back
        res, res_log = func(df=all_rules_df, **func_args, mine_rules=False)
        res = parse_rules_readable(res, multivar_rule_log)
    
    else:
        # Run test with new mining and translate encoded rules back
        res, res_log = func(df=prepared_df, **func_args, mine_rules=True)

    return res, res_log