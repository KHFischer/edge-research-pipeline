import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Callable, List

_EPS = 1e-12  # small constant to avoid divide-by-zero

def get_stat_registry() -> dict[str, callable]:
    """
    Returns the registry of all association statistics used for rule calculation and filtering.

    Each entry maps a statistic name to a callable that computes the metric given
    contingency table counts and precomputed probabilities.

    These statistics are automatically calculated for each rule and can be referenced
    in YAML-based filtering configurations (e.g. stat_min_lift, stat_bounds_confidence).

    Returns:
        dict[str, callable]: Mapping of metric names to calculation functions.
    """
    return {
        "support": lambda m: m["a"] / m["total"],
        "antecedent_support": lambda m: m["feature_total_1"] / m["total"],
        "consequent_support": lambda m: m["target_total"] / m["total"],
        "confidence": lambda m: m["a"] / (m["feature_total_1"] + _EPS),
        "lift": lambda m: m["confidence"] / (m["consequent_support"] + _EPS),
        "leverage": lambda m: m["support"] - m["antecedent_support"] * m["consequent_support"],
        "conviction": lambda m: (1 - m["consequent_support"] + _EPS) / (1 - m["confidence"] + _EPS),
        "zhangs_metric": lambda m: m["leverage"] / np.maximum.reduce([
            m["support"] * ((m["b"] + m["d"]) / m["total"]),
            m["consequent_support"] * (m["b"] / m["total"]),
            np.full_like(m["support"], _EPS)
        ]),
        "jaccard": lambda m: m["support"] / (
            m["antecedent_support"] + m["consequent_support"] - m["support"] + _EPS
        ),
        "representativity": lambda m: m["a"] / (m["target_total"] + _EPS),
        "certainty": lambda m: m["confidence"] - (m["c"] / (m["c"] + m["d"] + _EPS)),
        "kulczynski": lambda m: 0.5 * (
            m["confidence"] + (m["a"] / m["total"]) / (m["consequent_support"] + _EPS)
        ),
        "observations": lambda m: m["a"]
    }
    
def _get_binary_counts(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str
) -> pd.DataFrame:
    """
    Compute occurrence counts of (feature_value, target_class) pairs 
    for a set of binary features in a dataframe.

    Args:
        df (pd.DataFrame):
            Input dataframe containing binary features and a target column.
        feature_cols (list of str):
            List of feature column names. All must be strictly binary (0/1 or bool).
        target_col (str):
            Name of the target column. No restrictions on its datatype.

    Returns:
        pd.DataFrame:
            Long-format dataframe with columns:
                - 'feature': Feature name.
                - 'feature_value': 0 or 1 (feature state).
                - target_col: Target class or value.
                - 'count': Count of occurrences of that (feature_value, target_class) pair.

    Raises:
        ValueError:
            If any feature column contains non-binary values.

    Notes:
        This function prepares contingency table counts for association metric
        calculations. It assumes the target column is pre-cleaned and does not
        validate its contents.
    """
    # Coerce bool features to uint8 for consistency
    bool_cols = [c for c in feature_cols if df[c].dtype == bool]
    if bool_cols:
        df = df.copy()
        df[bool_cols] = df[bool_cols].astype("uint8")

    # Validate binary constraint for all feature columns
    for col in feature_cols:
        unique_vals = set(df[col].unique())
        if not unique_vals <= {0, 1}:
            raise ValueError(
                f"Feature '{col}' must be strictly binary (0/1 or bool). Found values: {sorted(unique_vals)}"
            )

    # Reshape dataframe: long-format rows per feature + target
    melted = df.melt(
        id_vars=target_col,
        value_vars=feature_cols,
        var_name="feature",
        value_name="feature_value"
    )

    # Compute counts per (feature, value, target) combination
    counts = (
        melted
        .groupby(["feature", "feature_value", target_col], observed=False)
        .size()
        .reset_index(name="count")
    )

    return counts

def calculate_association_metrics(
    df: pd.DataFrame,
    target_col: str = "return",
    stat_registry: Optional[Dict[str, Callable[[dict], float]]] = None
) -> pd.DataFrame:
    """
    Compute association metrics between binary features (antecedents) 
    and a target column (consequents) using vectorized contingency tables.

    Args:
        df (pd.DataFrame):
            Input dataframe containing binary features and a target column.
            Features must be strictly binary (0/1 or bool).
        target_col (str, optional):
            Name of the target column. Defaults to 'return'.

    Returns:
        pd.DataFrame:
            Dataframe containing:
                - antecedents: Feature descriptions (e.g. 'feature == 1').
                - consequents: Target class or value.
                - Computed statistics from the metrics registry.
            Sorted descending by 'support'.

    Notes:
        - Uses a registry-driven approach to compute all metrics.
        - Assumes input dataframe is pre-validated for binary features.
        - Metrics are calculated using vectorized Pandas operations.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    total_rows = len(df)
    if stat_registry is None:
        stat_registry = get_stat_registry()

    # Compute (feature, feature_value, target) counts
    counts = _get_binary_counts(df, feature_cols, target_col)

    # Pivot to get counts of feature==1 (a) and feature==0 (c) per target
    per_target = (
        counts
        .pivot_table(
            index=["feature", target_col],
            columns="feature_value",
            values="count",
            fill_value=0
        )
        .reset_index()
        .rename(columns={1: "a", 0: "c"})
    )

    # Compute total counts for each feature
    feat_totals = (
        counts
        .groupby(["feature", "feature_value"], observed=False)["count"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={1: "feature_total_1", 0: "feature_total_0"})
        .reset_index()
    )

    # Merge counts into a single contingency table
    m = per_target.merge(feat_totals, on="feature", how="left")
    m["b"] = m["feature_total_1"] - m["a"]
    m["d"] = m["feature_total_0"] - m["c"]
    m["target_total"] = m["a"] + m["c"]
    m["total"] = total_rows

    # Compute metrics using the registry
    for metric_name, func in stat_registry.items():
        m[metric_name] = func(m)

    # Add human-readable rule description columns
    m["antecedents"] = m["feature"] + " == 1"
    m["consequents"] = m[target_col]

    # Finalize output
    output_columns = ["antecedents", "consequents"] + list(stat_registry.keys())

    return (
        m[output_columns]
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )

def apply_statistic_filters(
    result_df: pd.DataFrame,
    filter_config: dict[str, object]
) -> pd.DataFrame:
    """
    Apply dynamic threshold filters to association rule metrics and
    annotate each rule with a 'selected' column.

    The filter_config must use keys in the format:
        - 'stat_min_{metric}'
        - 'stat_max_{metric}' or 'stat_upper_{metric}'
        - 'stat_lower_{metric}'
        - 'stat_bounds_{metric}': [lower_bound, upper_bound]
        - 'stat_range_{metric}': [lower_bound, upper_bound]

    Filters are applied as:
        - 'min': metric >= threshold
        - 'max'/'upper': metric <= threshold
        - 'lower': metric >= threshold
        - 'bounds': metric <= lower_bound or >= upper_bound (outside range)
        - 'range': metric inside [lower_bound, upper_bound] inclusive

    Args:
        result_df (pd.DataFrame):
            DataFrame containing rule metrics. Must include columns referenced by filter_config.
        filter_config (dict):
            Dictionary of filtering rules. Keys specify condition and metric;
            values specify thresholds (float or list of two floats).

    Returns:
        pd.DataFrame:
            Copy of result_df with added 'selected' boolean column indicating
            whether each rule passes all filter conditions.

    Raises:
        ValueError:
            If unknown filter key formats or invalid threshold specifications are provided.
    """
    mask = pd.Series(True, index=result_df.index)

    for key, threshold in filter_config.items():
        if not key.startswith("stat_") or key.count("_") < 2:
            raise ValueError(f"Invalid filter key format: '{key}'")

        _, condition, *metric_parts = key.split("_")
        metric = "_".join(metric_parts)

        if metric not in result_df.columns:
            raise ValueError(f"Metric '{metric}' not found in result dataframe columns.")

        column = result_df[metric]

        if condition == "min":
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for '{key}' must be numeric.")
            mask &= column >= threshold

        elif condition in {"max", "upper"}:
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for '{key}' must be numeric.")
            mask &= column <= threshold

        elif condition == "lower":
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for '{key}' must be numeric.")
            mask &= column >= threshold

        elif condition in {"bounds", "range"}:
            if not (isinstance(threshold, (list, tuple)) and len(threshold) == 2):
                raise ValueError(f"Threshold for '{key}' must be a 2-element list or tuple.")
            lower, upper = threshold
            if not all(isinstance(x, (int, float)) for x in (lower, upper)):
                raise ValueError(f"Both bounds for '{key}' must be numeric.")

            if condition == "bounds":
                mask &= (column <= lower) | (column >= upper)
            else:  # 'range'
                mask &= (column >= lower) & (column <= upper)

        else:
            raise ValueError(f"Unsupported filter condition '{condition}' in key '{key}'.")

    result_df = result_df.copy()
    result_df["selected"] = mask

    return result_df

def extract_stat_filter_config(cfg) -> dict[str, object]:
    """
    Parses the cfg dataclass to extract all keys starting with 'stat_'.

    Args:
        cfg: Config dataclass instance.

    Returns:
        Dictionary of stat_* keys and their corresponding values.
    """
    config_dict = vars(cfg)
    
    filter_config = {
        key: value
        for key, value in config_dict.items()
        if key.startswith("stat_")
    }
    
    if not filter_config:
        raise ValueError("No 'stat_' keys found in configuration.")

    return filter_config

def build_filter_config(cfg, overrides: dict[str, object] = None) -> dict[str, object]:
    """
    Combines parsed config with optional manual overrides.

    Args:
        cfg: Config dataclass instance.
        overrides: Optional manual override dictionary.

    Returns:
        Finalized filter_config dictionary.
    """
    filter_config = extract_stat_filter_config(cfg)
    
    if overrides:
        filter_config.update(overrides)  # Manual values take precedence

    return filter_config

def apply_statistic_filters(
    result_df: pd.DataFrame,
    filter_config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply dynamic filtering rules to rule metrics and annotate selection status.

    Filtering rules are defined in filter_config using flat keys in the format:
        - 'stat_min_{metric}'      → metric >= threshold
        - 'stat_max_{metric}'      → metric <= threshold
        - 'stat_bounds_{metric}'   → metric <= lower_bound or >= upper_bound
        - 'stat_range_{metric}'    → lower_bound <= metric <= upper_bound

    Metrics failing any filter are marked as not selected.
    The full dataframe is returned with a boolean 'selected' column.

    Args:
        result_df (pd.DataFrame):
            Rule metrics dataframe. Must include all metrics referenced in filter_config.
        filter_config (dict):
            Dictionary of filtering thresholds using flat, prefixed keys.

    Returns:
        result_df (pd.DataFrame):
            Copy of the dataframe with 'selected' column indicating pass/fail.
        summary_df (pd.DataFrame):
            One-row summary dataframe with rule counts and metric summary stats.

    Raises:
        ValueError:
            If key format, condition, metric existence, or threshold type is invalid.

    Notes:
        This function supports both fixed thresholds and interval-based conditions.
        New metrics can be added without modifying this function.

    Example:
        filter_config = {
            "stat_min_support": 0.01,
            "stat_bounds_lift": [0.9, 1.1]
        }
    """
    VALID_CONDITIONS = {"min", "max", "bounds", "range"}
    mask = pd.Series(True, index=result_df.index)

    for key, threshold in filter_config.items():
        if not key.startswith("stat_") or key.count("_") < 2:
            raise ValueError(f"Invalid filter key format: '{key}'. Must be 'stat_{{condition}}_{{metric}}'.")

        _, condition, *metric_parts = key.split("_")
        metric = "_".join(metric_parts)

        if condition not in VALID_CONDITIONS:
            raise ValueError(f"Invalid filter condition '{condition}' in key '{key}'. "
                             f"Must be one of {sorted(VALID_CONDITIONS)}.")

        if metric not in result_df.columns:
            raise ValueError(f"Metric '{metric}' from key '{key}' not found in dataframe columns.")

        column = result_df[metric]

        # Threshold validation
        if condition in {"min", "max"}:
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for '{key}' must be numeric.")
        elif condition in {"bounds", "range"}:
            if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
                raise ValueError(f"Threshold for '{key}' must be a list of two numeric values.")
            if not all(isinstance(val, (int, float)) for val in threshold):
                raise ValueError(f"Threshold list for '{key}' must contain numeric values only.")

        # Apply filtering
        if condition == "min":
            mask &= column >= threshold
        elif condition == "max":
            mask &= column <= threshold
        elif condition == "bounds":
            lower, upper = threshold
            mask &= (column <= lower) | (column >= upper)
        elif condition == "range":
            lower, upper = threshold
            mask &= (column >= lower) & (column <= upper)

    # Add 'selected' column
    result_df = result_df.copy()
    result_df["selected"] = mask

    # Summary statistics
    summary_data = {
        "total_rules": [len(result_df)],
        "rules_selected": [mask.sum()],
        "rules_not_selected": [(~mask).sum()],
    }

    for col in result_df.columns:
        if result_df[col].dtype.kind in {"f", "i"}:
            summary_data[f"{col}_min"] = [result_df[col].min()]
            summary_data[f"{col}_max"] = [result_df[col].max()]
            summary_data[f"{col}_mean"] = [result_df[col].mean()]

    summary_df = pd.DataFrame(summary_data)

    return result_df, summary_df

def extend_stat_registry(
    custom_stats: Optional[Dict[str, Callable[[dict], float]]] = None,
    exclude_stats: Optional[List[str]] = None,
    allow_override: bool = False
) -> Dict[str, Callable[[dict], float]]:
    """
    Create an extended statistics registry with optional additions and exclusions.

    This function clones the base statistics registry, optionally excludes unwanted
    metrics, and adds user-defined custom statistics. Each statistic function must 
    return a single numeric value per row (scalar) when applied.

    Args:
        custom_stats (Optional[Dict[str, Callable[[dict], float]]]): 
            Dictionary of custom metrics to add. Each function must accept a metrics 
            dictionary for a single row and return a float or int scalar.
        
        exclude_stats (Optional[List[str]]): 
            List of standard metric names to exclude from the final registry.

        allow_override (bool): 
            If True, custom_stats may overwrite existing metrics. 
            Defaults to False.

    Returns:
        Dict[str, Callable[[dict], float]]: 
            Dictionary mapping metric names to callable functions.

    Raises:
        ValueError: 
            If a custom_stat key conflicts with existing metrics and allow_override is False.
            If any custom_stat function fails to return a scalar on test input.
    """
    registry = get_stat_registry().copy()

    if exclude_stats:
        for stat_name in exclude_stats:
            registry.pop(stat_name, None)

    if custom_stats:
        for stat_name, func in custom_stats.items():
            if stat_name in registry and not allow_override:
                raise ValueError(
                    f"Custom statistic '{stat_name}' already exists. "
                    "Set allow_override=True to replace."
                )

            # Validate function returns scalar
            test_input = {
                "a": 1, "b": 1, "c": 1, "d": 1,
                "feature_total_1": 1, "feature_total_0": 1,
                "target_total": 1, "total": 1,
                "support": 0.5, "confidence": 0.5, "lift": 1.0
            }
            try:
                result = func(test_input)
            except Exception as e:
                raise ValueError(
                    f"Custom statistic '{stat_name}' raised an error on test input: {e}"
                ) from e

            if not isinstance(result, (int, float)):
                raise ValueError(
                    f"Custom statistic '{stat_name}' must return a numeric scalar. "
                    f"Received {type(result)}."
                )

            registry[stat_name] = func

    return registry

def composite_score(m):
    """
    Computes a heuristic composite score for a feature-target rule.

    This score combines three standard association metrics:
        - lift: measures how much the rule outperforms baseline probability.
        - confidence: indicates conditional success probability.
        - support: reflects pattern frequency, included as sqrt(support) 
                   to reward non-trivial prevalence without over-weighting common patterns.

    Formula:
        composite_score = lift * confidence * sqrt(support)

    This metric is not intended as a definitive strength measure but as a balanced,
    general-purpose example of combining multiple association metrics into a single score.
    It prioritizes signal strength (lift, confidence) while moderating for rare or overly common patterns (support).

    Returns:
        float: Composite score for a single rule.
    """
    return m["lift"] * m["confidence"] * (m["support"] ** 0.5)

def generate_statistics(
    df: pd.DataFrame,
    cfg,
    overrides: Optional[Dict[str, object]] = None,
    stat_registry: Optional[Dict[str, Callable[[dict], float]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate rule-level statistics and apply configurable selection filters.

    This function wraps the full pipeline of:
        - Dropping non-feature columns.
        - Computing association metrics for each binary feature vs. target.
        - Applying dynamic filtering rules using YAML-configurable thresholds.
        - Returning annotated results and a summary of filtering outcomes.

    Args:
        df (pd.DataFrame):
            Input dataframe, including all columns (IDs, dates, features, target).
        cfg:
            Configuration dataclass instance containing:
                - id_cols: List of identifier columns.
                - date_col: Name of the date column.
                - drop_cols: Columns to exclude from analysis.
                - target_col: Name of the binary/categorical target column.
        overrides (dict, optional):
            Optional manual overrides for filter_config.
            Overrides update the YAML-configured thresholds at runtime.

    Returns:
        result_df (pd.DataFrame):
            Dataframe with rule statistics and a 'selected' boolean column.
        stat_log (pd.DataFrame):
            Single-row dataframe summarizing filtering outcomes and basic metric statistics.

    Raises:
        ValueError:
            If target column is missing after dropping non-feature columns.

    Example:
        result_df, summary_df = generate_statistics(my_df, cfg)
    """
    overrides = overrides or {}

    # Drop non-feature columns
    df_working = df.copy()
    non_feature_cols = cfg.id_cols + [cfg.date_col] + cfg.drop_cols
    df_working.drop(columns=non_feature_cols, inplace=True, errors='ignore')

    # Ensure target column remains
    if cfg.target_col not in df_working.columns:
        raise ValueError(
            f"Target column '{cfg.target_col}' missing after dropping non-feature columns."
        )

    if stat_registry is None:
        stat_registry = get_stat_registry()

    # Calculate rule statistics
    result_df = calculate_association_metrics(
        df_working,
        target_col=cfg.target_col,
        stat_registry=stat_registry
    )
    # Load YAML + manual filter configuration
    filter_config = build_filter_config(cfg, overrides=overrides)

    # Apply filters and annotate 'selected'
    result_df, stat_log = apply_statistic_filters(result_df, filter_config)

    return result_df, stat_log
