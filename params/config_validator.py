import pandas as pd
import numpy as np
import yaml
from collections import ChainMap
import logging
from dataclasses import dataclass
from typing import List, Union
from params.config_schema import GROUPS
from typing import Dict, Any, List, Optional, Union, Set


def pretty_print_params(chainmap, show_layers=False):
    """
    Pretty print a ChainMap of parameters.

    Args:
        chainmap (ChainMap): The ChainMap object.
        show_layers (bool): If True, show each layer separately.
                            If False, show merged params.
    """
    import yaml

    if show_layers:
        for i, layer in enumerate(chainmap.maps):
            print(f"\n--- Layer {i} ---")
            print(yaml.dump(layer, sort_keys=True, default_flow_style=False))
    else:
        merged = dict(chainmap)
        print(yaml.dump(merged, sort_keys=True, default_flow_style=False))

def is_integer(params_dict, param_key):
    if not isinstance(params_dict[param_key], int):
        raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be an integer")

def is_float(params_dict, param_key):
    if not isinstance(params_dict[param_key], (float, int)):
        raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be a float or int")

def is_string(params_dict, param_key):
    if not isinstance(params_dict[param_key], str):
        raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be a string")

def is_bool(params_dict, param_key):
    if not isinstance(params_dict[param_key], bool):
        raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be a boolean")

def is_list(params_dict, param_key):
    if not isinstance(params_dict[param_key], list):
        raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be a list")

def is_in_choices(choices, params_dict, param_key):
    """
    Returns a validator function that checks whether a string
    is one of the allowed choices.

    Args:
        choices (Iterable[str]): Set or list of allowed strings.
    """
    def _check(x, params_dict, param_key):
        if not isinstance(x, str):
            raise ValueError(f"Param {params_dict[param_key]} in {param_key} must be a string")
        if x not in choices:
            raise ValueError(f"{params_dict[param_key]} in {param_key} must be one of {sorted(choices)}")
    return _check

def is_valid_quantile_bins(bins):
    """
    Validates that bins:
    - are a list of numbers
    - start with 0
    - end with 1
    - are strictly increasing
    - all between 0 and 1
    """
    if not isinstance(bins, list):
        raise ValueError("must be a list")
    if len(bins) < 2:
        raise ValueError("must have at least two elements (start and end)")
    if any(not isinstance(q, (float, int)) for q in bins):
        raise ValueError("all elements must be numbers")
    if bins[0] != 0:
        raise ValueError("must start with 0")
    if bins[-1] != 1:
        raise ValueError("must end with 1")
    if not all(0 <= q <= 1 for q in bins):
        raise ValueError("all elements must be between 0 and 1")
    if sorted(bins) != bins:
        raise ValueError("elements must be in increasing order")
    if len(set(bins)) != len(bins):
        raise ValueError("elements must be unique")

def is_valid_custom_bins(bins):
    """
    Validates that custom bins:
    - are a list of numbers
    - strictly increasing
    - no NaNs
    - optionally warn if no infinities at ends
    """
    if not isinstance(bins, list):
        raise ValueError("must be a list")
    if len(bins) < 2:
        raise ValueError("must have at least two elements")
    if any(not isinstance(q, (float, int)) for q in bins):
        raise ValueError("all elements must be numeric")
    if any(pd.isna(q) for q in bins):
        raise ValueError("bins cannot contain NaN")
    if sorted(bins) != bins:
        raise ValueError("elements must be in increasing order")
    if len(set(bins)) != len(bins):
        raise ValueError("elements must be unique")
    # Optional warning
    if not (np.isneginf(bins[0]) and np.isposinf(bins[-1])):
        print("⚠️ Warning: custom bins do not start with -inf and end with +inf. "
              "Values outside the bin edges will be assigned NaN.")

def is_valid_bin_labels(labels, bins):
    """
    Validates that bin labels:
    - are a list of strings
    - have no duplicates
    - match the required length: len(bins) - 1
    """
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    if len(labels) == 0:
        raise ValueError("labels list cannot be empty")
    if any(not isinstance(l, str) for l in labels):
        raise ValueError("all labels must be strings")
    if len(labels) != len(bins) - 1:
        raise ValueError(
            f"labels length must be len(bins)-1 ({len(bins)-1}), got {len(labels)}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")

def is_all_or_list_of_strings(params_dict, param_key):
    """
    Validates that x is either:
    - the string "all"
    - or a list of strings
    """
    x = params_dict[param_key]
    if isinstance(x, str):
        if x != "all":
            raise ValueError(f"if {param_key} a string, must be 'all', it is {x}")
    elif isinstance(x, list):
        if not all(isinstance(item, str) for item in x):
            raise ValueError(f"if {param_key} a list, all elements must be strings, it is {x}")
        if len(set(x)) != len(x):
            raise ValueError(f"list elements must be unique, they are {x}")
    else:
        raise ValueError(f"{param_key} must be 'all' or a list of strings, it is {x}")

def config_validator(
    params_dict: Dict[str, Any],
    groups: Dict[str, Any] = GROUPS,
    logger: Optional[logging.Logger] = None,
    strict: bool = True
) -> None:
    """
    Validates configuration parameters in a config dictionary (`params_dict`) against
    predefined type and value constraints specified in `groups`.

    Parameters
    ----------
    params_dict : dict
        Dictionary of configuration parameters to validate.
    groups : dict
        Dictionary defining parameter groupings by type or constraint category. Expected keys include:
        - 'int', 'float', 'string', 'bool', 'list', 'all_or_list'
        - 'val_from_list': list of {param_name: valid_values}
        - 'quantiles': list of {edges_param: labels_param}
    logger : Optional[logging.Logger], default=None
        Logger for validation messages. If None, no logging is performed.
    strict : bool, default=True
        If True, unknown parameters not found in any group raise an error.

    Raises
    ------
    ValueError
        If any parameter fails validation.
    """
    errors: List[str] = []
    known_keys: Set[str] = _gather_known_keys(groups)

    for key, value in params_dict.items():
        if logger:
            logger.debug(f"Validating param: {key} → {value}")

        if _validate_type(key, value, groups, errors):
            continue
        if _validate_special_cases(key, value, params_dict, groups, errors):
            continue
        if strict and key not in known_keys:
            errors.append(f"[{key}] unexpected parameter. "
                          "Consider adding it to GROUPS or disable strict mode.")

    if errors:
        message = "\n".join(errors)
        if logger:
            logger.error("Configuration validation failed:\n" + message)
        raise ValueError("Configuration validation failed:\n" + message)

    if logger:
        logger.info("Configuration validation passed.")


def _gather_known_keys(groups: Dict[str, Any]) -> Set[str]:
    known = set()
    for category in ['int', 'float', 'string', 'bool', 'list', 'all_or_list']:
        known.update(groups.get(category, []))
    for val_group in groups.get('val_from_list', []):
        known.update(val_group.keys())
    for quant_group in groups.get('quantiles', []):
        known.update(quant_group.keys())
        known.update(quant_group.values())
    return known


def _validate_type(key: str, value: Any, groups: Dict[str, Any], errors: List[str]) -> bool:
    """Validate type-based groups. Returns True if handled."""
    type_checks = {
        'int': int,
        'float': (float, int),
        'string': str,
        'bool': bool,
        'list': list,
    }
    for group, expected_type in type_checks.items():
        if key in groups.get(group, []):
            if not isinstance(value, expected_type):
                errors.append(f"[{key}] must be {group}, got {type(value).__name__}")
            return True
    if key in groups.get('all_or_list', []):
        try:
            is_all_or_list_of_strings(params_dict={key: value}, param_key=key)
        except ValueError as e:
            errors.append(f"[{key}] {str(e)}")
        return True
    return False


def _validate_special_cases(
    key: str,
    value: Any,
    params_dict: Dict[str, Any],
    groups: Dict[str, Any],
    errors: List[str]
) -> bool:
    """Handle val_from_list and quantiles cases. Returns True if handled."""
    # val_from_list group
    for group in groups.get('val_from_list', []):
        if key in group:
            try:
                is_in_choices(group[key], params_dict, key)(value, params_dict, key)
            except ValueError as e:
                errors.append(f"[{key}] {str(e)}")
            return True

    # quantiles group
    for quant_group in groups.get('quantiles', []):
        edges_key, labels_key = list(quant_group.items())[0]
        if key == edges_key:
            try:
                if edges_key == 'target_bins' and params_dict.get('target_binning_method') == 'custom':
                    is_valid_custom_bins(value)
                else:
                    is_valid_quantile_bins(value)
            except ValueError as e:
                errors.append(f"[{key}] {str(e)}")
            return True
        elif key == labels_key:
            edges = params_dict.get(edges_key)
            if edges is None:
                errors.append(f"[{key}] missing associated bins '{edges_key}' to validate labels")
            else:
                try:
                    is_valid_bin_labels(value, edges)
                except ValueError as e:
                    errors.append(f"[{key}] {str(e)}")
            return True
    return False

def load_params(
    default_params: Union[str, Dict[str, Any], None] = "params/default_params.yaml", 
    custom_params: Union[str, Dict[str, Any], None] = "params/custom_params.yaml",
    verbose: bool = False,
    logger: Optional[Any] = None,
) -> ChainMap:
    """
    Load and merge default and custom configuration parameters.

    Parameters
    ----------
    default_params : str, dict, or None, default="params/default_params.yaml"
        Path to a YAML file or a Python dictionary of default parameters.
        If None, treated as an empty dictionary.
    
    custom_params : str, dict, or None, default="params/custom_params.yaml"
        Path to a YAML file or a Python dictionary of custom parameters.
        If None, treated as an empty dictionary. Custom overrides default.

    verbose : bool, default=False
        If True, prints a formatted summary of the merged parameters
        and includes detailed validation logging (if a logger is provided).

    logger : logging.Logger or None, default=None
        Optional logger used to report validation status.

    Returns
    -------
    ChainMap
        A merged ChainMap of parameters with custom values overriding defaults.

    Raises
    ------
    ValueError
        If a parameter source is neither None, a dictionary, nor a valid YAML file path.
    """
    def _load_param_source(source: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return source
        if isinstance(source, str) and source.endswith(".yaml"):
            try:
                with open(source, "r") as f:
                    return yaml.safe_load(f) or {}
            except FileNotFoundError:
                return {}
        raise ValueError(f"Unsupported parameter type: {type(source)}")

    default_dict = _load_param_source(default_params)
    custom_dict = _load_param_source(custom_params)

    combined = ChainMap(custom_dict, default_dict)

    # Validate and optionally display configuration
    config_validator(combined, logger=logger, strict=True)
    if verbose:
        pretty_print_params(combined)

    return combined

@dataclass
class Config:
    """
    Configuration object for the full research pipeline.
    """
    # Setup
    run_name: str
    id_cols: List[str]
    date_col: str
    log_markdown: bool 
    log_json: bool
    log_max_rows: int
    drop_cols: List[str]
    
    # Missingness
    to_drop_high_missingness: bool
    missingness_col_thresh: float
    missingness_row_thresh: float

    # Imputation
    to_impute_numeric: bool
    impute_strategy: str
    to_impute_categorical: bool
    to_mask_high_impute: bool
    max_imputed: float

    # Winsorization
    to_winsorize: bool
    winsor_cols: Union[str, List[str]]
    winsor_grouping: str
    winsor_dt_units: int
    winsor_lower_quantile: float
    winsor_upper_quantile: float

    # Scaling
    scaling_method: str
    scale_cols: Union[str, List[str]]
    scale_grouping: str
    scale_dt_units: int
    robust_scale_quantile_range: List[int]
    quantile_rank_mode: str
    vector_scale_norm_type: str

    # Variance check
    to_drop_var: bool
    variance_threshold: float

    # Correlation check
    to_drop_corr: bool
    correlation_threshold: float

    # Feature Engineering
    engineer_cols: str
    to_engineer_ratios: bool
    to_engineer_dates: bool
    to_engineer_lags: bool
    
    # Lags
    lag_mode: str
    n_dt_list: List[int]
    flat_threshold: List[float]
    lag_num_missing: int
    
    # Binning
    bin_cols: Union[str, List[str]]
    bin_quantiles: List
    bin_quantile_labels: Union[None, List[str]]
    bin_grouping: str
    bin_dt_units: int
    to_sweep: bool
    to_drop_no_data: bool
    min_bin_obs: int
    min_bin_fraction: float

    # Target calculation
    to_calculate_target: bool
    target_periods: int
    price_col: str
    return_mode: str
    vol_window: int
    smoothing_method: str
    target_col: str
    
    # Target binning
    target_binning_method: str
    target_bins: List[float]
    target_labels: List[str]
    target_grouping: str
    target_n_dt: int
    target_nan_placeholder: str

    # Statistics Mask
    stat_min_support: float
    stat_min_observations: int
    stat_bounds_lift: List[float]
    stat_min_antecedent_support: float
    stat_min_consequent_support: float
    stat_min_confidence: float
    stat_min_representativity: float
    stat_min_leverage: float
    stat_min_conviction: float
    stat_min_zhangs_metric: float
    stat_min_jaccard: float
    stat_min_certainty: float
    stat_min_kulczynski: float
    
    # Data prep
    to_sample: bool
    sample_size: int
    drop_duplicates: bool
    
    # Synthetic data
    synth_silence: bool
    
    # SDV
    to_sdv: bool
    sdv_rows: int
    sdv_model: str
    sdv_verbose: bool
    
    # Synthcity
    to_synthcity: bool
    sc_rows: int
    sc_model: str
    sc_n_iter: int
    sc_batch_size: int
    sc_lr: float
    sc_device: str
    
    # Bad data Augmentation
    to_aug_imbalance: bool
    to_aug_flip_feats: bool
    to_aug_flip_targets: bool
    flip_feats_frac: float
    flip_targs_frac: float
    
    # Miners
    miners: List[str]
    corrupt_data: bool
    corrupt_target: str

    # Apriori
    apriori_min_support: float
    apriori_metric: str
    apriori_min_metric: float
    
    # Rulefit
    rulefit_tree_size: int
    rulefit_min_depth: int
    
    # Subgroup
    subgroup_top_n: int
    subgroup_depth: int
    subgroup_beam_width: int
    
    # CART
    cart_max_depth: int
    cart_criterion: str
    cart_random_state: int
    cart_min_samples_split: int
    cart_min_samples_leaf: int
    
    # Train Test
    perform_train_test: bool
    train_test_split_method: str
    train_test_splits: int
    train_test_ranges: List[List[str]]
    train_test_window_frac: float
    train_test_step_frac: float
    train_test_fractions: List[float]
    train_test_overlap: bool
    train_test_re_mine: bool
    
    # WFA
    perform_wfa: bool
    wfa_split_method: str
    wfa_splits: int
    wfa_ranges: List[List[str]]
    wfa_window_frac: float
    wfa_step_frac: float
    wfa_fractions: List[float]
    wfa_overlap: bool
    wfa_re_mine: bool
    
    # Bootstrap
    perform_bootstrap: bool
    block_size: int
    resample_method: str
    n_bootstrap: int
    bootstrap_verbose: bool

    # Null
    perform_null_fdr: bool
    shuffle_mode: str
    n_null: int
    null_verbose: bool
    early_stop_metric: str
    es_m_permutations: int
    rel_error_threshold: float

    # Multiple Corrections
    correction_metric: str
    correction_alpha: float
    fdr_mode: str