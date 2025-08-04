from typing import Dict, Any

"""
GROUPS: Dict[str, Any]
Parameter grouping schema used by `config_validator()`.

Keys are semantic types used to drive type and value validation.
- 'int', 'float', 'bool', etc. map to lists of param names.
- 'val_from_list': list of {param_name: valid_values}
- 'quantiles': list of {edges_param: labels_param} pairs.

Each param used in the Config dataclass must be included
Each param value must match the specified datatype.

Typical users don't need to edit this dict but advanced users 
may want to register additional parameters for validation.
"""

GROUPS = {

    'string': ['run_name', 'date_col', 'price_col', 'early_stop_metric', 'target_col', 'target_nan_placeholder', 'apriori_metric'],
    
    'float': ['missingness_row_thresh', 'missingness_col_thresh', 'max_imputed', 'winsor_lower_quantile', 
              'winsor_upper_quantile', 'correlation_threshold', 'variance_threshold', 'min_bin_fraction', 
              'stat_min_antecedent_support', 'stat_min_consequent_support',
              'stat_min_support', 'stat_min_confidence', 'stat_min_representativity', 
              'stat_min_leverage', 'stat_min_conviction',
              'stat_min_zhangs_metric', 'stat_min_jaccard', 'stat_min_certainty', 'stat_min_kulczynski', 
              'sc_lr', 'flip_feats_frac', 'flip_targs_frac',
              'apriori_min_support', 'apriori_min_metric', 
              'train_test_window_frac', 'train_test_step_frac',
              'wfa_window_frac', 'wfa_step_frac',
              'rel_error_threshold', 'correction_alpha'],

    'int': ['log_max_rows', 'winsor_dt_units', 'lag_num_missing', 'bin_dt_units', 'min_bin_obs', 'target_periods',
            'target_n_dt', 'vol_window',
            'stat_min_observations', 
            'sample_size', 'rulefit_tree_size', 'rulefit_min_depth', 'subgroup_top_n', 'subgroup_depth', 'subgroup_beam_width',
            'sdv_rows', 'sc_rows', 'sc_n_iter', 'sc_batch_size',
            'cart_max_depth', 'cart_random_state', 'cart_min_samples_split', 'cart_min_samples_leaf',
            'train_test_splits',
            'wfa_splits',
            'block_size', 'n_bootstrap',
            'n_null', 'es_m_permutations', 'scale_dt_units'],
    
    'bool': ['log_markdown', 'log_json',
             'to_drop_high_missingness', 'to_impute_numeric', 'to_impute_categorical', 'to_mask_high_impute', 
             'to_winsorize',
             'to_drop_var', 'to_drop_corr', 
             'to_engineer_ratios', 'to_engineer_dates', 'to_engineer_lags', 'to_sweep', 'to_drop_no_data',
             'to_calculate_target',
             'to_sample', 'drop_duplicates',
             'synth_silence', 'to_sdv', 'sdv_verbose', 'to_synthcity', 'corrupt_data',
             "to_aug_imbalance", "to_aug_flip_feats", "to_aug_flip_targets",
             'train_test_overlap', 'train_test_re_mine',
             'wfa_overlap', 'wfa_re_mine',
             'bootstrap_verbose', 'null_verbose', 'perform_train_test',
             'perform_wfa', 'perform_bootstrap', 'perform_null_fdr'
             ],
    
    'list': ['id_cols', 'drop_cols', 'robust_scale_quantile_range', 'n_dt_list', 'flat_threshold', 'bin_quantiles',
             'bin_quantile_labels', 'stat_bounds_lift', 'miners', 'train_test_ranges', 'train_test_fractions', 
             'wfa_ranges', 'wfa_fractions'],
    
    'all_or_list': ['winsor_cols', 'scale_cols', 'bin_cols'],

    'val_from_list': [{'winsor_grouping': {"none", "ids", "datetime", "datetime+ids"}},
                      {'scaling_method': {'zscore', 'robust', 'quantile_rank', 'unit_vector'}},
                      {'scale_grouping': {"none", "ids", "datetime", "datetime+ids"}},
                      {'engineer_cols': {"base", "all"}},
                      {'lag_mode': {"raw_only", "combined_only", "encoded_and_combined", "raw_and_combined"}},
                      {'return_mode': {"pct_change", "log_return", "vol_adjusted", "smoothed"}}, 
                      {'smoothing_method': {'median', 'mean', 'max', 'min'}}, 
                      {'bin_grouping': {"none", "ids", "datetime", "datetime+ids"}}, 
                      {'target_binning_method': {"quantile", "custom", "binary"}}, 
                      {'target_grouping': {"none", "ids", "datetime", "datetime+ids"}},
                      {'sdv_model': {'gaussian_copula', 'ctgan', 'tvae'}},
                      {'corrupt_target': {"none", "real", "synthetic", "both"}},
                      {'sc_model': {"ctgan", "tvae", "adsgan", "pategan", "rtvae"}},
                      {'sc_device': {"cpu", "cuda"}},
                      {'cart_criterion': {"gini", "entropy"}},
                      {'train_test_split_method': {"fractional", "temporal"}},
                      {'wfa_split_method': {"fractional", "temporal"}},
                      {'resample_method': {"traditional", "block", "block_ids"}},
                      {'shuffle_mode': {'target', 'rows', 'columns'}},
                      {'correction_metric': {'fdr_bh', 'fdr_by'}},
                      {'quantile_rank_mode': {"rank", "quantile_uniform", "quantile_normal"}},
                      {'vector_scale_norm_type': {"l2", "l1", "max"}},
                      {'impute_strategy': {'median', 'mean'}},
                      {'fdr_mode': {"greater", "less", "two-sided"}},
                     ],
    
    'quantiles': [{'bin_quantiles': 'bin_quantile_labels'}, 
                  {'target_bins': 'target_labels'}]
}