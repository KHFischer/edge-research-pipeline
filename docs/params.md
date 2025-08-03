## ⚙️ Config: Setup

| Parameter      | Type        | Default        | Description                                              | Valid Values / Notes                                  | Used In             |
|----------------|-------------|----------------|----------------------------------------------------------|--------------------------------------------------------|---------------------|
| run_name       | str         | 'my_first_test'| Identifier for the run, used in logs and exports         | Any non-empty string                                   | logging, saving results      |
| id_cols        | List[str]   | ['ticker']     | Columns that uniquely identify an instrument             | Must exist in input DataFrame                          | preprocessing  |
| date_col       | str         | 'date'         | Primary datetime column for time-based operations        | Must be parsable as datetime                           | preprocessing |
| log_markdown   | bool        | True           | Whether to write logs to markdown output                 | True or False                                          | logging             |
| log_json       | bool        | True           | Whether to write logs to structured JSON output          | True or False                                          | logging             |
| log_max_rows   | int         | 20             | Max number of rows to log from DataFrames                | Must be ≥ 0                                            | logging             |
| drop_cols      | List[str]   | []             | Columns to drop before rule mining phase                 | Used in all steps but excluded from mining             | preprocessing     |

## ⚙️ Config: Missingness

| Parameter               | Type   | Default | Description                                         | Valid Values / Notes                               | Used In           |
|-------------------------|--------|---------|-----------------------------------------------------|----------------------------------------------------|-------------------|
| to_drop_high_missingness| bool   | True    | Drop columns with excessive missing values          | True or False                                      | feature cleaning  |
| missingness_row_thresh  | float  | 0.9     | Drop row if ≥ this fraction is missing              | 0.0–1.0                                            | feature cleaning  |
| missingness_col_thresh  | float  | 0.9     | Drop column if ≥ this fraction is missing           | 0.0–1.0                                            | feature cleaning  |

## ⚙️ Config: Imputation

| Parameter              | Type   | Default | Description                                           | Valid Values / Notes                      | Used In           |
|------------------------|--------|---------|-------------------------------------------------------|-------------------------------------------|-------------------|
| to_impute_numeric      | bool   | True    | Whether to impute missing numeric values             | True or False                             | feature cleaning  |
| impute_strategy        | str    | 'median'| Strategy for numeric imputation                      | "mean", "median"                          | feature cleaning  |
| to_impute_categorical  | bool   | True    | Whether to impute missing categorical values         | True or False                             | feature cleaning  |
| to_mask_high_impute    | bool   | True    | Mask features exceeding max imputation threshold     | True or False                             | feature cleaning  |
| max_imputed            | float  | 0.5     | Max allowed imputed fraction before masking feature  | 0.0–1.0                                   | feature cleaning  |

## ⚙️ Config: Winsorization

| Parameter              | Type                | Default | Description                                          | Valid Values / Notes                                     | Used In           |
|------------------------|---------------------|---------|------------------------------------------------------|----------------------------------------------------------|-------------------|
| to_winsorize           | bool                | False   | Enable winsorization of numeric values               | True or False                                             | feature cleaning  |
| winsor_cols            | str or List[str]    | 'all'   | Columns to winsorize or 'all' for all numerics       | 'all' or list of column names                            | feature cleaning  |
| winsor_grouping        | str                 | datetime| How to group data before winsorizing                 | "none", "ids", "datetime", "datetime+ids"                | feature cleaning  |
| winsor_dt_units        | int                 | 60      | Grouping window in datetime units                    | Depends on granularity of `date_col`                     | feature cleaning  |
| winsor_lower_quantile  | float               | 0.01    | Lower bound for winsorization                        | 0.0–1.0, must be < upper quantile                        | feature cleaning  |
| winsor_upper_quantile  | float               | 0.99    | Upper bound for winsorization                        | 0.0–1.0, must be > lower quantile                        | feature cleaning  |

## ⚙️ Config: Scaling

| Parameter                  | Type              | Default           | Description                                          | Valid Values / Notes                                      | Used In                  |
|----------------------------|-------------------|-------------------|------------------------------------------------------|-----------------------------------------------------------|---------------------------|
| scaling_method             | str               | 'zscore'          | Method used to scale numeric features                | "zscore", "robust", "quantile_rank", "unit_vector"        | feature cleaning (scaling)|
| scale_cols                 | str or List[str]  | 'all'             | Columns to scale or 'all' for inferred numerics      | 'all' or list of column names                             | feature cleaning (scaling)|
| scale_grouping             | str               | 'none'            | Grouping method for scaling logic                    | "none", "ids", "datetime"                                 | feature cleaning (scaling)|
| scale_dt_units             | int               | 20                | Group size in datetime units for temporal scaling     | Integer > 0                                               | feature cleaning (scaling)|
| robust_scale_quantile_range| List[int]         | [25, 75]          | Quantile range used for robust scaling               | Typically [25, 75] for IQR                                | feature cleaning (scaling)|
| quantile_rank_mode         | str               | 'quantile_normal' | Mode for quantile rank scaling                       | "rank", "quantile_uniform", "quantile_normal"             | feature cleaning (scaling)|
| vector_scale_norm_type     | str               | 'l2'              | Norm type for unit vector scaling                    | "l2", "l1", "max"                                         | feature cleaning (scaling)|

## ⚙️ Config: Variance Check

| Parameter          | Type   | Default   | Description                                        | Valid Values / Notes                      | Used In           |
|--------------------|--------|-----------|----------------------------------------------------|-------------------------------------------|-------------------|
| to_drop_var        | bool   | False     | Whether to drop low-variance columns              | True or False                             | feature cleaning  |
| variance_threshold | float  | 0.00001   | Threshold below which variance triggers drop      | 0.0 or higher                             | feature cleaning  |

## ⚙️ Config: Correlation Check

| Parameter             | Type   | Default | Description                                         | Valid Values / Notes                    | Used In           |
|-----------------------|--------|---------|-----------------------------------------------------|-----------------------------------------|-------------------|
| to_drop_corr          | bool   | True    | Whether to drop highly correlated features          | True or False                           | feature cleaning  |
| correlation_threshold | float  | 0.9     | Absolute correlation threshold for dropping columns | 0.0–1.0                                 | feature cleaning  |

## ⚙️ Config: Feature Engineering

| Parameter            | Type   | Default | Description                                              | Valid Values / Notes                             | Used In             |
|----------------------|--------|---------|----------------------------------------------------------|--------------------------------------------------|---------------------|
| engineer_cols        | str    | 'base'  | Which columns to apply engineering to                   | "base" (pre-existing numerics) or "all"          | feature engineering |
| to_engineer_ratios   | bool   | True    | Generate all pairwise ratio features                    | True or False                                   | feature engineering |
| to_engineer_dates    | bool   | False   | Extract calendar features from date column              | True or False                                   | feature engineering |
| to_engineer_lags     | bool   | True    | Create temporal percent change lag features             | True or False                                   | feature engineering |

## ⚙️ Config: Lags

| Parameter         | Type        | Default                 | Description                                               | Valid Values / Notes                                           | Used In             |
|-------------------|-------------|-------------------------|-----------------------------------------------------------|----------------------------------------------------------------|---------------------|
| lag_mode          | str         | "encoded_and_combined"  | Determines how lag features are encoded and combined      | "raw_only", "combined_only", "encoded_and_combined", "raw_and_combined" | feature engineering |
| n_dt_list         | List[int]   | [4, 2, 1]               | Lag intervals to compute percent change over              | First element only used if `lag_mode == "raw_only"`           | feature engineering |
| flat_threshold    | List[float] | [-0.01, 0.01]           | Value range to classify flat lag behavior                 | Lower and upper bounds for flat classification                | feature engineering |
| lag_num_missing   | int         | 0                       | Max allowed missing values in lag pattern before dropping | 0 or higher                                                    | feature engineering |

## ⚙️ Config: Binning

| Parameter            | Type                  | Default                      | Description                                           | Valid Values / Notes                                | Used In                       |
|----------------------|-----------------------|------------------------------|-------------------------------------------------------|-----------------------------------------------------|-------------------------------|
| bin_cols             | str                   | 'all'                        | Columns to bin or 'all' for auto-selection            | 'all' or list of column names                       | dataframe encoding            |
| bin_quantiles        | List[float]           | [0, 0.25, 0.5, 0.75, 1.0]    | Quantile edges used for binning                      | Must start at 0 and end at 1                        | dataframe encoding            |
| bin_quantile_labels  | List[str] or None     | []                           | Custom labels for binned categories                  | Must match len(bin_quantiles) - 1 or be empty       | dataframe encoding            |
| bin_grouping         | str                   | 'none'                       | Grouping logic used during binning                   | "none", "ids", "datetime", "datetime+ids"           | dataframe encoding            |
| bin_dt_units         | int                   | 4                            | Grouping window size for time-based binning          | Integer > 0                                         | dataframe encoding            |
| to_sweep             | bool                  | True                         | Merge sparse bins into reserved fallback label       | True or False                                      | dataframe encoding            |
| to_drop_no_data      | bool                  | True                         | Drop no_data pattern columns after encoding          | True or False                                      | dataframe encoding            |
| min_bin_obs          | int                   | 10                           | Min count before a bin is considered valid           | Integer ≥ 1                                         | dataframe encoding            |
| min_bin_fraction     | float                 | 0.01                         | Min proportion before a bin is kept                  | 0.0–1.0                                             | dataframe encoding            |

## ⚙️ Config: Target Calculation

| Parameter           | Type   | Default         | Description                                             | Valid Values / Notes                              | Used In             |
|---------------------|--------|-----------------|---------------------------------------------------------|---------------------------------------------------|---------------------|
| to_calculate_target | bool   | True            | Whether to compute forward return target                | True or False                                     | target calculation  |
| target_periods      | int    | 20              | Lookahead window size for return calculation            | Integer ≥ 1                                       | target calculation  |
| price_col           | str    | 'adj_close'     | Price column used to compute returns                    | Must exist in input DataFrame                     | target calculation  |
| return_mode         | str    | 'pct_change'    | Method to calculate returns                             | "pct_change", "log_return", "vol_adjusted", "smoothed" | target calculation  |
| vol_window          | int    | 20              | Window for volatility adjustment                        | Only used if return_mode == 'vol_adjusted'        | target calculation  |
| smoothing_method    | str    | 'median'        | Smoothing function for return calculation               | "median", "mean", "max", "min"                    | target calculation  |
| target_col          | str    | 'forward_return'| Output column name for computed target                  | Custom string                                     | target calculation  |

## ⚙️ Config: Target Binning

| Parameter              | Type        | Default                     | Description                                             | Valid Values / Notes                                     | Used In             |
|------------------------|-------------|-----------------------------|---------------------------------------------------------|----------------------------------------------------------|---------------------|
| target_binning_method  | str         | 'custom'                    | Method for binning the target column                    | "quantile", "custom", "binary"                           | target calculation  |
| target_bins            | List[float] | [-.inf, -0.05, 0.05, .inf]  | Thresholds to define bin edges                          | Varies by method: custom → numeric, quantile → 0–1       | target calculation  |
| target_labels          | List[str]   | ["down", "sideways", "up"] | Labels for the resulting binned categories              | Must match len(target_bins) - 1                          | target calculation  |
| target_grouping        | str         | "datetime"                  | How to group data before binning                        | "none", "ids", "datetime", "datetime+ids"                | target calculation  |
| target_n_dt            | int         | 20                          | Time window size if grouping by datetime                | Integer ≥ 1                                               | target calculation  |
| target_nan_placeholder | str         | "no_data"                   | Fill value for NaNs post-binning                        | Any string; used in downstream encoding                  | target calculation  |

## ⚙️ Config: Statistics Mask

| Parameter                   | Type        | Default | Description                                             | Valid Values / Notes                           | Used In              |
|-----------------------------|-------------|---------|---------------------------------------------------------|------------------------------------------------|----------------------|
| stat_min_support            | float       | 0.01    | Min support for full rule (fraction of total)           | 0.0–1.0                                        | statistics calculator |
| stat_min_observations       | int         | 20      | Min absolute count matching the rule                    | Integer ≥ 0                                     | statistics calculator |
| stat_bounds_lift            | List[float] | [0.9, 1.1] | Acceptable range for lift value                         | [lower, upper]                                 | statistics calculator |
| stat_min_antecedent_support | float       | -1      | Min support of rule antecedent                          | -1 disables check                               | statistics calculator |
| stat_min_consequent_support | float       | -1      | Min support of rule consequent                          | -1 disables check                               | statistics calculator |
| stat_min_confidence         | float       | -1      | Min conditional probability of consequent given antecedent | -1 disables check                            | statistics calculator |
| stat_min_representativity   | float       | -1      | Min segment coverage (representativity score)           | -1 disables check                               | statistics calculator |
| stat_min_leverage           | float       | -1      | Min difference from expected co-occurrence              | -1 disables check                               | statistics calculator |
| stat_min_conviction         | float       | -1      | Min directional implication strength                    | -1 disables check                               | statistics calculator |
| stat_min_zhangs_metric      | float       | -1      | Min normalized directional association score            | -1 disables check                               | statistics calculator |
| stat_min_jaccard            | float       | -1      | Min overlap ratio between A and B                       | -1 disables check                               | statistics calculator |
| stat_min_certainty          | float       | -1      | Min certainty (confidence scaled by class balance)      | -1 disables check                               | statistics calculator |
| stat_min_kulczynski         | float       | -1      | Min average of P(A|B) and P(B|A)                        | -1 disables check                               | statistics calculator |

## ⚙️ Config: Synthetic Data

| Parameter        | Type   | Default | Description                                           | Valid Values / Notes                              | Used In                  |
|------------------|--------|---------|-------------------------------------------------------|---------------------------------------------------|--------------------------|
| synth_silence     | bool   | True    | Suppress logs and warnings during generation          | True or False                                     | synthetic data generation |
| corrupt_data      | bool   | False   | Whether to apply corruption to input data             | True or False                                     | synthetic data generation |
| corrupt_target    | str    | "none"  | Augment target labels with noise or corruption        | "none", "real", "synthetic", "both"               | synthetic data generation |

## ⚙️ Config: SDV

| Parameter     | Type   | Default           | Description                                      | Valid Values / Notes                                 | Used In                  |
|---------------|--------|-------------------|--------------------------------------------------|------------------------------------------------------|--------------------------|
| to_sdv        | bool   | False             | Whether to generate synthetic data with SDV      | True or False                                        | synthetic data generation |
| sdv_rows      | int    | 500               | Number of synthetic samples to generate          | Integer ≥ 1                                          | synthetic data generation |
| sdv_model     | str    | 'gaussian_copula' | SDV model to use for generation                  | "gaussian_copula", "ctgan", "tvae"                   | synthetic data generation |
| sdv_verbose   | bool   | False             | Whether to print SDV quality score               | True or False                                        | synthetic data generation |

## ⚙️ Config: Miners

| Parameter | Type       | Default                          | Description                                     | Valid Values / Notes                                         | Used In     |
|-----------|------------|----------------------------------|-------------------------------------------------|--------------------------------------------------------------|-------------|
| miners    | List[str]  | ["univar", "rulefit", "subgroup"]| List of rule mining algorithms to run           | "apriori", "rulefit", "subgroup", "elcs", "cn2", "cart"       | rule mining |

## ⚙️ Config: Apriori

| Parameter            | Type   | Default | Description                                      | Valid Values / Notes                        | Used In     |
|----------------------|--------|---------|--------------------------------------------------|---------------------------------------------|-------------|
| apriori_min_support  | float  | 0.01    | Minimum support threshold for rule generation    | 0.0–1.0                                     | rule mining |
| apriori_metric       | str    | "lift"  | Metric used to evaluate Apriori rules            | "lift", "confidence", "leverage", etc.      | rule mining |
| apriori_min_metric   | float  | 0.0     | Minimum value for selected Apriori metric        | Depends on chosen metric                    | rule mining |

## ⚙️ Config: Rulefit, Subgroup, CART

| Parameter              | Type   | Default | Description                                         | Valid Values / Notes                         | Used In       |
|------------------------|--------|---------|-----------------------------------------------------|----------------------------------------------|----------------|
| rulefit_tree_size      | int    | 3       | Max tree depth when extracting rules from RuleFit   | Integer ≥ 1                                   | rule mining    |
| rulefit_min_depth      | int    | 2       | Min rule depth to include from RuleFit              | Integer ≥ 1                                   | rule mining    |
| subgroup_top_n         | int    | 50      | Number of top subgroups to return                   | Integer ≥ 1                                   | rule mining    |
| subgroup_depth         | int    | 3       | Max depth for Subgroup Discovery rules              | Integer ≥ 1                                   | rule mining    |
| subgroup_beam_width    | int    | 50      | Beam search width for Subgroup Discovery            | Integer ≥ 1                                   | rule mining    |
| cart_max_depth         | int    | 5       | Max depth of the CART decision tree                 | Integer ≥ 1 or None for unlimited             | rule mining    |
| cart_criterion         | str    | "gini"  | Splitting criterion used in CART                    | "gini", "entropy"                             | rule mining    |
| cart_random_state      | int    | 42      | Random seed for CART reproducibility                | Any integer                                   | rule mining    |
| cart_min_samples_split | int    | 2       | Min samples needed to split a node in CART          | Integer ≥ 2                                   | rule mining    |
| cart_min_samples_leaf  | int    | 1       | Min samples required at each CART leaf node         | Integer ≥ 1                                   | rule mining    |

## ⚙️ Config: Train Test

| Parameter               | Type           | Default                                    | Description                                                 | Valid Values / Notes                                       | Used In         |
|-------------------------|----------------|--------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------|------------------|
| perform_train_test      | bool           | True                                       | Whether to perform a train/test split evaluation            | True or False                                               | validation tests |
| train_test_split_method | str            | "fractional"                               | Strategy for splitting the dataset                          | "temporal", "fractional"                                    | validation tests |
| train_test_splits       | int            | 2                                          | Number of train/test splits to create                       | Integer ≥ 1                                                 | validation tests |
| train_test_ranges       | List[List[str]]| [["2010-01-01", "2015-01-01"], ["2015-01-01", "2020-01-01"]] | Manual start/end overrides per split                        | List of pairs or None                                       | validation tests |
| train_test_window_frac  | float          | 0.6                                        | Fraction of data used per training window                   | 0.0–1.0                                                     | validation tests |
| train_test_step_frac    | float          | 0.0                                        | Step size between split windows as a data fraction          | 0.0–1.0                                                     | validation tests |
| train_test_fractions    | List[float]    | [0.6, 0.4]                                 | Fractional allocation of train/test within each window      | Must sum to 1.0                                             | validation tests |
| train_test_overlap      | bool           | False                                      | Whether train/test windows can overlap                      | True or False                                               | validation tests |
| train_test_re_mine      | bool           | False                                      | Re-mine rules per split or reuse globally                   | True or False                                               | validation tests |

## ⚙️ Config: WFA (Walk Forward Analysis)

| Parameter         | Type             | Default                                    | Description                                               | Valid Values / Notes                                       | Used In         |
|-------------------|------------------|--------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|------------------|
| perform_wfa       | bool             | True                                       | Whether to run Walk Forward Analysis                      | True or False                                               | validation tests |
| wfa_split_method  | str              | "temporal"                                 | Strategy for splitting dataset                            | "temporal", "fractional"                                    | validation tests |
| wfa_splits        | int              | 4                                          | Number of WFA splits to generate                          | Integer ≥ 1                                                 | validation tests |
| wfa_ranges        | List[List[str]]  | [["2010-01-01", "2017-01-01"], ...]        | Optional manual date range overrides per split            | List of [start, end] pairs or None                          | validation tests |
| wfa_window_frac   | float            | 0.4                                        | Training window size as a fraction of total data          | 0.0–1.0                                                     | validation tests |
| wfa_step_frac     | float            | 0.2                                        | Step size between windows as fraction of data             | 0.0–1.0                                                     | validation tests |
| wfa_fractions     | List[float]      | [0.25, 0.25, 0.25, 0.25]                   | Relative train/test fractions per split                   | Must sum to 1.0                                             | validation tests |
| wfa_overlap       | bool             | True                                       | Whether train/test windows can chronologically overlap     | True or False                                               | validation tests |
| wfa_re_mine       | bool             | True                                       | Re-mine rules on each split or use static rule set        | True or False                                               | validation tests |

## ⚙️ Config: Bootstrap

| Parameter           | Type   | Default     | Description                                          | Valid Values / Notes                          | Used In           |
|---------------------|--------|-------------|------------------------------------------------------|------------------------------------------------|-------------------|
| perform_bootstrap   | bool   | True        | Whether to run bootstrap resampling validation       | True or False                                  | validation tests  |
| n_bootstrap         | int    | 1000        | Number of bootstrap iterations to perform            | Integer ≥ 1                                    | validation tests  |
| bootstrap_verbose   | bool   | True        | Show progress bar during bootstrap                   | True or False                                  | validation tests  |
| resample_method     | str    | 'traditional'| Resampling strategy used for bootstrapping           | "traditional", "block", "block_ids"            | validation tests  |
| block_size          | int    | 4           | Number of rows per bootstrap block (if block method) | Integer ≥ 1                                    | validation tests  |

## ⚙️ Config: Null Distribution

| Parameter             | Type    | Default     | Description                                                  | Valid Values / Notes                            | Used In           |
|-----------------------|---------|-------------|--------------------------------------------------------------|-------------------------------------------------|-------------------|
| perform_null_fdr      | bool    | True        | Run null distribution test and apply FDR correction          | True or False                                   | validation tests  |
| shuffle_mode          | str     | "target"    | How to shuffle data for null generation                      | "target", "rows", "columns"                     | validation tests  |
| n_null                | int     | 1000        | Max number of null permutations to generate                  | Integer ≥ 1                                     | validation tests  |
| null_verbose          | bool    | True        | Show progress bar during null test generation                | True or False                                   | validation tests  |
| early_stop_metric     | str     | "lift"      | Metric used for early stopping relative error check          | Any rule metric (e.g. lift, confidence)         | validation tests  |
| es_m_permutations     | int     | 50          | Window size for evaluating relative error stability          | Integer ≥ 1                                     | validation tests  |
| rel_error_threshold   | float   | 0.01        | Relative error threshold for early stopping                  | 0.0–1.0                                         | validation tests  |

## ⚙️ Config: Multiple Corrections

| Parameter           | Type   | Default     | Description                                      | Valid Values / Notes                       | Used In           |
|---------------------|--------|-------------|--------------------------------------------------|--------------------------------------------|-------------------|
| correction_metric   | str    | 'fdr_bh'     | Method for multiple hypothesis correction        | "fdr_bh", "fdr_by"                          | validation tests  |
| correction_alpha    | float  | 0.05         | P-value threshold for statistical significance   | 0.0–1.0                                    | validation tests  |
| fdr_mode            | str    | "two-sided"  | Directionality of FDR hypothesis test            | "greater", "less", "two-sided"             | validation tests  |

