"""
Example: Cleaning Module Usage
Location: examples/cleaning_example.py

Demonstrates usage of the core cleaning functions:
- apply_column_type_cleaning
- handle_missing_data
- handle_outliers_and_redundancy
- normalize_features
- clean_pipeline

See: docs/cleaning.md for detailed param docs and output descriptions.
"""

from pathlib import Path
from edge_research.preprocessing.cleaning import (
    apply_column_type_cleaning,
    handle_missing_data,
    handle_outliers_and_redundancy,
    normalize_features,
    clean_pipeline
)
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from params.config_validator import load_params, Config

# --- SETUP ---

# Load sample feature DataFrame (replace with your own as needed)
_, feature_df = load_samples()

# Load config (YAML or dict supported)
params = load_params("params/default_params.yaml", "params/custom_params.yaml")
cfg = Config(**params)

# Setup results/logging
run_name = "Example_Run"
save_folder = Path("data/results") / run_name
save_folder.mkdir(parents=True, exist_ok=True)
logger = PipelineLogger(log_path=save_folder / f"{run_name}_log.md")

# --- 1. Column Type Cleaning ---

# Enforce and audit dtypes (numeric, datetime, category, etc.)
df, cleaning_log = apply_column_type_cleaning(feature_df)
print("After column type cleaning:", df.dtypes.head(), "\n")

# --- 2. Missing Data Handling ---

# Minimal example: just drop nothing, just impute
df, missing_log = handle_missing_data(
    df=df,
    id_cols=['ticker'],
    run_drop_high_missingness=False,
    run_impute_numeric=True,
    run_impute_categorical=True
)
print("Missing data handled (minimal):", df.isna().sum().head(), "\n")

# Full config-driven example
df, missing_log = handle_missing_data(df=df, **{
    "id_cols": cfg.id_cols,
    "run_drop_high_missingness": cfg.to_drop_high_missingness,
    "row_thresh": cfg.missingness_row_thresh,
    "col_thresh": cfg.missingness_col_thresh,
    "run_impute_numeric": cfg.to_impute_numeric,
    "run_impute_categorical": cfg.to_impute_categorical,
    "run_mask_high_imputation": cfg.to_mask_high_impute,
    "impute_strategy": cfg.impute_strategy,
    "max_imputed": cfg.max_imputed
})

# --- 3. Outlier & Redundancy Handling ---

# Manual override example: only drop high-correlation columns
df, outliers_log = handle_outliers_and_redundancy(df=df, **{
    'id_cols': ['ticker'],
    'date_col': 'date',
    'to_drop_var': False,
    'to_drop_corr': True,
    'correlation_threshold': 0.9
})

# Full config-driven example
df, outliers_log = handle_outliers_and_redundancy(df=df, **{
    "id_cols": cfg.id_cols,
    "date_col": cfg.date_col,
    "to_winsorize": cfg.to_winsorize,
    "to_drop_var": cfg.to_drop_var,
    "to_drop_corr": cfg.to_drop_corr,
    "winsor_cols": cfg.winsor_cols,
    "winsor_grouping": cfg.winsor_grouping,
    "winsor_dt_units": cfg.winsor_dt_units,
    "winsor_lower_quantile": cfg.winsor_lower_quantile,
    "winsor_upper_quantile": cfg.winsor_upper_quantile,
    "variance_threshold": cfg.variance_threshold,
    "correlation_threshold": cfg.correlation_threshold
})

# --- 4. Feature Normalization ---

# Minimal example: z-score scaling, no grouping
df, scale_log = normalize_features(
    df=df,
    id_cols=['ticker'],
    date_col='date',
    scaling_method='zscore'
)

# Full config-driven example
df, scale_log = normalize_features(df=df, **{
    "id_cols": cfg.id_cols,
    "date_col": cfg.date_col,
    "scaling_method": cfg.scaling_method,
    "grouping": cfg.scale_grouping,
    "n_datetime_units": cfg.scale_dt_units,
    "quantile_range": cfg.robust_scale_quantile_range,
    "mode": cfg.quantile_rank_mode,
    "norm_type": cfg.vector_scale_norm_type
})

# --- 5. Pipeline: All Steps At Once ---

# Run the full cleaning pipeline (all four steps above)
df_cleaned, logs = clean_pipeline(feature_df, cfg, logger)
print("Pipeline output shape:", df_cleaned.shape)
# Optionally override specific params at call time
df_cleaned, logs = clean_pipeline(feature_df, cfg, logger, row_thresh=0.95, norm_type="l2")

# --- Outputs ---

# At each step, logs (dicts or DataFrames) record exactly what changed for auditability
# For full details, consult: docs/cleaning.md or see the 'logs' variable at each step