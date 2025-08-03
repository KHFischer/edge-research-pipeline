"""
Example: Target Creation Module Usage
Location: examples/target_example.py

Demonstrates usage of the core target creation functions:
- compute_forward_return
- merge_features_with_returns
- bin_target_column
- target_pipeline

See: docs/target.md for parameter docs and output descriptions.
"""

from pathlib import Path
import numpy as np
from edge_research.preprocessing.target import (
    compute_forward_return,
    merge_features_with_returns,
    bin_target_column,
    target_pipeline
)
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from params.config_validator import load_params, Config
from edge_research.preprocessing.cleaning import clean_pipeline
from edge_research.preprocessing.engineering import engineer_pipeline

# --- SETUP ---

# Load sample feature and price data
hloc_df, feature_df = load_samples()

# Load config (YAML or dict supported)
params = load_params("params/default_params.yaml", "params/custom_params.yaml")
cfg = Config(**params)

# Setup results/logging
run_name = cfg.run_name
save_folder = Path("data/results") / run_name
save_folder.mkdir(parents=True, exist_ok=True)
logger = PipelineLogger(log_path=save_folder / f"{run_name}_log.md")

# --- Cleaning & Engineering steps (recommended before target creation) ---
df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)
df_engineered, logs = engineer_pipeline(df_cleaned, cfg, logger)

# --- 1. Calculate Forward Return from HLOC ---

# Minimal example: default period, column names
target = compute_forward_return(
    price_df=hloc_df,
    id_cols=['ticker'],
    date_col='date',
    price_col='adj_close',
    target_col='forward_return'
)
print("Target (minimal) head:\n", target.head())

# Full config-driven example
target = compute_forward_return(
    price_df=hloc_df,
    n_periods=cfg.target_periods,
    id_cols=cfg.id_cols,
    date_col=cfg.date_col,
    target_col=cfg.target_col,
    price_col=cfg.price_col,
    return_mode=cfg.return_mode,
    vol_window=cfg.vol_window,
    smoothing_method=cfg.smoothing_method
)
print("Target (config-driven) head:\n", target.head())

# --- 2. Forward Merge Target with Features ---

df = merge_features_with_returns(
    feature_df=df_engineered,
    returns_df=target,
    id_cols=cfg.id_cols,
    feature_date_col=cfg.date_col,
    returns_date_col=cfg.date_col
)
print("Shape after merge:", df.shape)

# --- 3. Bin Target Column ---

# Minimal example: custom bins/labels
df_target, target_bin_log = bin_target_column(
    df,
    id_cols=['ticker'],
    date_col='date',
    binning_method='custom',
    bins=[-np.inf, -0.05, 0.05, np.inf],
    labels=['down', 'sideways', 'up'],
    target_col='forward_return'
)
print("Binned target value counts:\n", df_target['forward_return'].value_counts())

# Full config-driven example
df_target, target_bin_log = bin_target_column(
    df,
    binning_method=cfg.target_binning_method,
    bins=cfg.target_bins,
    labels=cfg.target_labels,
    target_col=cfg.target_col,
    id_cols=cfg.id_cols,
    date_col=cfg.date_col,
    grouping=cfg.target_grouping,
    n_datetime_units=cfg.target_n_dt,
    nan_placeholder=cfg.target_nan_placeholder
)

# --- 4. Pipeline: All Steps At Once ---

df_target, target_logs = target_pipeline(
    df_engineered, cfg, hloc_df, logger
)
print("Final target shape (pipeline):", df_target.shape)

# Optionally override specific params at call time
df_target, target_logs = target_pipeline(
    df_engineered, cfg, hloc_df, logger,
    binning_method='custom'
)
print("Final target shape (custom binning):", df_target.shape)

# --- Reference ---

# Each function/log provides full auditability.
# For parameter reference and log details, see: docs/target.md or the config dataclass.
