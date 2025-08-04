"""
Example: Feature Engineering Module Usage
Location: examples/engineering_example.py

Demonstrates usage of the core engineering functions:
- engineer_features
- encode_data
- engineer_pipeline

See: docs/engineering.md for detailed parameter docs and output descriptions.
"""

from pathlib import Path
from edge_research.preprocessing.engineering import (
    engineer_features,
    encode_data,
    engineer_pipeline
)
import edge_research
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from edge_research.params.config_validator import load_params, Config
from edge_research.preprocessing.cleaning import clean_pipeline  # cleaning assumed done first

# --- SETUP ---

# Load sample feature DataFrame (replace with your own as needed)
_, feature_df = load_samples()

# Load config (YAML or dict supported)
# Resolve params directory inside installed package
params_dir = Path(edge_research.__path__[0]) / "params"

# Load configuration files
default_params = params_dir / "default_params.yaml"
custom_params = params_dir / "custom_params.yaml"

params = load_params(str(default_params), str(custom_params))
cfg = Config(**params)

# Setup results/logging
run_name = cfg.run_name
save_folder = Path("data/results") / run_name
save_folder.mkdir(parents=True, exist_ok=True)
logger = PipelineLogger(log_path=save_folder / f"{run_name}_log.md")

# --- Cleaning step (required before engineering) ---
df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)

# --- 1. Feature Engineering ---

# Minimal example: only base columns, no derived features
df, engineer_features_logs = engineer_features(
    df=df_cleaned,
    date_col='date',
    id_cols=['ticker'],
    engineer_cols='base',
    to_engineer_dates=False,
    to_engineer_ratios=False,
    to_engineer_lags=False
)
print("Features after minimal engineering:", df.columns.tolist())

# Full config-driven example
df, engineer_features_logs = engineer_features(
    df=df_cleaned,
    date_col=cfg.date_col,
    id_cols=cfg.id_cols,
    engineer_cols=cfg.engineer_cols,
    to_engineer_dates=cfg.to_engineer_dates,
    to_engineer_ratios=cfg.to_engineer_ratios,
    to_engineer_lags=cfg.to_engineer_lags,
    lag_mode=cfg.lag_mode,
    n_dt_list=cfg.n_dt_list,
    flat_threshold=cfg.flat_threshold
)
print("Features after full engineering:", df.columns.tolist())

# --- 2. Data Encoding ---

# Minimal example: no encoding, no sweeping, just id/date columns
df, encode_data_logs = encode_data(
    df=df_cleaned,
    id_cols=['ticker'],
    date_col='date',
    to_sweep=False,
    to_drop_no_data=False
)
print("Columns after minimal encoding:", df.columns.tolist())

# Full config-driven example
df, encode_data_logs = encode_data(
    df=df,
    id_cols=cfg.id_cols,
    date_col=cfg.date_col,
    drop_cols=cfg.drop_cols,
    bin_cols=cfg.bin_cols,
    bin_quantiles=cfg.bin_quantiles,
    bin_quantile_labels=cfg.bin_quantile_labels,
    bin_grouping=cfg.bin_grouping,
    bin_dt_units=cfg.bin_dt_units,
    to_sweep=cfg.to_sweep,
    to_drop_no_data=cfg.to_drop_no_data,
    min_bin_obs=cfg.min_bin_obs,
    min_bin_fraction=cfg.min_bin_fraction,
    lag_num_missing=cfg.lag_num_missing
)
print("Columns after full encoding:", df.columns.tolist())

# --- 3. Pipeline: All Steps At Once ---

df_engineered, logs = engineer_pipeline(df_cleaned, cfg, logger)
print("Shape after full engineering pipeline:", df_engineered.shape)

# Optionally overwrite any arg manually
df_engineered, logs = engineer_pipeline(
    df_cleaned, cfg, logger,
    to_sweep=False,
    lag_num_missing=5
)
print("Shape with manual pipeline override:", df_engineered.shape)

# --- Reference ---

# Each step produces logs for auditability.
# For details on available parameters, see: docs/engineering.md or config.
