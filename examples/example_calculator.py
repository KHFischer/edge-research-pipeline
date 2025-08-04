"""
Example: Statistics Calculator Module Usage
Location: examples/statistics_example.py

Demonstrates usage of the statistics calculator:
- generate_statistics

See: docs/statistics.md for parameter documentation and output details.
"""

from pathlib import Path
import edge_research
from edge_research.statistics.calculator import generate_statistics
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from edge_research.params.config_validator import load_params, Config
from edge_research.preprocessing.cleaning import clean_pipeline
from edge_research.preprocessing.engineering import engineer_pipeline
from edge_research.preprocessing.target import target_pipeline
from edge_research.rules_mining.mining import data_prep_pipeline

# --- SETUP ---

# Load sample feature and price data
hloc_df, feature_df = load_samples()

# Load config (YAML or dict supported)
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

# --- Preprocessing steps (calculator requires one-hot dataframe) ---

# 1. Cleaning
df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)
# 2. Feature engineering
df_engineered, logs = engineer_pipeline(df_cleaned, cfg, logger)
# 3. Target creation
df_target, target_logs = target_pipeline(df_engineered, cfg, hloc_df, logger)
# 4. Data prep (e.g. one-hot encoding)
df_onehot, prep_logs = data_prep_pipeline(df_target, cfg, logger)

# --- 1. Basic Statistics Generation ---

stats_df, stats_log = generate_statistics(df_onehot, cfg)
print("Stats shape:", stats_df.shape)
print("Example stats columns:", stats_df.columns.tolist())

# --- 2. Statistics Generation with Overrides ---

stats_df, stats_log = generate_statistics(
    df_onehot,
    cfg,
    overrides={"stat_min_support": 5, "stat_min_lift": 1.2}
)
print("Stats with overrides shape:", stats_df.shape)

# --- Reference ---

# For all parameter options and log structure, see: docs/calculator.md or config.
