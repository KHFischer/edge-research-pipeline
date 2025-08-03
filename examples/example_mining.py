"""
Example: Rule Mining Module Usage
Location: examples/mining_example.py

Demonstrates usage of the rule mining pipeline, including:
- prepare_dataframe_for_mining
- generate_combined_synthetic_data
- augment_dataset
- data_prep_pipeline
- mine_stats
- mining_pipeline

See: docs/mining.md for full param docs and details.
"""

from pathlib import Path
from edge_research.rules_mining.mining import (
    prepare_dataframe_for_mining,
    generate_combined_synthetic_data,
    augment_dataset,
    data_prep_pipeline,
    mining_pipeline,
    mine_stats
)
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from params.config_validator import load_params, Config
from edge_research.preprocessing.cleaning import clean_pipeline
from edge_research.preprocessing.engineering import engineer_pipeline
from edge_research.preprocessing.target import target_pipeline

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

# --- Preprocessing steps (required for mining) ---
df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)
df_engineered, logs = engineer_pipeline(df_cleaned, cfg, logger)
df_target, target_logs = target_pipeline(df_engineered, cfg, hloc_df, logger)

# --- 1. Prepare DataFrame for Mining ---

# Config-driven example
real_df, prep_log = prepare_dataframe_for_mining(
    df_target,
    date_col=cfg.date_col,
    id_cols=cfg.id_cols,
    drop_cols=cfg.drop_cols,
    target_col=cfg.target_col,
    to_sample=cfg.to_sample,
    sample_size=cfg.sample_size,
    drop_duplicates=cfg.drop_duplicates
)

# Minimal example
real_df, prep_log = prepare_dataframe_for_mining(
    df_target,
    date_col='date',
    id_cols=['ticker'],
    drop_cols=['location'],
    target_col='forward_return'
)
print("Prepared real_df shape:", real_df.shape)

# --- 2. Create Synthetic Data ---

# Config-driven
synth_df, synth_logs = generate_combined_synthetic_data(
    df=real_df,
    target_col=cfg.target_col,
    to_sdv=cfg.to_sdv,
    to_synthcity=cfg.to_synthcity,
    sdv_model=cfg.sdv_model,
    sdv_rows=cfg.sdv_rows,
    sdv_verbose=cfg.sdv_verbose,
    sc_model=cfg.sc_model,
    sc_rows=cfg.sc_rows,
    sc_n_iter=cfg.sc_n_iter,
    sc_batch_size=cfg.sc_batch_size,
    sc_lr=cfg.sc_lr,
    sc_device=cfg.sc_device,
    silence=cfg.synth_silence
)

# Minimal
synth_df, synth_logs = generate_combined_synthetic_data(
    df=real_df,
    to_sdv=True,
    to_synthcity=False,
    sdv_model='gaussian_copula',
    sdv_rows=500
)
print("Synthetic df shape:", synth_df.shape)

# --- 3. Data Augmentation ---

# Config-driven
augmented_real_df = augment_dataset(
    df=real_df,
    target_col=cfg.target_col,
    to_aug_imbalance=cfg.to_aug_imbalance,
    to_aug_flip_feats=cfg.to_aug_flip_feats,
    to_aug_flip_targets=cfg.to_aug_flip_targets,
    flip_feats_frac=cfg.flip_feats_frac,
    flip_targs_frac=cfg.flip_targs_frac
)

# Minimal
augmented_real_df = augment_dataset(
    df=real_df,
    target_col='forward_return',
    to_aug_imbalance=False,
    to_aug_flip_feats=False,
    to_aug_flip_targets=False
)

print("Augmented real_df shape:", augmented_real_df.shape)

# --- 4. Data Prep Pipeline (one-hot and ready for mining) ---

df_onehot, prep_logs = data_prep_pipeline(df_target, cfg, logger)
print("df_onehot shape:", df_onehot.shape)

# Overrides
df_onehot, prep_logs = data_prep_pipeline(df_target, cfg, logger, to_aug_imbalance=False)

# --- 5. Perform Rule Mining ---

# Config-driven (full verbose kwargs)
final_stats_df, logs, rules_df = mine_stats(
    df=df_onehot, cfg=cfg,
    target_col=cfg.target_col,
    miners=cfg.miners,
    apriori_min_support=cfg.apriori_min_support,
    apriori_metric=cfg.apriori_metric,
    apriori_min_metric=cfg.apriori_min_metric,
    rulefit_tree_size=cfg.rulefit_tree_size,
    rulefit_min_depth=cfg.rulefit_min_depth,
    subgroup_top_n=cfg.subgroup_top_n,
    subgroup_depth=cfg.subgroup_depth,
    subgroup_beam_width=cfg.subgroup_beam_width,
    cart_max_depth=cfg.cart_max_depth,
    cart_criterion=cfg.cart_criterion,
    cart_random_state=cfg.cart_random_state,
    cart_min_samples_split=cfg.cart_min_samples_split,
    cart_min_samples_leaf=cfg.cart_min_samples_leaf
)

# Minimal example
final_stats_df, logs, rules_df = mine_stats(
    df=df_onehot, cfg=cfg,
    target_col='forward_return',
    miners=['univar']
)
print("Rules shape (minimal):", rules_df.shape)

# --- 6. Full Mining Pipeline ---

final_stats_df, rules_df, mining_logs = mining_pipeline(df_onehot, cfg, logger)
print("Final stats df shape:", final_stats_df.shape)

# With overrides
final_stats_df, rules_df, mining_logs = mining_pipeline(df_onehot, cfg, logger, miners=['univar'])

# --- Reference ---

# All steps produce logs for traceability. See docs/mining.md for parameter explanations and advanced usage.
