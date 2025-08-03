"""
Example: Validation Testing Module Usage
Location: examples/validation_example.py

Demonstrates usage of core validation functions:
- validate_train_test, train_test_pipeline
- validate_wfa, wfa_pipeline
- validate_bootstrap, bootstrap_pipeline
- validate_null, null_pipeline
- validate_multiple_tests, fdr_pipeline

See: docs/validation.md for parameter and log details.
"""

from pathlib import Path
from edge_research.validation_tests.validation import (
    validate_train_test,
    train_test_pipeline,
    validate_wfa,
    wfa_pipeline,
    validate_bootstrap,
    bootstrap_pipeline,
    validate_null,
    null_pipeline,
    validate_multiple_tests,
    fdr_pipeline
)
from edge_research.rules_mining.mining import data_prep_pipeline
from edge_research.utils.utils import load_samples
from edge_research.logger.logger import PipelineLogger
from params.config_validator import load_params, Config
from edge_research.preprocessing.cleaning import clean_pipeline
from edge_research.preprocessing.engineering import engineer_pipeline
from edge_research.preprocessing.target import target_pipeline

# --- SETUP ---

hloc_df, feature_df = load_samples()
params = load_params("params/default_params.yaml", "params/custom_params.yaml")
cfg = Config(**params)

run_name = cfg.run_name
save_folder = Path("data/results") / run_name
save_folder.mkdir(parents=True, exist_ok=True)
logger = PipelineLogger(log_path=save_folder / f"{run_name}_log.md")

# --- Preprocessing (required for validation) ---
df_cleaned, _ = clean_pipeline(feature_df, cfg, logger)
df_engineered, _ = engineer_pipeline(df_cleaned, cfg, logger)
df_target, _ = target_pipeline(df_engineered, cfg, hloc_df, logger)
df_onehot, _ = data_prep_pipeline(df_target, cfg, logger)

# --- 1. Train/Test Validation ---

# Verbose args example
train_test_results, pipeline_logs = validate_train_test(
    df=feature_df,
    hloc=hloc_df,
    cfg=cfg,
    logger=logger,
    date_col=cfg.date_col,
    target_col=cfg.target_col,
    train_test_splits=cfg.train_test_splits,
    train_test_ranges=cfg.train_test_ranges,
    train_test_split_method=cfg.train_test_split_method,
    train_test_window_frac=cfg.train_test_window_frac,
    train_test_step_frac=cfg.train_test_step_frac,
    train_test_fractions=cfg.train_test_fractions,
    train_test_overlap=cfg.train_test_overlap,
    train_test_re_mine=cfg.train_test_re_mine
)
print("Train/test results shape:", train_test_results.shape)

# Config-driven pipeline
train_test_results, train_test_log, pipeline_logs = train_test_pipeline(feature_df, hloc_df, cfg, logger)
# With overrides
train_test_results, train_test_log, pipeline_logs = train_test_pipeline(
    feature_df, hloc_df, cfg, logger, train_test_re_mine=False
)

# --- 2. Walk Forward Analysis (WFA) ---

# Verbose args example
wfa_results, pipeline_logs = validate_wfa(
    feature_df,
    hloc_df,
    cfg,
    logger=logger,
    date_col=cfg.date_col,
    target_col=cfg.target_col,
    wfa_split_method=cfg.wfa_split_method,
    wfa_splits=cfg.wfa_splits,
    wfa_ranges=cfg.wfa_ranges,
    wfa_window_frac=cfg.wfa_window_frac,
    wfa_step_frac=cfg.wfa_step_frac,
    wfa_fractions=cfg.wfa_fractions,
    wfa_overlap=cfg.wfa_overlap,
    wfa_re_mine=cfg.wfa_re_mine
)
print("WFA results shape:", wfa_results.shape)

# Config-driven pipeline
wfa_results, wfa_log, pipeline_logs = wfa_pipeline(feature_df, hloc_df, cfg, logger)
# With overrides
wfa_results, wfa_log, pipeline_logs = wfa_pipeline(
    feature_df, hloc_df, cfg, logger, wfa_re_mine=True
)

# --- 3. Bootstrap Resampling ---

# Full verbose
bootstrap_results, bootstrap_log = validate_bootstrap(
    df=df_onehot,
    cfg=cfg,
    logger=logger,
    date_col=cfg.date_col,
    id_cols=cfg.id_cols,
    target_col=cfg.target_col,
    n_bootstrap=cfg.n_bootstrap,
    verbose=cfg.bootstrap_verbose,
    resample_method=cfg.resample_method,
    block_size=cfg.block_size
)
print("Bootstrap results shape:", bootstrap_results.shape)

# Config-driven pipeline
bootstrap_results, bootstrap_log = bootstrap_pipeline(df_onehot, cfg, logger)
# With overrides
bootstrap_results, bootstrap_log = bootstrap_pipeline(df_onehot, cfg, logger, verbose=True)

# --- 4. Null Distribution ---

# Full verbose
null_df, null_log = validate_null(
    df=df_onehot,
    cfg=cfg,
    logger=logger,
    target_col=cfg.target_col,
    n_null=cfg.n_null,
    shuffle_mode=cfg.shuffle_mode,
    early_stop_metric=cfg.early_stop_metric,
    es_m_permutations=cfg.es_m_permutations,
    rel_error_threshold=cfg.rel_error_threshold,
    verbose=cfg.null_verbose
)
print("Null distribution shape:", null_df.shape)

# Config-driven pipeline
null_df, null_log = null_pipeline(df_onehot, cfg, logger)
# With overrides
null_df, null_log = null_pipeline(df_onehot, cfg, logger, n_null=1000)

# --- 5. FDR Multiple Testing Correction ---

# FDR requires mining results and a null distribution.
from edge_research.rules_mining.mining import mining_pipeline

mining_res, rules_df, mining_logs = mining_pipeline(df_onehot, cfg, logger)

# Full verbose
fdr_res, fdr_log = validate_multiple_tests(
    mining_res=mining_res,
    null_df=null_df,
    early_stop_metric=cfg.early_stop_metric,
    mode=cfg.fdr_mode,
    correction_alpha=cfg.correction_alpha,
    correction_metric=cfg.correction_metric
)
print("FDR-corrected results shape:", fdr_res.shape)

# Config-driven pipeline
fdr_res, fdr_log = fdr_pipeline(mining_res, null_df, cfg, logger)
# With overrides
fdr_res, fdr_log = fdr_pipeline(mining_res, null_df, cfg, logger, early_stop_metric="leverage")

# --- Reference ---

# Each step produces detailed logs and audit trails.
# See docs/validation.md for all parameter and output details.
