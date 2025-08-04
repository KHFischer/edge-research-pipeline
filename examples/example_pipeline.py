# 📊 Full Pipeline Execution — Single Run vs. Grid Mode

# 💡 This example shows how to run the full rule mining pipeline either for a single parameter configuration
# using inline or YAML-based settings, or for a parameter grid defined in a YAML file.

# === 1. Imports ===
import edge_research
from edge_research.pipeline.pipeline import edge_research_pipeline, grid_edge_research_pipeline
from pathlib import Path

# Resolve params directory inside installed package
params_dir = Path(edge_research.__path__[0]) / "params"
data_path = Path(edge_research.__path__[0]) / "data"

# Load configuration files
default_params = params_dir / "default_params.yaml"
custom_params = params_dir / "custom_params.yaml"
grid_params = params_dir / "grid_params.yaml"

# === 2. Single-Run Pipeline with Custom Parameters (inline dict) ===
results, logs = edge_research_pipeline(
    to_train_test=True,
    to_wfa=True,
    to_bootstrap=True,
    to_null_fdr=True,
    default_params=str(default_params),
    custom_params={"target_n_dt": 60},  # Can also be a path to YAML
    feature_path=data_path/"fundamentals_sample.parquet",
    hloc_path=data_path/"data/hloc_sample.parquet",
    res_save_path="data/results/",
    res_filetype="csv",
    verbose=True,
)

# === 3. Grid-Based Pipeline Run (YAML config with param_space and n_jobs) ===
results_list = grid_edge_research_pipeline(str(grid_params))

# === 4. CLI Equivalent ===
# python edge_research_notebook/scripts/pipeline/main.py params/grid_params.yaml

# === 5. Output Inspection ===
print(f"Single run result keys: {list(results.keys())}")
print(f"Grid run completed: {len(results_list)} total runs")

# See docs/pipeline.md for full config options and behavior notes.
