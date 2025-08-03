# 📂 pipeline

## 🧠 What This Module Does
- Runs the full edge research pipeline from a single config or a parameter grid.
- Supports CLI and programmatic execution for batch rule mining and validation.
- Includes optional start/stop logic to skip already-completed runs in grid mode.

## 🧰 Main Features
- **Single-run pipeline** via `edge_research_pipeline()` using config dicts.
- **Grid-based batch runner** via `grid_edge_research_pipeline()` with YAML-driven param sweeps.
- **CLI interface** via `main.py` for command-line execution.
- **Start/stop aware**: skips grid runs if the result folder already exists.
- Config merging logic supports layered overrides: `default → custom → grid`.

## 🚀 How to Use

### 🔧 Option 1: Programmatic Execution (Single Run)
```python
from pipeline import edge_research_pipeline

results, logs = edge_research_pipeline(
    to_train_test=True,
    to_wfa=True,
    to_bootstrap=True,
    to_null_fdr=True,
    default_params="params/default_params.yaml",
    custom_params={"target_n_dt": 60},
    feature_path="data/fundamentals_sample.parquet",
    hloc_path="data/hloc_sample.parquet",
    res_save_path="data/results/",
    res_filetype="csv",
    verbose=True,
)
```

### 🔁 Option 2: Grid Run from YAML

```python
from pipeline import grid_edge_research_pipeline

results_list = grid_edge_research_pipeline("params/grid_params.yaml")
```

### 🖥️ Option 3: CLI Execution

```bash
python edge_research_notebook/scripts/pipeline/main.py params/grid_params.yaml
```

### 📄 Sample YAML Snippet (`grid_params.yaml`)

```yaml
base_run_name: "param_grid"
res_save_path: "data/results/param_grid_1"
n_jobs: 2
param_space:
  target_n_dt: [60, 120]
```

## ⚠️ Design Notes / Caveats

* Grid skipping logic only checks if the result folder exists — does not verify completeness or content integrity.
* All paths in YAML are resolved **relative to the project root**, not to the config file.
* Verbose output only applies when `n_jobs: 1`.
* Config layers apply in order of **default → custom → grid**, with later sources overriding earlier ones.
* Assumes input files exist and are properly formatted; does not validate schema.

## 🧪 Testing Status

* Unit tested for:

  * Path resolution
  * Grid generation
  * Parallel and serial execution
  * Skip logic
* Not yet tested:

  * Output integrity (i.e., whether result folders contain complete/valid files)

## 🔗 Related Modules

* [`cleaning/`](../cleaning): for data preprocessing
* [`mining/`](../mining): contains mining algorithms used in this pipeline
* [`validation/`](../validation): houses statistical validation tests invoked from here

