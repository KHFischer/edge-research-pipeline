## 🧠 Function: `edge_research_pipeline`

---

### 📄 What It Does

- Runs a complete, configurable quantitative research validation pipeline, orchestrating feature loading, config parsing, data preprocessing, and one or more validation steps (Train/Test, Walk-Forward Analysis, Bootstrap, Null/FDR).
- Saves all configs and outputs (tables, logs, YAML files) to a specified results directory for full traceability.

---

### 🚦 When to Use

- Use when you want to execute a reproducible end-to-end research validation, from data loading through statistical testing, using settings defined in YAML config files.
- Ideal for batch runs, audits, or when you need all artifacts saved for later review, sharing, or compliance.
- Not intended for interactive, step-by-step analysis; use modular functions directly for custom or ad-hoc workflows.

---

### 🔢 Inputs and Outputs

**Inputs:**

- `to_train_test` (`bool`): Run Train/Test split validation. Default `True`.
- `to_wfa` (`bool`): Run Walk-Forward Analysis. Default `True`.
- `to_bootstrap` (`bool`): Run bootstrap resampling validation. Default `True`.
- `to_null_fdr` (`bool`): Run Null/FDR validation. Default `True`.
- `default_params` (`str | Path`): Path to default YAML config file.
- `custom_params` (`str | Path`): Path to custom YAML config file (overrides defaults).
- `feature_path` (`str | Path`): Path to features dataset (`.csv` or `.parquet`).
- `hloc_path` (`str | Path`): Path to HLOC dataset (`.csv` or `.parquet`).
- `res_save_path` (`str | Path`): Output directory for all pipeline results.
- `res_filetype` (`str`): Output file format for all results; `'csv'` or `'parquet'`.
- `verbose` (`bool`): Print progress messages to stdout. Default `True`.

**Outputs:**

- `results` (`dict[str, pd.DataFrame]`): Keyed by step (e.g., `"train_test_results"`, `"wfa_results"`, etc.), each value is a result DataFrame.
- `logs` (`dict[str, Any]`): Keyed by step or artifact (e.g., `"train_test_log"`, `"null_log"`), each value is a DataFrame or metadata log.

---

### ⚠️ Design Notes / Gotchas

- **All output and config files are saved to the results folder for reproducibility.** YAMLs used at runtime are copied to the same directory.
- At least one validation step (`to_train_test`, `to_wfa`, `to_bootstrap`, or `to_null_fdr`) must be `True`, or the function will raise `ValueError`.
- Assumes all helper pipeline steps (data loading, config, loggers, etc.) are implemented and importable.
- Will overwrite results/logs if run multiple times with the same `run_name` in the same results folder.
- Expects compatible YAML schema and input tables; will raise errors if files are missing or malformed.
- Does **not** mutate input files or input data.

---

### 🔗 Related Functions

- [`train_test_pipeline`](#)
- [`wfa_pipeline`](#)
- [`bootstrap_pipeline`](#)
- [`null_pipeline`](#)
- [`fdr_pipeline`](#)
- [`load_params`], [`Config`], [`load_table`], [`save_table`], [`copy_yaml_flat`], [`PipelineLogger`]

---

### 🧪 Testing Status

- Unit tests are present and parameterized for all toggles and core scenarios (see `test_edge_research_pipeline`).
- Tests check exception raising, result key coverage, output types, and output folder creation.
- **Edge cases not fully covered:** Failure of downstream helpers (e.g., if one step crashes or returns bad data), malformed YAMLs, or unsupported filetypes in input.

---

## 🧠 Function: grid_edge_research_pipeline

### 📄 What It Does
- Runs a batch of edge research pipeline experiments using all combinations of user-specified parameters, optionally in parallel.
- Reads grid configuration from a YAML file, executes the pipeline for each parameter set, and returns all results.

### 🚦 When to Use
- Use when you want to automate running your research pipeline across many parameter settings (e.g., for hyperparameter search, ablation, or sensitivity analysis).
- Ideal for large batch runs where reproducibility and config-driven setup are required.
- Not intended for interactive or ad-hoc single-run usage—use a single-pipeline call for that.

### 🔢 Inputs and Outputs
**Inputs:**
- `grid_path` (`str`): Path to YAML file specifying:
  - `param_space` (`dict[str, list[Any]]`): Parameter names and list of values for grid expansion.
  - `n_jobs` (`int`, optional): Number of parallel workers (`1` means no parallelism).
  - Other keys needed by the research pipeline (e.g., `base_run_name`, data paths, etc.).

**Outputs:**
- `List[Tuple[Any, Any]]`: List of tuples `(results, logs)` for each configuration.  
  (Exact type/format depends on `edge_research_pipeline` implementation.)

### ⚠️ Design Notes / Gotchas
- **YAML config must contain `param_space` as a dict**; raises `ValueError` if not.
- **`n_jobs` < 1 or non-int** will raise a `ValueError`.
- Will attempt to run *all* combinations in the parameter grid—be mindful of grid size (combinatorial explosion risk).
- Parallelism uses joblib; over-allocation of workers can lead to out-of-memory if your pipeline is resource-heavy.
- No mutation of inputs; results collected and returned as a list.
- Relies on helper functions: `generate_param_grid` and `_run_single_grid_config`.
- Exceptions in individual pipeline runs are not caught—errors will interrupt the whole batch.

### 🔗 Related Functions
- `generate_param_grid` — expands parameter grid from config
- `_run_single_grid_config` — executes a single pipeline run for a given config
- `edge_research_pipeline` — the main research pipeline logic (not shown here)

### 🧪 Testing Status
- Has dedicated pytest unit tests covering:
  - Serial and parallel runs
  - Invalid and empty inputs
  - One-config and multi-config scenarios
  - Output structure and exception handling for bad configs
- Additional real-world grid explosion and resource-limit edge cases should be tested as the pipeline evolves.
