# 📂 logger

## 🧠 What This Module Does

- Provides a custom logger class (`PipelineLogger`) used across all pipeline stages.
- Supports logging in both **JSON format** (structured, machine-readable) and **Markdown tables** (readable summaries).
- Ensures all experiment steps are auditable, human-readable, and exportable to file.

## 🧰 Main Features

- ✅ Dual-format logging: Markdown and/or JSON
- ✅ Integrates seamlessly into any step of the pipeline
- ✅ Lightweight API: just `.log()` and `.save()` calls
- ✅ Configurable via `cfg` or manual overrides
- ✅ Automatically handles creation of save paths and filenames
- ✅ Output is cleanly formatted for presentation or parsing

## 🚀 How to Use

```python
from edge_research.logger.logger import PipelineLogger
from pathlib import Path

# Define result save path and logger config
run_name = cfg.run_name  # e.g., "experiment_01"
log_markdown = cfg.log_markdown  # e.g., True
log_json = cfg.log_json  # e.g., False

res_save_path = "data/results"
save_folder = Path(res_save_path) / run_name
save_folder.mkdir(exist_ok=True, parents=True)

# Initialize logger
logger = PipelineLogger(
    log_path=save_folder / f"{run_name}_log.md",
    log_markdown=log_markdown,
    log_json=log_json
)

# Example usage -- from FDR validation test
logger.log_step(
    step_name="FDR Multiple Correction validation test",
    info=fdr_kwargs,
    df=fdr_log,
    max_rows=log_max_rows,
)
```

You can call `.log()` repeatedly during the pipeline, then call `.save()` once at the end of a stage.

## ⚙️ Configuration Reference

This logger can be configured via a central `cfg` object or passed manually.

| Config Key     | Description                          |
| -------------- | ------------------------------------ |
| `run_name`     | Name of the run, used in file naming |
| `log_markdown` | Enable Markdown logging (True/False) |
| `log_json`     | Enable JSON logging (True/False)     |

These values can be set via YAML or constructed dynamically inside your script.

## ⚠️ Design Notes / Caveats

* Assumes the log path’s parent directories already exist (caller must create them).
* Markdown format is optimized for table rendering, not for large text blobs.
* JSON mode outputs flat key-value pairs — not nested dicts.
* Logs are **not streamed** to disk immediately — `.save()` must be called explicitly.
* Designed to be run **once per stage** (not long-running loggers like `logging` module).

## 🧪 Testing Status

* Fully unit tested:

  * Markdown and JSON output correctness
  * Log overwrite and appending behavior
  * Handling of various input types (str, int, float, list)
* Edge case coverage for:

  * Empty logs
  * Invalid paths or formats

## 🔗 Related Modules

* Used throughout all pipeline steps (e.g., `cleaning`, `engineering`, `mining`)
* Logs from this module are saved alongside results for traceability
* Compatible with `cfg` objects and CLI overrides

