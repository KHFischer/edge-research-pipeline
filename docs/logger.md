## 🧠 Class: PipelineLogger

### 📄 What It Does
- Logs key steps of a data pipeline as human-readable Markdown, machine-readable JSON, or both.
- Captures step names, parameters, and optional DataFrame snapshots incrementally as the pipeline executes.

### 🚦 When to Use
- Use when building a research, backtesting, or data cleaning pipeline where **auditability** and **process transparency** are required.
- Enables users to retrace exactly what was done to the data, when, and with what parameters.
- Suitable for any ETL, data science, or feature engineering workflow that benefits from traceable logs.
- Do **not use** if your pipeline is purely performance-critical and doesn’t require step-level auditing.

### 🔢 Inputs and Outputs

**Constructor Arguments:**
- `log_path` (`str`): Base path for log files (without extension).
- `log_markdown` (`bool`): Enable Markdown logging (`.md`). Default: `True`.
- `log_json` (`bool`): Enable JSON structured logging (`.json`). Default: `False`.

**Method: log_step() Arguments:**
- `step_name` (`str`): Short description of the current pipeline step.
- `info` (`dict`): Parameters or metadata (converted to JSON).
- `df` (`pd.DataFrame`, optional): Optional DataFrame to log (snapshot).
- `max_rows` (`int`, optional): Maximum number of rows from the DataFrame to log (default: 20).

**Outputs:**
- Writes logs to one or more files:
  - `*.md`: Incremental Markdown audit log (human-readable).
  - `*.json`: Newline-delimited JSON log (machine-readable).
- Each log_step() appends a new entry.

### ⚠️ Design Notes / Gotchas
- **Overwrites** any existing log files at the same path when initialized.
- Logs are written incrementally: no log history stored in memory.
- JSON log is newline-delimited JSON objects (one per log step).
- DataFrames are truncated to `max_rows` in both Markdown and JSON logs to avoid excessive file size.
- Empty or missing DataFrames are logged as “No DataFrame provided.” (Markdown) or `null` (JSON).
- Safe for repeated use across multiple pipeline stages.
- Avoid using sensitive data in parameters if JSON logs may be shared.

### 🔗 Related Functions
- No direct dependencies; designed for general pipeline logging.
- Typically used alongside pipeline components like:
  - `apply_column_type_cleaning()`
  - `handle_missing_data()`
  - `cleaning_pipeline()`

### 🧪 Testing Status
- Fully unit tested:
  - Markdown-only, JSON-only, and dual-mode logging.
  - Logs with and without DataFrames.
  - Max row truncation logic.
  - File creation and content correctness.
  - Logs written incrementally and in correct formats.
- No known uncovered edge cases.
