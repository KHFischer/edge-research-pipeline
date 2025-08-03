# 🧩 examples/

This folder contains clean, copy-pasteable usage examples for each major component of the edge research pipeline.

## 🎯 Purpose

These examples are written for:
- **Hands-on learners** who prefer seeing the pipeline in action
- Users exploring how to **run functions with configs or manual overrides**
- Developers validating pipeline behavior without reading full documentation

Each file is a runnable `.py` script that demonstrates one module's core functionality using:
- Sample data (`load_samples()`)
- Clean, minimal config setups (both inline and config-driven)
- Logging and output tracing when appropriate

## 📁 Files Overview

| File                    | Demonstrates Functions From              |
|-------------------------|------------------------------------------|
| `example_cleaning.py`   | `cleaning.py` — missing handling, types, scaling |
| `example_engineering.py`| `engineering.py` — lags, ratios, binning, encoding |
| `example_target.py`     | `target.py` — label generation           |
| `example_calculator.py` | `calculator.py` — stats and metrics      |
| `example_mining.py`     | `mining.py` — rule mining and parsing    |
| `example_validation.py` | `validation.py` — bootstrap, WFA, FDR    |
| `example_config.py`     | `config_validator.py` — loading, merging configs |
| `example_logger.py`     | `logger.py` — how to attach and inspect logs |
| `example_pipeline.py`   | `pipeline.py` — full end-to-end pipeline (single + grid) |

## 🚀 How to Use

From the project root:
```bash
python examples/example_cleaning.py
```

Or open the file in your IDE and run selected sections interactively.

Examples are structured to:

* Show minimal usage first
* Then show full config-driven usage
* Then show optional overrides

## 📎 Note

* These examples are **not unit tests** — they’re for human-readable demonstration and validation.
* Sample data lives in `./data/` and is automatically loaded where needed.
* See `docs/<module>.md` for detailed parameter descriptions.
