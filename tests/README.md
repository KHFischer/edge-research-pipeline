# 🧪 tests/

This folder contains unit tests for all major components of the edge research pipeline. Each test file corresponds to a logical module and is written using `pytest`.

## ✅ Test Structure

Each test file is organized to validate:
- Core functionality and expected outputs
- Input validation and error handling
- Edge cases and data type robustness
- Output schema (e.g., shape, column names)
- Config-driven behavior when applicable

All tests are written for clarity, maintainability, and modularity. Dummy inputs are generated inline using `pandas` and `numpy` — no test data files are required.

## 📁 Files Overview

| Test File               | Tests Corresponding Module(s)             |
|-------------------------|-------------------------------------------|
| `test_cleaning.py`      | `cleaning.py` — missing handling, coercion, etc. |
| `test_engineering.py`   | `engineering.py` — feature creation       |
| `test_target.py`        | `target.py` — target label creation       |
| `test_calculator.py`    | `calculator.py` — statistics + metrics    |
| `test_mining.py`        | `mining.py` — rule mining and parsing     |
| `test_validation.py`    | `validation.py` — bootstrap, FDR, WFA     |
| `test_pipeline.py`      | `pipeline.py` — grid runner, CLI loading  |
| `test_config_validator.py` | `config_validator.py` — parameter loading & merging |
| `test_logger.py`        | `logger.py` — structured logging helpers  |

## 🚀 Running Tests

From the project root, run:
```bash
pytest tests/
```

To run a specific file:

```bash
pytest tests/test_cleaning.py
```

To run with verbosity and capture:

```bash
pytest -v --capture=no
```

## 📌 Guidelines for New Tests

When adding tests:

* Mirror the structure of the module under test.
* Use `pytest.mark.parametrize` for variation coverage.
* Keep test functions small and self-contained.
* Use `assert_frame_equal()` for DataFrame output.
* Always test both "happy path" and failure modes (e.g., wrong types, shapes).

## 🧪 Test Philosophy

These tests are designed to:

* Ensure pipeline correctness under expected conditions
* Catch regressions when modules are refactored
* Serve as living usage examples for module behavior

