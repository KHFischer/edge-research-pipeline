# 🧠 edge_research/

This directory contains the core implementation modules for the edge research pipeline. Each subfolder corresponds to a major step or support system in the research workflow, and contains modular, reusable components that can be run independently or as part of a full pipeline.

---

## 🧭 Module Overview

| Subfolder               | Purpose                                                      |
|-------------------------|--------------------------------------------------------------|
| `preprocessing/`        | Data cleaning, feature engineering, and supervised target creation |
| `rules_mining/`         | Rule mining algorithms and interpretable signal generation   |
| `validation_tests/`     | Statistical and robustness tests (e.g. bootstrap, WFA, FDR)   |
| `statistics/`           | Core metrics and utility functions for scoring rules/features |
| `pipeline/`             | Full pipeline orchestration (single run, grid runner, CLI)   |
| `logger/`               | Structured logging configuration used across modules         |
| `utils/`                | Lightweight utilities (config loading, schema checks, etc.)

---

## 🔌 How These Modules Fit Together

1. **Preprocessing**: Prepares cleaned, engineered features and supervised targets.
2. **Rule Mining**: Generates candidate rules from features using interpretable models.
3. **Validation Tests**: Evaluates rule robustness and predictive persistence.
4. **Statistics**: Provides reusable functions for scoring, filtering, and diagnostics.
5. **Pipeline**: Wraps the full process into a single callable (or CLI-executable) unit.
6. **Logger / Utils**: Shared infrastructure — logging, config management, helpers.

Each submodule is fully documented in its own `README.md`.

---

## 🚀 Usage Modes

These modules are used in:
- CLI-driven pipeline runs via `pipeline/main.py`
- Programmatic experiments via `pipeline/edge_research_pipeline(...)`
- Component-level exploration via the `examples/` folder

---

## 🛠 Dev Notes

- All submodules are independently importable (no implicit shared state).
- Configs are passed explicitly, either as `dict` or via YAML → `Config` object.
- All modules support verbose logging via the shared logger setup.

---

## 🔗 Related Resources

- 🧪 `tests/`: Unit tests per module using `pytest`
- 📚 `docs/`: Full parameter and function documentation
- 🧩 `examples/`: Copy-pasteable usage examples for each module

