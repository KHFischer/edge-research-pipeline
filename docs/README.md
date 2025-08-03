# 📚 Documentation Index

This folder contains module-level documentation for all major components of the edge research pipeline. Each `.md` file corresponds to one logical module or subsystem and contains structured documentation for all relevant functions in that module.

## 🧭 Structure

Each doc file includes:
- A short description of the module’s purpose
- Function-level documentation with:
  - What it does
  - When to use it
  - Inputs and outputs
  - Design notes and testing status

## 📂 Module Documentation

| File              | Description                                               |
|-------------------|-----------------------------------------------------------|
| `cleaning.md`     | String cleanup, type coercion, missing value handling     |
| `engineering.md`  | Feature construction (lags, ratios, binning, encoding)    |
| `target.md`       | Target column creation and labeling logic                 |
| `mining.md`       | Rule mining algorithms and feature generation             |
| `validation.md`   | In-sample and out-of-sample validation test logic         |
| `pipeline.md`     | Full pipeline orchestration (single run + grid runner)    |
| `calculator.md`   | Core statistics and metrics calculation                   |
| `logger.md`       | Structured logging interfaces and helpers                 |
| `config_validator.md` | Config loading and validation rules                   |

## ⚙️ Parameter References

| File              | Description                                               |
|-------------------|-----------------------------------------------------------|
| `params.md`       | All default and override-able parameters for single runs  |
| `grid_params.md`  | Additional options for grid search execution and orchestration |

- Parameters are grouped by module/function and include defaults, expected types, and valid values.
- Use these to build, validate, or override your YAML config files.

## 🧪 How These Docs Are Maintained

- Docs are written manually to ensure clarity and consistency.
- Function-level doc blocks are kept in sync with the actual code but are not auto-generated.
- When updating a module, remember to update its `.md` file here.

## 🔗 Usage Tip

If you're using this package programmatically or via CLI and want to understand what a function or config option does, check the corresponding `.md` file here.

