# 📂 params

## 🧠 What This Module Does

- Provides a structured, validated configuration system for all pipeline components.
- Loads and merges YAML-based parameter files into a strict, schema-validated `Config` object.
- Ensures all pipeline steps receive well-typed, consistent configuration without boilerplate or missing keys.

## 🧰 Main Features

- ✅ **Default + custom parameter merging** via `default_params.yaml` and `custom_params.yaml`
- ✅ **Schema enforcement** via `config_schema.py` using `dataclass` with type checking
- ✅ **Safe parameter loading** with built-in validation and override support
- ✅ **Programmatic config override** from Python `dict` (e.g., CLI workflows)
- ✅ Fully compatible with every pipeline step (e.g., cleaning, mining, validation)

## 🚀 How to Use

### 🔹 Load Config from YAML

```python
from params.config_validator import load_params, Config

# Load default and custom parameter files
default_params_path = "params/default_params.yaml"
custom_params_path = "params/custom_params.yaml"

params = load_params(default_params_path, custom_params_path, verbose=False)
cfg = Config(**params)
````

### 🔹 Load Config from Dict (Override Style)

```python
# Use defaults, override with dict
custom_override = {"run_name": "my_custom_run"}
params = load_params(default_params_path, custom_override, verbose=False)
cfg = Config(**params)

print(cfg.run_name)  # => "my_custom_run"
```

### 🔹 Use in Pipeline Step

```python
from scripts.preprocessing.cleaning import clean_pipeline

df_cleaned, logs = clean_pipeline(feature_df, cfg, logger)
```

## ⚙️ Configuration Reference

* All valid keys, types, and expected values are documented in:
  📄 `docs/params.md`

* YAML files must be structured as flat key-value mappings (no nested dicts).

* All keys in `Config` must be present — partial configs will raise validation errors.

## ⚠️ Design Notes / Caveats

* **All config keys are required** — no default inference inside `Config`. Use `default_params.yaml` to ensure completeness.
* Parameter values must match expected types exactly (`bool`, `int`, `float`, `str`, `list`, etc.).
* Merging behavior: `custom_params.yaml` (or dict) **overwrites** `default_params.yaml` at the top level.
* Config schema is defined in `config_schema.py` via a `@dataclass` — change types there if schema evolves.
* Verbose mode in `load_params()` prints merged and validated config for debugging.

## 🧪 Testing Status

* Unit tested:

  * YAML merging logic
  * Schema validation and type enforcement
  * Dict vs file-based loading behavior
* Edge cases:

  * Missing keys and bad types are explicitly tested
  * Fallback logic and silent failures are **not allowed**

## 🔗 Related Modules

* Used by all modules that require config injection (e.g., `preprocessing`, `rules_mining`, `validation_tests`)
* Compatible with `logger/` for run\_name-based log paths
* Examples for usage in `examples/params_example.py`
