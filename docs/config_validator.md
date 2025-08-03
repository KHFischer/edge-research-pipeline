## 🧠 Function: `config_validator`

### 📄 What It Does
- Validates the keys and values in a configuration dictionary against expected types and value constraints defined in a `groups` schema.
- Raises a `ValueError` if any parameter fails validation or if unknown parameters are present (in strict mode).

### 🚦 When to Use
- Use this function to enforce structured, type-safe, and value-constrained config input before running any downstream pipeline or model.
- Especially useful when loading configs from user input, YAML files, or other sources where schema enforcement is critical.

> Example: Run `config_validator(cfg_dict, GROUPS)` before using `cfg_dict` in a pipeline to ensure all expected fields are present and valid.

- ✅ Use when configs are dynamic or user-supplied
- ❌ Don’t use if schema enforcement is already handled by a strict parser or dataclass validator

### 🔢 Inputs and Outputs
- **Inputs**
  - `params_dict` (`Dict[str, Any]`): The actual config parameters to validate
  - `groups` (`Dict[str, Any]`): Schema grouping with keys such as:
    - `'int'`, `'float'`, `'string'`, `'bool'`, `'list'`, `'all_or_list'`
    - `'val_from_list'`: List of `{param_name: [valid_values]}`
    - `'quantiles'`: List of `{edges_param: labels_param}`
  - `logger` (`Optional[logging.Logger]`, default=`None`): Optional logger for verbose validation reporting
  - `strict` (`bool`, default=`True`): If True, unexpected parameters will raise an error

- **Outputs**
  - Returns `None`. Raises `ValueError` if validation fails.

### ⚠️ Design Notes / Gotchas
- The `groups` structure must be complete and accurate — keys missing from `groups` may cause valid parameters to be flagged in strict mode.
- Handles complex validations like:
  - `'all_or_list'`: must be `"all"` or a list of strings
  - `'val_from_list'`: must match an allowed list of values
  - `'quantiles'`: validates bin/label shape and consistency
- Uses several external validators (`is_all_or_list_of_strings`, `is_valid_quantile_bins`, etc.) — these must be defined in the runtime context.
- All logging is optional and guarded by a logger check.
- Does **not** mutate `params_dict`.

### 🔗 Related Functions
- `_gather_known_keys`: collects all expected keys from `groups`
- `_validate_type`: checks type constraints
- `_validate_special_cases`: handles custom value checks and grouped validation
- External validators:
  - `is_all_or_list_of_strings`
  - `is_in_choices`
  - `is_valid_custom_bins`
  - `is_valid_quantile_bins`
  - `is_valid_bin_labels`

### 🧪 Testing Status
- ✅ Fully unit tested with `pytest`
  - Covers normal, invalid, and edge cases
  - Includes `val_from_list`, `quantiles`, and strict mode logic
- ⛔ Assumes external helper functions are already tested — not tested internally here

## 🧠 Function: `load_params`

### 📄 What It Does
- Loads default and custom configuration parameters from YAML files or Python dictionaries.
- Merges the two sets of parameters with custom values overriding defaults, validates them, and optionally logs and prints them.

### 🚦 When to Use
- Use this function when you want to load config values for a pipeline or module from either `.yaml` files or programmatic dicts, with support for user overrides.
- Particularly useful in projects that support user customization of default config settings.

> Example: Load `params/default_params.yaml` and override with `params/custom_params.yaml` or in-memory `dict`s before passing into a model or research pipeline.

- ✅ Use when:
  - You want layered config loading and validation.
  - You want the same logic to work in both script-based and interactive workflows.
- ❌ Don’t use if:
  - You need strict schema enforcement (this only validates structure/types via `config_validator`).
  - You require merging from non-YAML or non-dict sources (e.g., JSON, CLI args — extend separately).

### 🔢 Inputs and Outputs

**Inputs**
- `default_params` (`str | dict | None`): YAML file path or dict of defaults. If `None`, treated as empty.
- `custom_params` (`str | dict | None`): YAML file path or dict of custom values. Overrides defaults.
- `verbose` (`bool`): If `True`, logs and prints config details.
- `logger` (`logging.Logger | None`): Optional logger for validation messages.

**Output**
- `ChainMap`: Merged dictionary-like object with custom parameters on top of defaults.

### ⚠️ Design Notes / Gotchas
- Silent fallback on missing YAML files (returns `{}` instead of raising) — may hide errors if not careful.
- Will raise `ValueError` if input is not a dict or `.yaml` string path.
- Uses `ChainMap`, which behaves like a merged view — it does not copy or deep merge nested structures.
- Depends on external `config_validator()` and `pretty_print_params()` to handle validation and printing — must be defined in scope.

### 🔗 Related Functions
- `config_validator`: Validates merged config structure.
- `pretty_print_params`: Optional display of final config (if `verbose=True`).
- `_load_param_source`: Private helper inside the function to load and normalize input.

### 🧪 Testing Status
- ✅ Unit tested with `pytest`
  - Covers merging logic, file input, type errors, and fallback behavior
  - Uses monkeypatching to isolate external dependencies
- ⚠️ Assumes `config_validator` handles its own validation logic — not tested in this context
