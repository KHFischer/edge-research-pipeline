"""
Example: Config Loading Usage
Location: examples/config_loading_example.py

Demonstrates loading and using pipeline configuration via:
- YAML config files (recommended)
- Python dict (advanced/override use only)

See: docs/config.md for all supported parameters and Config dataclass structure.
"""

import edge_research
from edge_research.params.config_validator import load_params, Config
from pathlib import Path

# --- 1. Load Config from YAML Files (Recommended) ---

# Resolve params directory inside installed package
params_dir = Path(edge_research.__path__[0]) / "params"

# Load configuration files
default_params = params_dir / "default_params.yaml"
custom_params = params_dir / "custom_params.yaml"

params = load_params(str(default_params), str(custom_params))
cfg = Config(**params)

print("Run name from config (YAML):", cfg.run_name)  # e.g., 'my_first_test' (default or custom value)

# --- 2. Load Config with Python Dict Override (Advanced) ---

custom_params_dict = {"run_name": "my_run"}
params = load_params(str(default_params), custom_params_dict)
cfg = Config(**params)

print("Run name with dict override:", cfg.run_name)  # 'my_run'

# Note: Supplying all defaults as a dict is NOT recommended due to the large number of required keys.

# --- 3. Verbose Parameter Logging (For Debugging) ---

params = load_params(str(default_params), custom_params_dict, verbose=True)
# This prints all parameters and their values for full transparency.

# --- 4. Edit Custom YAML for Persistent Changes ---

# For most users, directly edit 'params/custom_params.yaml' to override any value.
# Custom values always take precedence over defaults.

# --- Reference ---

# cfg is a Config dataclass (not a dict). Use dot notation for access, e.g.:
print("Example metric:", cfg.apriori_metric)

# For a full list of parameters, see: docs/config.md or the Config dataclass definition.