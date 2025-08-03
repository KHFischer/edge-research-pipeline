# 📂 preprocessing

## 🧠 What This Module Does

This module prepares raw or messy financial data for rule mining by handling all preprocessing steps. It standardizes data inputs, engineers informative features, and safely creates target columns for supervised learning or signal discovery.

Specifically, it performs:
- **Cleaning**: Detects and handles missing values, outliers, scaling issues, and inconsistent types.
- **Engineering**: Builds temporal, categorical, and ratio-based features for downstream mining tasks.
- **Target Creation**: Merges engineered features with forward returns and constructs a configurable target variable (binary, quantile, custom bins).

## 🧰 Main Features

### Cleaning (`cleaning.py`)
- Missing value imputation and detection (thresholding, ffill)
- Outlier handling (winsorization, custom logic)
- Scaling and normalization (z-score, robust, quantile, unit vector)
- Redundancy pruning (low variance, high correlation)
- Custom column detection, conversion, and enforcement
- Summary stats via `generate_data_summary()`

### Engineering (`engineering.py`)
- Temporal lag features (single or multiple horizons)
- Datetime feature extraction
- Ratio and binning generation
- Categorical encoding (OHE, sweep)
- Combined `engineer_pipeline()` and validation helper

### Target (`target.py`)
- Log-return, volatility-adjusted, or smoothed targets
- Binning (quantile, binary, or custom thresholds)
- Forward merge logic (HLOC-aware, safe for backtests)
- Full pipeline with configuration and validation tools

## 🚀 How to Use

Each component can be used independently or as a chained pipeline. Config is passed via a central `cfg` object and optionally overridden inline.

```python
from scripts.preprocessing.cleaning import clean_pipeline, generate_data_summary
from scripts.preprocessing.engineering import engineer_pipeline, validate_pipeline_input
from scripts.preprocessing.target import target_pipeline, validate_target_pipeline_input
from pprint import pprint

# Step 1: Clean data
df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger, **{"to_winsorize": True})

# Step 2: Engineer features
df_engineered, engineer_logs = engineer_pipeline(df_cleaned, cfg, logger, **{"to_drop_no_data": True})

# Step 3: Create target column
df_target, target_logs = target_pipeline(df_engineered, cfg, hloc_df, logger, **{"mode": "quantile_normal"})

# Optionally validate structure and summarize
summary = validate_target_pipeline_input(df_target, cfg.id_cols, cfg.date_col, cfg.target_col, cfg.drop_cols)
pprint(summary, sort_dicts=False)
````

Configuration options are extensive and can be found in `docs/params`:

* Imputation, Winsorization, Scaling, Feature Engineering
* Target mode and binning logic

## ⚠️ Design Notes / Caveats

* ⚠️ Assumes a consistent schema with `id_cols` and `date_col` set in config
* ⚠️ Input DataFrame is not modified in-place, but returned copies may share memory unless deep copied
* ⚠️ `target_pipeline` expects `hloc` (High-Low-Open-Close) DataFrame for forward returns
* ✅ Supports both YAML-based config or inline parameter overrides
* ❗ Requires `pandas` datetime columns to be timezone-naive and consistently formatted

## 🧪 Testing Status

* ✅ All pipeline stages covered by unit tests
* ⚠️ Edge-case validation (e.g., all-NaN columns, empty bins) under active development
* 🧪 Config schema validation planned but not yet enforced

## 🔗 Related Modules

* [`mining`](../mining): Consumes preprocessed data for rule discovery
* [`validation`](../validation): Applies statistical tests to mined rules
* [`data_prep`](../data_prep): Handles synthetic and corrupted data generation for stress testing

