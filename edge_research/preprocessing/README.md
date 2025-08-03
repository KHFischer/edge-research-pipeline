# 📂 preprocessing

## 🧠 What This Module Does

This module handles all preprocessing steps required before signal mining. It includes:
- **Cleaning**: missing data handling, outlier treatment, scaling, and column filtering.
- **Engineering**: temporal features, ratio creation, binning, encoding, and other feature construction.
- **Target Creation**: forward merging with HLOC data to create supervised learning targets safely.

## 🧰 Main Features

- ✅ Missingness handling: drop, impute, forward fill
- ✅ Outlier removal: winsorization, z-normalization
- ✅ Scaling: z-score, robust, quantile, unit norm
- ✅ Feature engineering: ratios, lags, datetime features, binning, one-hot encoding
- ✅ Target creation: forward returns, volatility adjustment, quantile binning
- ✅ Fully configurable pipeline with override support for every step
- ✅ Per-step summary statistics with `generate_data_summary()` and `validate_*_input()`

## 🚀 How to Use

All components can be run individually or as part of a custom preprocessing pipeline. Configuration can be passed via a YAML file or overridden at runtime.

### 🔹 Cleaning
```python
from edge_research.preprocessing.cleaning import clean_pipeline, generate_data_summary
from pprint import pprint

df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)

# Optional override
df_cleaned, clean_logs = clean_pipeline(
    feature_df, cfg, logger,
    **{"to_winsorize": True, "to_drop_var": False, "to_drop_corr": True}
)

# Summary
clean_summary = generate_data_summary(df_cleaned)
pprint(clean_summary, sort_dicts=False)
```

### 🔹 Engineering
```python
from edge_research.preprocessing.engineering import engineer_pipeline, validate_pipeline_input

df_engineered, engineer_logs = engineer_pipeline(df_cleaned, cfg, logger)

# Optional override
df_engineered, engineer_logs = engineer_pipeline(
    df_cleaned, cfg, logger,
    **{"to_sweep": False, "to_drop_no_data": True}
)

# Summary
engineer_summary = validate_pipeline_input(df_engineered, cfg.id_cols, cfg.date_col, cfg.drop_cols)
pprint(engineer_summary, sort_dicts=False)
```

### 🔹 Target Creation
```python
from edge_research.preprocessing.target import target_pipeline, validate_target_pipeline_input

df_target, target_logs = target_pipeline(df_engineered, cfg, hloc, logger)

# Optional override
df_target, target_logs = target_pipeline(
    df_engineered, cfg, hloc, logger,
    **{"mode": "quantile_normal", "norm_type": "l2"}
)

# Summary
target_summary = validate_target_pipeline_input(
    df_target, cfg.id_cols, cfg.date_col, cfg.target_col, cfg.drop_cols
)
pprint(target_summary, sort_dicts=False)
```

For full working examples and sample configs, see:
📄 `examples/preprocessing_example.py`

## ⚙️ Configuration Reference

This module uses a config dictionary or YAML file to drive behavior.

For a complete list of parameters and expected formats, refer to:

📘 `docs/params/`:

* `missingness.md`
* `imputation.md`
* `winsorization.md`
* `scaling.md`
* `variance_check.md`
* `correlation_check.md`
* `feature_engineering.md`
* `lags.md`
* `binning.md`
* `target_calculation.md`
* `target_binning.md`

## ⚠️ Design Notes / Caveats

* Expects a **datetime index or column** for all time-aware operations.
* Columns listed in `drop_cols`, `id_cols`, or `date_col` must be clearly specified in config.
* Assumes **numeric columns** after cleaning for most engineering and target operations.
* The pipeline **does not mutate input dataframes** in place.
* All functions return a cleaned/enhanced dataframe + log dictionary for auditability.
* Target creation assumes the presence of an external HLOC dataframe for merging returns.

## 🧪 Testing Status

* All major components are unit tested with pytest.
* Edge case coverage (e.g., empty columns, all-missing, duplicate column names) is partially tested.
* Overrides and manual config injection tested via regression harness in `tests/test_preprocessing.py`.

## 🔗 Related Modules

* `mining/`: Next stage in the pipeline — rule mining and signal discovery.
* `utils/`: Contains shared utilities (e.g. config loaders, logging setup, schema validators).
* `examples/`: Working scripts for different pipeline stages and full pipeline runs.
