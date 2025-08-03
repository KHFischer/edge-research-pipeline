# 📂 statistics

## 🧠 What This Module Does

- Calculates statistical metrics on binary one-hot encoded DataFrames.
- Used internally to evaluate rule strength and edge characteristics in the rule mining pipeline.
- Focuses on computing metrics between antecedents and consequents, such as lift, support, and confidence.

## 🧰 Main Features

- ✅ Computes common association rule statistics:
  - Support
  - Lift
  - Confidence
  - Conviction
  - Leverage
- ✅ Supports masking and filtering via config (e.g., `stat_min_support`)
- ✅ Returns results as a structured dataframe + log dict
- ✅ Used automatically inside mining and validation pipelines

## 🚀 How to Use

Although typically invoked internally via `data_prep_pipeline()` in the rule mining module, it can be used directly for debugging, custom workflows, or prototyping.

### 🔹 Basic Usage

```python
from edge_research.statistics.calculator import generate_statistics

stats_df, stats_log = generate_statistics(df_onehot, cfg)
````

### 🔹 With Overrides

```python
stats_df, stats_log = generate_statistics(
    df_onehot,
    cfg,
    overrides={"stat_min_support": 5, "stat_min_lift": 1.2}
)
```

### 📌 Input Example (One-Hot DataFrame)

| sector=communication services | sector=consumer cyclical | sector=consumer defensive |
| ----------------------------- | ------------------------ | ------------------------- |
| 0                             | 1                        | 0                         |
| 0                             | 0                        | 1                         |
| 1                             | 0                        | 0                         |

This binary matrix represents antecedent/consequent membership and is expected by `generate_statistics()`.

## ⚙️ Configuration Reference

Behavior is controlled by the config object or YAML under the `stat_*` keys. For a full list of options, see:

📄 `docs/params.md` → **⚙️ Config: Statistics Mask**

Common params include:

* `stat_min_support`
* `stat_min_lift`
* `stat_use_masking`
* `stat_confidence_bounds`

## ⚠️ Design Notes / Caveats

* Expects a **binary one-hot encoded DataFrame** with no NaNs.
* Assumes input rows represent **co-occurrence of rule antecedents and consequents**.
* Relies on column names to infer rule semantics — do not rename columns arbitrarily.
* No internal validation of dataframe shape — expected to be pre-cleaned.
* Returns both the stats dataframe and a `stats_log` dictionary with raw counts and debug metadata.

## 🧪 Testing Status

* Unit tested with synthetic one-hot matrices and known rule combinations.
* Edge cases for zero support, all-zero rows, and perfect correlation are covered.
* Coverage of override logic tested separately in rule mining integration tests.

## 🔗 Related Modules

* `mining/`: Calls `generate_statistics()` after candidate rule generation.
* `validation/`: May use stats output to filter or validate rule quality.
* `preprocessing/`: Responsible for one-hot encoding prior to stats calculation.
