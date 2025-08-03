# 📂 rules_mining

## 🧠 What This Module Does

- Performs rule mining on a preprocessed dataset using a configurable set of mining algorithms.
- Used in both standalone signal discovery workflows and during validation splits in testing pipelines.
- Supports flexible data augmentation (e.g., synthetic data) prior to mining.

## 🧰 Main Features

- ✅ Data preparation pipeline with:
  - Synthetic data generation (e.g. SDV, SynthCity)
  - Corruption/noise injection
  - Column filtering and normalization
- ✅ Rule mining pipeline with support for:
  - Multiple algorithms (Apriori, RuleFit, Subgroup Discovery, CN2, ELSC, CART/C4.5)
  - Configurable miner selection and filtering
  - Verbose logging and rule export
- ✅ Modular override system for rapid experimentation

## 🚀 How to Use

The typical workflow involves:
1. Preparing a dataset for mining
2. Running one or more mining algorithms on the prepared data

### 🔹 Step 1: Prepare Dataset

```python
from edge_research.rules_mining.mining import data_prep_pipeline

df_mining, prep_logs = data_prep_pipeline(df_target, cfg, logger)

# With overrides (e.g., disable logging or force synthetic data)
df_mining, prep_logs = data_prep_pipeline(
    df_target, cfg, logger,
    **{"silence": True}
)
```

### 🔹 Step 2: Run Mining Algorithms

```python
from edge_research.rules_mining.mining import mining_pipeline

results, rule_counts, mining_logs = mining_pipeline(df_mining, cfg, logger)

# Example with custom miners
results, rule_counts, mining_logs = mining_pipeline(
    df_mining, cfg, logger,
    **{"miners": ["rulefit", "elcs", "cn2"]}
)
```

For a full end-to-end example, refer to:
📄 `examples/mining_example.py`

## ⚙️ Configuration Reference

All mining and prep behavior is driven by config keys. For a complete list of options, see:

📘 `docs/params.md`:

* `⚙️ Config: Synthetic Data`
* `⚙️ Config: SDV`
* `⚙️ Config: Miners`
* `⚙️ Config: Apriori`
* `⚙️ Config: Rulefit, Subgroup, CART`

## ⚠️ Design Notes / Caveats

* Input must be **fully preprocessed** and encoded—use `preprocessing/` module before calling `data_prep_pipeline()`.
* Synthetic data generation (if enabled) can significantly alter rule quality—use carefully when benchmarking.
* Rule mining assumes **binary, one-hot features** for all miners.
* Output formats are standardized across miners but individual algorithms may emit extra metadata.
* `mining_pipeline()` does **not perform validation** — for full evaluation, use the `validation_tests/` module.

## 🧪 Testing Status

* Unit tests cover:

  * Core miner logic
  * Pipeline-level integration
  * Config override behavior
* Edge cases for empty rulesets, failed miners, and synthetic noise are partially covered.
* Extended miner stress tests live under `tests/mining/`.

## 🔗 Related Modules

* `preprocessing/`: Required to prepare datasets for mining
* `statistics/`: Called internally to compute rule statistics (lift, support, etc.)
* `validation_tests/`: Uses this module to mine rules across train/test splits
* `examples/`: Demonstrates end-to-end use of this module in realistic research scenarios
