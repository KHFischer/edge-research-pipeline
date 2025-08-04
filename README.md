# 🧠 Edge Research Pipeline

The **Edge Research Pipeline** is a modular, privacy-first research toolkit for **rule mining**, **pattern discovery**, and **interpretable machine learning** on **tabular datasets**. It supports automated **feature engineering**, **target labeling**, **robust validation**, and **signal discovery** workflows across domains including **quantitative finance**, **structured data mining**, and **subgroup analysis**, its techniques are broadly applicable to any domain involving structured data and statistical rule discovery.

[![PyPI version](https://img.shields.io/pypi/v/edge-research-pipeline)](https://pypi.org/project/edge-research-pipeline/)

---

## 🚀 Key Features

A flexible, modular Python library enabling you to:

* **Clean, normalize, and transform** tabular datasets
* **Engineer features** relevant to finance, statistics, and other structured-data domains
* **Generate and label custom targets** for supervised tasks
* **Discover signals** using rule mining and pattern search methods
* **Perform robust validation tests** (e.g., train/test splits, bootstrap, walk-forward analysis, false discovery rate)
* **Reproduce results** with complete configuration export and local-only processing
* **Efficiently execute parameter grids** via function calls or a CLI

---

## 🔒 Privacy by Design

All computations run **locally**—no data ever leaves your environment. Designed explicitly for regulated industries, confidential research, and reproducible workflows.

---

````markdown
## 📦 Installation

Install the latest release from **PyPI**:

```bash
pip install edge-research-pipeline
```

📦 [View on PyPI →](https://pypi.org/project/edge-research-pipeline/)

---

### 🛠️ Advanced Option (Dev/Offline)

To install using the repo-based dependencies file:

```bash
pip install -r ./requirements.txt
```

 **Note:** This file was generated via `pipreqs` and may need further validation in some environments.

---

## ⚠️ Compatibility Notes & Optional Dependencies

This project includes optional support for advanced mining and synthetic data tools like `orange3` and `synthcity`. These libraries are powerful but have strict, conflicting version requirements that cannot be satisfied simultaneously in a single install.

### 🧨 Known Conflicts

* `orange3` requires `xgboost >=1.7.4, <2.1`
* `synthcity` requires `xgboost >=2.1.0`
* `xgbse` (a dependency of `synthcity`) enforces this version split
* Installing both libraries together will cause `pip install` to fail due to an irreconcilable conflict on `xgboost`


### ✅ Resolution

To avoid these conflicts:

* The core package **does not include** `orange3` or `synthcity` by default
* You can install them separately using **extras**:

 ```bash
 pip install edge-research-pipeline[orange]     # for orange3-based rule data generation
 pip install edge-research-pipeline[synth]      # for synthetic data workflows
 ```

⚠️ **Note:** Installing both `orange3` and `synthcity` via extras will fail due to incompatible `xgboost` requirements.
If you need both, install the pipeline without either extra:

```bash
pip install edge-research-pipeline
```

Then manually install each library:

```bash
pip install orange3
pip install synthcity
```

This bypasses pip’s dependency resolver and allows both to coexist — but may require you to manage compatibility manually.

---


### ⚠️ Additional Dependency Warnings

Some third-party tools (e.g., `torch`, `scipy`, `pandas`, `databricks`, `ydata-profiling`) may also have mutually incompatible version constraints depending on your environment. We strongly recommend installing this package in a **clean virtual environment** to prevent dependency resolution issues:

```bash
python -m venv erp_env
.\erp_env\Scripts\activate      # Windows
# source erp_env/bin/activate   # macOS/Linux
pip install edge-research-pipeline
```

---

## 🧩 Quick Start Example

Run a full pipeline example via the command line:

```bash
python edge_research/pipeline/main.py params/grid_params.yaml
```

Or check the ready-to-run examples in the [`examples/`](./examples/) directory.

---
<!--
Keywords:
rule mining, pattern discovery, interpretable machine learning, feature engineering,
subgroup discovery, tabular ML, signal validation, financial machine learning, data cleaning pipeline,
synthcity, orange3, CN2 rule induction, robust backtesting, rule-based modeling, bootstrapping, walk-forward analysis
-->

## 📁 Project Structure

```text
edge-research-pipeline
├── data/                  # Sample datasets (sandbox only)
├── docs/                  # Documentation per module
├── edge_research/         # Core logic modules
│   ├── logger/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── rules_mining/
│   ├── statistics/
│   ├── utils/
│   └── validation_tests/
├── examples/              # Copy-pasteable usage examples
├── params/                # Configuration files
├── tests/                 # Unit tests for major functions
├── LICENSE
├── README.md
└── requirements.txt
```

Detailed explanations for each subfolder are available within their respective READMEs.

---

## ⚙️ Configuration Philosophy

Configuration files are managed via YAML files within `./params/`:

* **`default_params.yaml`**: Base configuration with mandatory default values (do not modify)
* **`custom_params.yaml`**: Override specific parameters from defaults
* **`grid_params.yaml`**: Parameters specifically for orchestrating grid pipeline runs

**Precedence hierarchy:**

* For pipeline runs (`pipeline.py` or CLI):
  `grid_params > custom_params > default_params`
* For direct function calls:
  `custom_params > default_params`

Parameters can also be directly overridden by passing a Python dictionary at runtime.

---

## 🧪 Testing

Unit tests cover all major logical functions, ensuring correctness and robustness. Tests are written using `pytest`. Short utility functions, simple wrappers, and internal helpers are generally not included.

Run tests via:

```bash
pytest tests/
```

---

## 🤝 Contributing

We welcome contributions! Follow these guidelines:

* Keep your commits focused and atomic
* Always provide clear, descriptive commit messages
* Add or update tests for any new feature or bug fix
* Follow existing code style (e.g., use `black` and `flake8` for Python formatting)
* Document new functionality thoroughly within the relevant `.md` file in `docs/`
* Respect privacy-by-design principles—no logging or external data exposure

Feel free to open issues for discussions or submit pull requests directly.

---

## 📄 License

This project is licensed under the **Edge Research Personal Use License (ERPUL)**.
The Edge Research Pipeline is free for personal and academic use.  
**Commercial use requires a license.**

👉 See [PRICING.md](./PRICING.md) for full license tiers and support options.

- ✅ Free for personal, student, and academic use (with citation)
- 💼 Commercial use requires approval (temporarily waived)
- 🔒 No redistribution without permission

See [`LICENSE`](./LICENSE) for full terms.

![License: ERPUL](https://img.shields.io/badge/license-ERPUL-blue)


