# 🧠 Edge Research Pipeline

The **Edge Research Pipeline** is a modular, privacy-first research toolkit designed for discovering, validating, and analyzing patterns in tabular datasets. Originally built for **quantitative finance**, its techniques are broadly applicable to any domain involving structured data and statistical rule discovery.

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

## 📦 Installation

Install required dependencies using:

```bash
pip install -r ./requirements.txt
```

**Note:** Dependencies were generated via `pipreqs` and may need further validation.

---

## 🧩 Quick Start Example

Run a full pipeline example via the command line:

```bash
python edge_research/pipeline/main.py params/grid_params.yaml
```

Or check the ready-to-run examples in the [`examples/`](./examples/) directory.

---

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

- ✅ Free for personal, student, and academic use (with citation)
- 💼 Commercial use requires approval (temporarily waived)
- 🔒 No redistribution without permission

See [`LICENSE`](./LICENSE) for full terms.

![License: ERPUL](https://img.shields.io/badge/license-ERPUL-blue)
