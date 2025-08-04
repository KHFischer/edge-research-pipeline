# 📂 data/

This folder contains small sample datasets for testing and exploring the pipeline.

## 📊 Included Files

- `hloc_sample.parquet` — Sample historical price data with HLOC format
- `fundamentals_sample.parquet` — Sample tabular fundamentals/features dataset

## 🎯 Purpose

These datasets are:
- Small (+-1.5MB combined) and self-contained
- Designed to showcase pipeline functionality and expected data formats
- Useful for tutorials, validation tests, and dry runs

## ⚠️ Limitations

- **Do not use for production or decision-making.**
- These files are synthetic or heavily truncated examples.
- They are not statistically valid, complete, or up-to-date.

## 📎 Usage

The pipeline automatically loads these samples if you follow the examples in:
- `docs/pipeline.md`
- `examples/` (if applicable)

You can also load them manually:
```python
from edge_research.utils.utils import load_samples

features, hloc = load_samples()
````

## ✅ Tip

If you're preparing your own data, follow the column formats and dtypes used here as a reference.