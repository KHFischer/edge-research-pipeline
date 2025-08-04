from pathlib import Path
import pandas as pd
from typing import Tuple
import edge_research

def load_samples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load example sample datasets for quick testing and demos.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - hloc_sample: synthetic HLOC time series dataset
        - fundamentals_sample: synthetic fundamentals dataset

    Notes
    -----
    Assumes data is located in `./data/` relative to project root.
    Used in examples, tests, and notebooks for reproducibility.
    """
    data_path = Path(edge_research.__path__[0]) / "data"
    hloc_sample = pd.read_parquet(data_path / "hloc_sample.parquet")
    fundamentals_sample = pd.read_parquet(data_path / "fundamentals_sample.parquet")
    return hloc_sample, fundamentals_sample
    