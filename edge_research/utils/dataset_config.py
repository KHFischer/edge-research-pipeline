import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

def validate_datasets(
    df: pd.DataFrame,
    hloc_data: pd.DataFrame,
    params_dict: Dict[str, any],
    verbose: bool = True,
    strict: bool = True
):
    """
    Validates that df and hloc_data comply with expected structure and content.
    
    Raises ValueError if validation fails.
    """

    if verbose:
        logger = logging.getLogger("edge_research")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
    else:
        logger = None

    errors = []
    warnings = []

    # Extract config parameters
    required_cols = params_dict['id_cols'] + [params_dict['date_col']]
    price_col = params_dict['price_col']
    drop_cols = params_dict['drop_cols']
    date_col = params_dict['date_col']

    # 1. Check required columns in df
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"[df] Missing required column: '{col}'")

    # 2. Check required columns in hloc_data
    for col in required_cols:
        if col not in hloc_data.columns:
            errors.append(f"[hloc_data] Missing required column: '{col}'")

    # 3. Check price_col in hloc_data
    if price_col not in hloc_data.columns:
        errors.append(f"[hloc_data] Missing price column: '{price_col}'")
    else:
        # Attempt to convert price_col to numeric
        if not np.issubdtype(hloc_data[price_col].dtype, np.number):
            try:
                hloc_data[price_col] = pd.to_numeric(hloc_data[price_col])
                if logger:
                    logger.info(f"[hloc_data] Converted '{price_col}' to numeric.")
            except Exception:
                errors.append(f"[hloc_data] Column '{price_col}' could not be converted to numeric.")

    # 4. Check drop_cols in df
    for col in drop_cols:
        if col not in df.columns:
            msg = f"[df] Config drop_col '{col}' not found in df."
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)

    # 5. Check date_col convertibility in df
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception:
        errors.append(f"[df] '{date_col}' could not be converted to datetime.")

    if df[date_col].isnull().any():
        errors.append(f"[df] '{date_col}' contains null values after datetime conversion.")

    # 6. Check date_col convertibility in hloc_data
    try:
        hloc_data[date_col] = pd.to_datetime(hloc_data[date_col], errors="raise")
    except Exception:
        errors.append(f"[hloc_data] '{date_col}' could not be converted to datetime.")

    if hloc_data[date_col].isnull().any():
        errors.append(f"[hloc_data] '{date_col}' contains null values after datetime conversion.")

    # 7. Check that feature_cols is not empty
    feature_cols = [col for col in df.columns if col not in required_cols + drop_cols]
    if not feature_cols:
        errors.append("[df] No feature columns remain after excluding required_cols and drop_cols.")

    # Raise if any errors were collected
    if errors:
        error_msg = "\n".join(errors)
        if logger:
            logger.error("Dataset validation failed:\n" + error_msg)
        raise ValueError("Dataset validation failed:\n" + error_msg)

    # Optionally log warnings
    if warnings and logger:
        for w in warnings:
            logger.warning(w)

    if logger:
        logger.info("Dataset validation passed successfully.")