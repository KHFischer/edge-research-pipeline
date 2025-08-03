import pandas as pd
import numpy as np
from scripts.cleaning_engineering.cleaning import clean_main
from scripts.cleaning_engineering.engineering import engineer_main
from scripts.cleaning_engineering.target import target_returns_main
from scripts.utils.utils import one_hot_encode_features
from params.config_validator import Config
from scripts.utils.central_logger import MarkdownLogger

def prepare_data_main(
    df: pd.DataFrame, 
    hloc_data: pd.DataFrame, 
    cfg: Config, 
    logger: MarkdownLogger
) -> pd.DataFrame:
    
    # Clean data
    df = clean_main(
        df=df,
        logger=logger,
        id_cols=cfg.id_cols, 
        date_col=cfg.date_col,
        to_drop_high_missingness=cfg.to_drop_high_missingness,
        missingness_row_thresh=cfg.missingness_row_thresh,
        missingness_col_thresh=cfg.missingness_col_thresh,
        to_impute=cfg.to_impute,
        max_imputed=cfg.max_imputed,
        to_winsorize=cfg.to_winsorize,
        winsor_cols=cfg.winsor_cols,
        winsor_grouping=cfg.winsor_grouping,
        winsor_dt_units=cfg.winsor_dt_units,
        winsor_lower_quantile=cfg.winsor_lower_quantile,
        winsor_upper_quantile=cfg.winsor_upper_quantile,
        to_z_normalize=cfg.to_z_normalize,
        z_cols=cfg.z_cols,
        z_grouping=cfg.z_grouping,
        z_dt_units=cfg.z_dt_units,
        to_drop_var=cfg.to_drop_var,
        variance_threshold=cfg.variance_threshold,
        to_drop_corr=cfg.to_drop_corr,
        correlation_threshold=cfg.correlation_threshold
    )
    
    # Engineer features + binning
    df = engineer_main(
        df=df,
        logger=logger, 
        id_cols=cfg.id_cols, 
        date_col=cfg.date_col,
        to_generate_ratios=cfg.to_generate_ratios,
        to_generate_lags=cfg.to_generate_lags,
        lag_periods=cfg.lag_periods, 
        bin_grouping=cfg.bin_grouping, 
        bin_dt_units=cfg.bin_dt_units
    )
    
    # Create and encode target column
    df = target_returns_main(
        df=df,
        hloc_data=hloc_data,
        logger=logger, 
        id_cols=cfg.id_cols, 
        date_col=cfg.date_col,
        price_col=cfg.price_col,
        lookforward_dt_units=cfg.lookforward_dt_units,
        target_binning_method=cfg.target_binning_method,
        target_bins=cfg.target_bins,
        target_labels=cfg.target_labels,
        target_grouping=cfg.target_grouping,
        target_dt_units=cfg.target_dt_units
    )
    
    # One hot encode feature columns
    df = one_hot_encode_features(
        df=df,
        id_cols=cfg.id_cols,
        date_col=cfg.date_col,
        drop_cols=cfg.drop_cols,
    )

    # Ensure features are encoded 0/1
    features = [col for col in df.columns if col not in [cfg.date_col] + cfg.id_cols + ['return']]
    df[features] = df[features].astype(bool).astype('uint8')
    
    return df