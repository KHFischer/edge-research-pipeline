import pandas as pd
import numpy as np
from typing import List, Optional
from typing import Tuple

from scripts.utils.central_logger import MarkdownLogger
from params.config_validator import Config
from scripts.validation_tests.validation_utils import evaluator_toggled
from scripts.rules_mining.mining_main import run_edge_discovery_pipeline

def shuffle_dataframe(
    df: pd.DataFrame,
    mode: str = "target",
    target_col: Optional[str] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a shuffled version of the dataframe according to the specified mode.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        mode (str): One of 'target', 'rows', or 'columns'.
        target_col (str, optional): Name of the target column to shuffle when mode='target'.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Shuffled dataframe.
    """
    rng = np.random.default_rng(random_state)
    df_copy = df.copy()

    if mode == "target":
        if target_col is None:
            raise ValueError("target_col must be specified when mode='target'.")
        shuffled_target = df_copy[target_col].sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        df_copy[target_col] = shuffled_target
        return df_copy

    elif mode == "rows":
        df_shuffled = df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df_shuffled

    elif mode == "columns":
        for col in df_copy.columns:
            df_copy[col] = rng.permutation(df_copy[col].values)
        return df_copy

    else:
        raise ValueError("mode must be one of 'target', 'rows', or 'columns'.")

def maybe_tqdm(iterable, use_tqdm: bool, **tqdm_kwargs):
    if use_tqdm:
        from tqdm.auto import tqdm
        return tqdm(iterable, **tqdm_kwargs)
    return iterable

def compute_relative_error(
    df: pd.DataFrame,
    metric_col: str,
    iteration_col: str,
    m_recent: int
) -> float:
    """
    Computes the relative error of the metric over the last M estimates.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing metric estimates.
        metric_col (str): Column name with the metric values (e.g., percentile estimates).
        iteration_col (str): Column indicating iteration number (e.g., 'perm_num').
        m_recent (int): Number of most recent estimates to consider.
    
    Returns:
        float: Relative error = std / mean over the last M estimates.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in dataframe.")
    if iteration_col not in df.columns:
        raise ValueError(f"Column '{iteration_col}' not found in dataframe.")
    
    # Sort by iteration to ensure ordering
    df_sorted = df.sort_values(iteration_col).reset_index(drop=True)
    
    if len(df_sorted) < m_recent:
        raise ValueError(f"Not enough estimates: got {len(df_sorted)}, need at least {m_recent}.")
    
    # Get the last M estimates
    recent_values = df_sorted[metric_col].iloc[-m_recent:]
    
    mean_value = np.mean(recent_values)
    std_value = np.std(recent_values)
    
    if mean_value == 0:
        return np.nan  # Avoid division by zero
    
    rel_error = std_value / mean_value
    return rel_error

def summarize_null_distribution(
    null_df: pd.DataFrame,
    metric_col: str,
    iteration_col: str
) -> pd.DataFrame:
    """
    Creates a summary dataframe of null distribution statistics for logging.
    
    Parameters:
        null_df (pd.DataFrame): DataFrame of all null results.
        metric_col (str): Column name of the metric to summarize (e.g., 'lift').
        iteration_col (str): Column name tracking permutation/test number.
        
    Returns:
        pd.DataFrame: Single-row dataframe with summary statistics.
    """
    if metric_col not in null_df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataframe.")
    if iteration_col not in null_df.columns:
        raise ValueError(f"Iteration column '{iteration_col}' not found in dataframe.")
    
    n_permutations = null_df[iteration_col].nunique()
    n_observations = len(null_df)
    
    metric_series = null_df[metric_col]
    
    summary = {
        "metric_mean": metric_series.mean(),
        "metric_std": metric_series.std(),
        "metric_min": metric_series.min(),
        "metric_max": metric_series.max(),
        "metric_q05": metric_series.quantile(0.05),
        "metric_q50": metric_series.quantile(0.50),
        "metric_q95": metric_series.quantile(0.95),
        "n_permutations": n_permutations,
        "n_observations": n_observations
    }
    
    return pd.DataFrame([summary])

def generate_null_distribution(
    df: pd.DataFrame,
    cfg,
    n_null: int,
    shuffle_mode: str,
    early_stop_metric: str,
    es_m_permutations: int,
    rel_error_threshold: float = 0.01,
    mine_rules: bool = True,
    verbose: bool = True,
    logger=None,
    target_col: str = 'return',
) -> Tuple[pd.DataFrame, float]:
    """
    Generates a null distribution of rule metrics with optional early stopping based on relative error.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        cfg: Config object for rule mining.
        n_null (int): Maximum number of permutations.
        shuffle_mode (str): One of 'target', 'rows', 'columns'.
        target_col (str): Name of the target column to shuffle.
        early_stop_metric (str): Column name of the metric to track for convergence.
        es_m_permutations (int): Number of recent estimates to use for relative error.
        rel_error_threshold (float): Threshold for stopping.
        mine_rules (bool): Whether to re-mine rules each iteration.
        verbose (bool): Whether to show tqdm progress.
        logger: Logger instance.
        
    Returns:
        Tuple[pd.DataFrame, float]: (Dataframe of all results, final relative error)
    """
    all_results = []
    
    iterator = maybe_tqdm(
        range(n_null),
        verbose,
        total=n_null,
        desc="Creating Null Distribution"
    )
    
    for i in iterator:
        
        # Shuffle
        shuffled = shuffle_dataframe(
            df=df,
            mode=shuffle_mode,
            target_col=target_col
        )
        
        # Run mining
        res, _, _ = run_edge_discovery_pipeline(
            prepared_df=shuffled,
            cfg=cfg,
            logger=logger,
            mine_rules=mine_rules
        )
        
        res["test_num"] = i
        all_results.append(res)
        
        # Check early stopping every es_m_permutations
        if (i + 1) % es_m_permutations == 0 and (i + 1) >= es_m_permutations:
            interim_df = pd.concat(all_results, ignore_index=True)
            rel_error = compute_relative_error(
                df=interim_df,
                metric_col=early_stop_metric,
                iteration_col="test_num",
                m_recent=es_m_permutations
            )
            if rel_error < rel_error_threshold:
                final_df = interim_df
                null_log = summarize_null_distribution(final_df, early_stop_metric, "test_num")
                logger.log_step(
                    step_name="Null Test",
                    info={
                        "n_null": n_null,
                        "shuffle_mode": shuffle_mode,
                        "target_col": target_col,
                        "early_stop_metric": early_stop_metric,
                        "es_m_permutations": es_m_permutations,
                        "rel_error_threshold": rel_error_threshold,
                        "mine_rules": mine_rules,
                        "final_rel_error": rel_error,
                    },
                    df=null_log
                )
                return final_df, null_log

    # If loop finishes without early stopping
    final_df = pd.concat(all_results, ignore_index=True)
    rel_error = compute_relative_error(
        df=final_df,
        metric_col=early_stop_metric,
        iteration_col="test_num",
        m_recent=es_m_permutations
    )
    null_log = summarize_null_distribution(final_df, early_stop_metric, "test_num")
    null_log['final_rel_error'] = rel_error
    logger.log_step(
        step_name="Null Test",
        info={
            "n_null": n_null,
            "shuffle_mode": shuffle_mode,
            "target_col": target_col,
            "early_stop_metric": early_stop_metric,
            "es_m_permutations": es_m_permutations,
            "rel_error_threshold": rel_error_threshold,
            "mine_rules": mine_rules,
            "final_rel_error": rel_error,
        },
        df=null_log
    )
    return final_df, null_log

def null_main(
    prepared_df: pd.DataFrame, 
    cfg: Config,
    logger: MarkdownLogger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    func = generate_null_distribution
    func_args = {
        "cfg": cfg, 
        "n_null": cfg.n_null, 
        "shuffle_mode": cfg.shuffle_mode, 
        "early_stop_metric": cfg.early_stop_metric, 
        "es_m_permutations": cfg.es_m_permutations, 
        "rel_error_threshold": cfg.rel_error_threshold, 
        "verbose": cfg.null_verbose,
        "logger": logger
    }
    
    null_df, null_log = evaluator_toggled(
        prepared_df, 
        cfg, 
        func, 
        func_args, 
        cfg.null_discovery_mode, 
        logger, 
    )
    return null_df, null_log