import os
import sys

# Get the directory two levels up from this script (i.e., edge_research_notebook/)
THIS_FILE = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
sys.path.insert(0, SRC_ROOT)

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Literal, Mapping, List
import datetime
import shutil
import yaml
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from itertools import product

"""
Some libraries print long warnings that are unrelated to the edge research pipeline.
Set SUPPRESS to False if you wish to see them.
Else they are suppressed. 
"""
import warnings
import contextlib
import io
@contextlib.contextmanager
def suppress_import_warnings():
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        # Suppress stderr (KeOps prints go here)
        with contextlib.redirect_stderr(io.StringIO()):
            yield

# conditionally suppress at import time
SUPPRESS = True
if SUPPRESS:
    with suppress_import_warnings():
        os.environ["KEOPS_VERBOSE"] = "0"
        from params.config_validator import load_params, Config
        from scripts.logger.logger import PipelineLogger
        from scripts.preprocessing.cleaning import clean_pipeline
        from scripts.preprocessing.engineering import engineer_pipeline
        from scripts.preprocessing.target import target_pipeline
        from scripts.rules_mining.mining import data_prep_pipeline, mining_pipeline
        from scripts.validation_tests.validation import train_test_pipeline, wfa_pipeline, bootstrap_pipeline, null_pipeline, fdr_pipeline

def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a table from CSV or Parquet based on file extension.

    Parameters
    ----------
    path : str or Path
        Path to the file. Must end in .csv or .parquet

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (only .csv or .parquet allowed)")

def save_table(df: pd.DataFrame, path: Union[str, Path], filetype: Literal["csv", "parquet"]) -> None:
    """
    Save a DataFrame to CSV or Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Output path (extension will be replaced based on `filetype`).
    filetype : 'csv' or 'parquet'
        Format to save in.
    """
    path = Path(path)
    path = path.with_suffix(f".{filetype}")

    if filetype == "csv":
        df.to_csv(path, index=False)
    elif filetype == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")


def copy_yaml_to_output(source_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """
    Copy a YAML file to a new output directory for traceability.

    Parameters
    ----------
    source_path : str or Path
        Path to the original YAML file.
    output_dir : str or Path
        Target directory to copy the YAML file into.

    Returns
    -------
    Path
        Destination path of the copied YAML file.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"YAML config not found: {source_path}")
    if source_path.suffix.lower() != ".yaml":
        raise ValueError(f"Expected .yaml file, got: {source_path.name}")

    dest_path = output_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    return dest_path



def save_params_to_yaml(params: Mapping, path: Union[str, Path]) -> None:
    """
    Save a merged parameter dictionary to a YAML file.

    Parameters
    ----------
    params : dict-like
        Merged config dictionary.
    path : str or Path
        Output file path for saving the YAML.
    """
    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(dict(params), f, sort_keys=False)


def edge_research_pipeline(
    to_train_test: bool = True,
    to_wfa: bool = True,
    to_bootstrap: bool = True,
    to_null_fdr: bool = True,
    default_params: Union[str, Path] = "params/default_params.yaml",
    custom_params: Union[str, Path] = "params/custom_params.yaml",
    feature_path: Union[str, Path] = "./data/fundamentals_sample.parquet",
    hloc_path: Union[str, Path] = "./data/hloc_sample.parquet",
    res_save_path: Union[str, Path] = "data/results",
    res_filetype: str = "csv",
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Run the full edge research validation pipeline as configured in YAML files.

    Supports modular execution of Train/Test, Walk-Forward Analysis (WFA), Bootstrap, and Null/FDR validation steps.
    Saves all config files and results for full traceability.

    Parameters
    ----------
    to_train_test : bool, optional
        Run Train/Test validation step. Default is True.
    to_wfa : bool, optional
        Run Walk-Forward Analysis validation step. Default is True.
    to_bootstrap : bool, optional
        Run Bootstrap resampling validation step. Default is True.
    to_null_fdr : bool, optional
        Run Null/FDR (False Discovery Rate) validation step. Default is True.
    default_params : str or Path, optional
        Path to default YAML parameter file.
    custom_params : str or Path, optional
        Path to custom YAML parameter file (overrides defaults).
    feature_path : str or Path, optional
        Path to feature dataset (.csv or .parquet).
    hloc_path : str or Path, optional
        Path to HLOC (High/Low/Open/Close) dataset (.csv or .parquet).
    res_save_path : str or Path, optional
        Directory where all results and logs will be saved.
    res_filetype : str, optional
        File format for results ('csv' or 'parquet'). Default is 'csv'.
    verbose : bool, optional
        Print progress updates. Default is True.

    Returns
    -------
    results : dict
        Mapping of result type to DataFrames.
    logs : dict
        Mapping of log type to logs and auxiliary outputs.

    Raises
    ------
    ValueError
        If no validation steps are enabled.
    FileNotFoundError
        If provided config or data files are missing.

    Examples
    --------
    >>> edge_research_pipeline(
    ...     to_train_test=True,
    ...     to_wfa=False,
    ...     res_filetype="parquet"
    ... )
    """

    def _print(msg: str) -> None:
        if verbose:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] {msg}")

    # --- Setup ---
    _print("Starting pipeline...")

    # Load config
    params = load_params(default_params, custom_params, verbose=False)
    cfg = Config(**params)

    # Load input data
    hloc_df = load_table(hloc_path)
    feature_df = load_table(feature_path)

    # Create results directory and logger
    run_name = getattr(cfg, "run_name", "default_run")
    log_markdown = getattr(cfg, "log_markdown", True)
    log_json = getattr(cfg, "log_json", False)
    save_folder = Path(res_save_path) / run_name
    save_folder.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(
        log_path=save_folder / f"{run_name}_log",
        log_markdown=log_markdown,
        log_json=log_json
    )

    # Save YAML configs for traceability
    save_params_to_yaml(params, Path(save_folder) / "merged_config.yaml")

    _print("Setup completed.")

    if not any([to_train_test, to_wfa, to_bootstrap, to_null_fdr]):
        raise ValueError("No validation steps enabled. Enable at least one of: to_train_test, to_wfa, to_bootstrap, to_null_fdr.")

    results = {}
    logs = {}

    # --- Train/Test Validation ---
    if to_train_test:
        _print("Performing Train/Test validation...")
        tt_results, tt_log, tt_pipeline_logs = train_test_pipeline(feature_df, hloc_df, cfg, logger)
        results["train_test_results"] = tt_results
        logs["train_test_log"] = tt_log
        logs["train_test_pipeline_logs"] = tt_pipeline_logs
        save_table(tt_results, save_folder / "train_test_results", res_filetype)
        save_table(tt_log, save_folder / "train_test_log", res_filetype)
        _print("Train/Test validation complete.")

    # --- Walk-Forward Analysis (WFA) ---
    if to_wfa:
        _print("Performing Walk-Forward Analysis (WFA) validation...")
        wfa_results, wfa_log, wfa_pipeline_logs = wfa_pipeline(feature_df, hloc_df, cfg, logger)
        results["wfa_results"] = wfa_results
        logs["wfa_log"] = wfa_log
        logs["wfa_pipeline_logs"] = wfa_pipeline_logs
        save_table(wfa_results, save_folder / "wfa_results", res_filetype)
        save_table(wfa_log, save_folder / "wfa_log", res_filetype)
        _print("WFA validation complete.")

    # --- Bootstrap & Null/FDR Validation (share prepped data) ---
    if to_bootstrap or to_null_fdr:
        _print("Preparing data for Bootstrap/Null/FDR validation...")
        df_cleaned, clean_logs = clean_pipeline(feature_df, cfg, logger)
        df_engineered, eng_logs = engineer_pipeline(df_cleaned, cfg, logger)
        df_target, target_logs = target_pipeline(df_engineered, cfg, hloc_df, logger)
        df_onehot, prep_logs = data_prep_pipeline(df_target, cfg, logger)

        if to_bootstrap:
            _print("Performing Bootstrap validation...")
            bootstrap_results, bootstrap_log = bootstrap_pipeline(df_onehot, cfg, logger)
            results["bootstrap_results"] = bootstrap_results
            logs["bootstrap_log"] = bootstrap_log
            save_table(bootstrap_results, save_folder / "bootstrap_results", res_filetype)
            save_table(bootstrap_log, save_folder / "bootstrap_log", res_filetype)
            _print("Bootstrap validation complete.")

        if to_null_fdr:
            _print("Performing rule mining for FDR...")
            mining_res, rules_df, mining_logs = mining_pipeline(df_onehot, cfg, logger)
            results["mining_results"] = mining_res
            logs["mining_logs"] = mining_logs
            save_table(mining_res, save_folder / "mining_results", res_filetype)

            _print("Generating null distribution...")
            null_df, null_log = null_pipeline(df_onehot, cfg, logger)
            results["null_df"] = null_df
            logs["null_log"] = null_log
            save_table(null_df, save_folder / "null_df", res_filetype)
            save_table(null_log, save_folder / "null_log", res_filetype)

            _print("Performing FDR multiple testing correction...")
            fdr_res, fdr_log = fdr_pipeline(mining_res, null_df, cfg, logger)
            results["fdr_res"] = fdr_res
            logs["fdr_log"] = fdr_log
            save_table(fdr_res, save_folder / "fdr_res", res_filetype)
            save_table(fdr_log, save_folder / "fdr_log", res_filetype)

    _print(f"Pipeline finished. Results saved to: {save_folder}")

    return results, logs

def generate_param_grid(param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from a dictionary of param: list-of-values.
    Always returns grid in sorted key order for determinism.
    """
    keys = sorted(param_space.keys())  # <--- Sort keys for deterministic order
    values_product = product(*(param_space[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in values_product]

from pathlib import Path
from typing import Any, Dict, Tuple

def _run_single_grid_config(i: int, params: Dict[str, Any], grid_params: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Runs a single grid configuration for the research pipeline unless output folder already exists.
    If the target results folder exists, skips execution and returns empty results/logs.

    Parameters
    ----------
    i : int
        Index of the current grid iteration (for run naming).
    params : dict
        Parameter overrides for this run.
    grid_params : dict
        Full grid and global config.

    Returns
    -------
    results, logs : Tuple[Any, Any]
        Results and logs as returned by edge_research_pipeline, or empty dicts if skipped.
    """
    # Defensive copy to avoid mutating shared objects
    params = dict(params)
    params['run_name'] = f"{grid_params['base_run_name']}_{i+1}"

    # Check if results folder exists; skip if so
    save_folder = Path(grid_params['res_save_path']) / params['run_name']
    if save_folder.exists():
        print(f"[SKIP] Iteration {i}: Run '{params['run_name']}' skipped (results exist at: {save_folder}).")
        return {}, {}

    # Otherwise, run the pipeline
    results, logs = edge_research_pipeline(
        to_train_test=grid_params['to_train_test'],
        to_wfa=grid_params['to_wfa'],
        to_bootstrap=grid_params['to_bootstrap'],
        to_null_fdr=grid_params['to_null_fdr'],
        default_params=grid_params['default_params'],
        custom_params=params,
        feature_path=grid_params['feature_path'],
        hloc_path=grid_params['hloc_path'],
        res_save_path=grid_params['res_save_path'],
        res_filetype=grid_params['res_filetype'],
        verbose=grid_params['verbose'],
    )
    return results, logs

def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    # Resolve relative to project root, not config file
    return os.path.normpath(os.path.join(SRC_ROOT, path))
    
def grid_edge_research_pipeline(grid_path: str) -> List[Tuple[Any, Any]]:
    """
    Orchestrates a grid search over parameter combinations for the edge research pipeline,
    with optional parallel execution and basic start/stop (idempotent) support.

    Loads a grid specification from a YAML file, generates all parameter combinations,
    and executes the research pipeline for each configuration—serially or in parallel
    depending on `n_jobs`.

    If a run's result folder (determined by base_run_name and index) already exists,
    that configuration will be skipped, enabling users to safely restart interrupted
    or partially completed grid runs without re-executing finished jobs.

    Limitation:
      - The pipeline only checks for the existence of the final save folder. If a folder
        exists but is incomplete or corrupted (e.g., run failed mid-execution), the run
        will be skipped and must be manually cleaned up for a true rerun.

    Parameters
    ----------
    grid_path : str
        Path to YAML file containing:
          - param_space: dict of parameter lists for grid expansion
          - n_jobs: int (number of parallel workers; 1 disables parallelism)
          - base_run_name, and other required pipeline settings

    Returns
    -------
    List[Tuple[Any, Any]]
        List of (results, logs) tuples from each pipeline run. Skipped runs return empty dicts.
    """

    # Load and validate grid parameters
    with open(grid_path, "r") as f:
        grid_params = yaml.safe_load(f) or {}
    
    # Ensure valid paths
    for key in ["default_params", "custom_params", "feature_path", "hloc_path", "res_save_path"]:
        if key in grid_params:
            grid_params[key] = resolve_path(grid_params[key])

    if 'param_space' not in grid_params or not isinstance(grid_params['param_space'], dict):
        raise ValueError("Missing or invalid 'param_space' in grid configuration.")
    if 'n_jobs' in grid_params and (not isinstance(grid_params['n_jobs'], int) or grid_params['n_jobs'] < 1):
        raise ValueError("'n_jobs' must be a positive integer if specified.")

    grid = generate_param_grid(grid_params['param_space'])
    n_jobs = grid_params.get('n_jobs', 1)

    # Main execution
    if n_jobs == 1:
        results_list = []
        for i, params in tqdm(list(enumerate(grid)), total=len(grid), desc="Testing Param Grid"):
            result = _run_single_grid_config(i, params, grid_params)
            results_list.append(result)
    else:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_grid_config)(i, params, grid_params)
            for i, params in enumerate(grid)
        )
    return results_list

