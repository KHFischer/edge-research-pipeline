import pytest
import pandas as pd
from pathlib import Path
import shutil
import tempfile
import types
import yaml
import os

from edge_research.config_validator import (
    edge_research_pipeline,
    grid_edge_research_pipeline
)

# Test for edge_research_pipeline()
@pytest.fixture
def temp_yaml_files(tmp_path):
    # Create two minimal YAML files for default and custom params
    default_yaml = tmp_path / "default_params.yaml"
    custom_yaml = tmp_path / "custom_params.yaml"
    default_yaml.write_text("run_name: test_run\nlog_markdown: true\nlog_json: false\n")
    custom_yaml.write_text("")  # Empty custom for override/no-op
    return str(default_yaml), str(custom_yaml)

@pytest.fixture
def temp_data_files(tmp_path):
    # Create two simple CSV files for features and hloc
    feature_file = tmp_path / "features.csv"
    hloc_file = tmp_path / "hloc.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(feature_file, index=False)
    pd.DataFrame({"ticker": ["X", "Y"], "date": ["2020-01-01", "2020-01-02"], "adj_close": [10, 20]}).to_csv(hloc_file, index=False)
    return str(feature_file), str(hloc_file)

@pytest.fixture
def patch_dependencies(monkeypatch):
    # Patch all pipeline dependencies with fakes/minimal mocks
    monkeypatch.setattr("my_module.load_params", lambda d, c, verbose=False: {"run_name": "test_run", "log_markdown": True, "log_json": False})
    monkeypatch.setattr("my_module.Config", lambda **kwargs: types.SimpleNamespace(**kwargs))
    monkeypatch.setattr("my_module.load_table", lambda path: pd.DataFrame({"col": [1, 2]}))
    monkeypatch.setattr("my_module.PipelineLogger", lambda log_path, log_markdown, log_json: None)
    monkeypatch.setattr("my_module.copy_yaml_flat", lambda src, dst: Path(dst) / Path(src).name)
    # All the following just return tuple of empty DataFrames/dicts for minimal interface
    monkeypatch.setattr("my_module.train_test_pipeline", lambda *a, **kw: (pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}), {}))
    monkeypatch.setattr("my_module.wfa_pipeline", lambda *a, **kw: (pd.DataFrame({"w": [1]}), pd.DataFrame({"x": [2]}), {}))
    monkeypatch.setattr("my_module.clean_pipeline", lambda *a, **kw: (pd.DataFrame({"y": [1]}), {}))
    monkeypatch.setattr("my_module.engineer_pipeline", lambda *a, **kw: (pd.DataFrame({"z": [1]}), {}))
    monkeypatch.setattr("my_module.target_pipeline", lambda *a, **kw: (pd.DataFrame({"t": [1]}), {}))
    monkeypatch.setattr("my_module.data_prep_pipeline", lambda *a, **kw: (pd.DataFrame({"u": [1]}), {}))
    monkeypatch.setattr("my_module.bootstrap_pipeline", lambda *a, **kw: (pd.DataFrame({"boot": [1]}), pd.DataFrame({"b": [2]})))
    monkeypatch.setattr("my_module.mining_pipeline", lambda *a, **kw: (pd.DataFrame({"mine": [1]}), pd.DataFrame({"rules": [2]}), {}))
    monkeypatch.setattr("my_module.null_pipeline", lambda *a, **kw: (pd.DataFrame({"null": [1]}), pd.DataFrame({"n": [2]})))
    monkeypatch.setattr("my_module.fdr_pipeline", lambda *a, **kw: (pd.DataFrame({"fdr": [1]}), pd.DataFrame({"l": [2]})))
    monkeypatch.setattr("my_module.save_table", lambda df, path, filetype: None)

@pytest.mark.parametrize(
    "kwargs,expected_keys",
    [
        ({"to_train_test": True, "to_wfa": False, "to_bootstrap": False, "to_null_fdr": False}, ["train_test_results"]),
        ({"to_train_test": False, "to_wfa": True, "to_bootstrap": False, "to_null_fdr": False}, ["wfa_results"]),
        ({"to_train_test": False, "to_wfa": False, "to_bootstrap": True, "to_null_fdr": False}, ["bootstrap_results"]),
        ({"to_train_test": False, "to_wfa": False, "to_bootstrap": False, "to_null_fdr": True}, ["mining_results", "null_df", "fdr_res"]),
        ({"to_train_test": True, "to_wfa": True, "to_bootstrap": True, "to_null_fdr": True}, ["train_test_results", "wfa_results", "bootstrap_results", "mining_results", "null_df", "fdr_res"])
    ]
)
def test_pipeline_outputs_and_keys(
    temp_yaml_files, temp_data_files, patch_dependencies, kwargs, expected_keys
):
    default_params, custom_params = temp_yaml_files
    feature_path, hloc_path = temp_data_files
    results, logs = edge_research_pipeline(
        default_params=default_params,
        custom_params=custom_params,
        feature_path=feature_path,
        hloc_path=hloc_path,
        res_save_path=tempfile.mkdtemp(),
        verbose=False,
        **kwargs
    )
    # Check expected result keys present
    for key in expected_keys:
        assert key in results

def test_pipeline_raises_on_no_tests_enabled(temp_yaml_files, temp_data_files, patch_dependencies):
    default_params, custom_params = temp_yaml_files
    feature_path, hloc_path = temp_data_files
    with pytest.raises(ValueError, match="No validation steps enabled"):
        edge_research_pipeline(
            to_train_test=False,
            to_wfa=False,
            to_bootstrap=False,
            to_null_fdr=False,
            default_params=default_params,
            custom_params=custom_params,
            feature_path=feature_path,
            hloc_path=hloc_path,
            res_save_path=tempfile.mkdtemp(),
            verbose=False
        )

def test_pipeline_output_types(temp_yaml_files, temp_data_files, patch_dependencies):
    default_params, custom_params = temp_yaml_files
    feature_path, hloc_path = temp_data_files
    results, logs = edge_research_pipeline(
        to_train_test=True,
        to_wfa=True,
        default_params=default_params,
        custom_params=custom_params,
        feature_path=feature_path,
        hloc_path=hloc_path,
        res_save_path=tempfile.mkdtemp(),
        verbose=False
    )
    # All results should be DataFrames, all logs can be any type (dict, DataFrame, etc.)
    for v in results.values():
        assert isinstance(v, pd.DataFrame)
    assert isinstance(logs, dict)

def test_pipeline_with_invalid_filetype(temp_yaml_files, temp_data_files, patch_dependencies):
    default_params, custom_params = temp_yaml_files
    feature_path, hloc_path = temp_data_files
    # The code does not explicitly error on unknown filetype but should not crash
    results, logs = edge_research_pipeline(
        to_train_test=True,
        default_params=default_params,
        custom_params=custom_params,
        feature_path=feature_path,
        hloc_path=hloc_path,
        res_save_path=tempfile.mkdtemp(),
        res_filetype="unsupported",
        verbose=False
    )
    assert "train_test_results" in results

def test_pipeline_creates_output_folder(temp_yaml_files, temp_data_files, patch_dependencies):
    import os
    default_params, custom_params = temp_yaml_files
    feature_path, hloc_path = temp_data_files
    temp_dir = tempfile.mkdtemp()
    results, logs = edge_research_pipeline(
        to_train_test=True,
        default_params=default_params,
        custom_params=custom_params,
        feature_path=feature_path,
        hloc_path=hloc_path,
        res_save_path=temp_dir,
        verbose=False
    )
    # The expected run_name is test_run
    expected = Path(temp_dir) / "test_run"
    assert expected.exists() and expected.is_dir()

# Test for grid_edge_research_pipeline()
# ---- Mock helpers ----
def dummy_generate_param_grid(param_space):
    # Simple product: e.g. {"a":[1,2],"b":[3]} → [{"a":1,"b":3}, {"a":2,"b":3}]
    keys = list(param_space.keys())
    return [dict(zip(keys, vals)) for vals in zip(*[param_space[k] for k in keys])]

def dummy_run_single_grid_config(i, params, grid_params):
    # Return predictable mock result for testing output shape
    return ({"result": i, "params": params}, {"log": i})

# Monkeypatch for testing (normally would use mocker fixture or monkeypatch)
@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    monkeypatch.setattr("my_module.generate_param_grid", dummy_generate_param_grid)
    monkeypatch.setattr("my_module._run_single_grid_config", dummy_run_single_grid_config)

# ---- Fixtures for temp config ----
@pytest.fixture
def temp_yaml_config():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
        path = tmp.name
        yield path
    os.remove(path)

# ---- Tests ----

def test_grid_pipeline_normal_input(temp_yaml_config):
    config = {
        "param_space": {"x": [1, 2], "y": [3, 4]},
        "n_jobs": 1,
        "base_run_name": "test_run",
        "to_train_test": True,
        "to_wfa": False,
        "to_bootstrap": False,
        "to_null_fdr": False,
        "default_params": {},
        "feature_path": "",
        "hloc_path": "",
        "res_save_path": "",
        "res_filetype": "csv",
        "verbose": False,
    }
    with open(temp_yaml_config, "w") as f:
        yaml.dump(config, f)

    results = grid_edge_research_pipeline(temp_yaml_config)
    assert isinstance(results, list)
    assert len(results) == 2  # Our dummy_generate_param_grid produces 2 configs
    for res, log in results:
        assert isinstance(res, dict)
        assert isinstance(log, dict)
        assert "result" in res
        assert "params" in res

def test_grid_pipeline_parallel_input(temp_yaml_config):
    config = {
        "param_space": {"x": [1], "y": [2]},
        "n_jobs": 2,
        "base_run_name": "parallel",
        "to_train_test": True,
        "to_wfa": False,
        "to_bootstrap": False,
        "to_null_fdr": False,
        "default_params": {},
        "feature_path": "",
        "hloc_path": "",
        "res_save_path": "",
        "res_filetype": "csv",
        "verbose": False,
    }
    with open(temp_yaml_config, "w") as f:
        yaml.dump(config, f)
    results = grid_edge_research_pipeline(temp_yaml_config)
    assert len(results) == 1
    res, log = results[0]
    assert res["params"]["x"] == 1
    assert res["params"]["y"] == 2

@pytest.mark.parametrize(
    "bad_config,expected_exc",
    [
        ({}, ValueError),  # Missing param_space
        ({"param_space": []}, ValueError),  # param_space wrong type
        ({"param_space": {"a": [1]}, "n_jobs": 0}, ValueError),  # n_jobs < 1
        ({"param_space": {"a": [1]}, "n_jobs": -2}, ValueError),  # n_jobs negative
        ({"param_space": {"a": [1]}, "n_jobs": "two"}, ValueError),  # n_jobs wrong type
    ]
)
def test_grid_pipeline_invalid_input(temp_yaml_config, bad_config, expected_exc):
    # Add required dummy values to avoid KeyError in pipeline under test
    bad_config.setdefault("base_run_name", "bad")
    bad_config.setdefault("to_train_test", True)
    bad_config.setdefault("to_wfa", False)
    bad_config.setdefault("to_bootstrap", False)
    bad_config.setdefault("to_null_fdr", False)
    bad_config.setdefault("default_params", {})
    bad_config.setdefault("feature_path", "")
    bad_config.setdefault("hloc_path", "")
    bad_config.setdefault("res_save_path", "")
    bad_config.setdefault("res_filetype", "csv")
    bad_config.setdefault("verbose", False)
    with open(temp_yaml_config, "w") as f:
        yaml.dump(bad_config, f)
    with pytest.raises(expected_exc):
        grid_edge_research_pipeline(temp_yaml_config)

def test_grid_pipeline_empty_param_space(temp_yaml_config):
    config = {
        "param_space": {},
        "n_jobs": 1,
        "base_run_name": "empty",
        "to_train_test": True,
        "to_wfa": False,
        "to_bootstrap": False,
        "to_null_fdr": False,
        "default_params": {},
        "feature_path": "",
        "hloc_path": "",
        "res_save_path": "",
        "res_filetype": "csv",
        "verbose": False,
    }
    with open(temp_yaml_config, "w") as f:
        yaml.dump(config, f)
    results = grid_edge_research_pipeline(temp_yaml_config)
    # Our dummy param_grid returns empty if param_space is empty
    assert results == []

def test_grid_pipeline_one_config(temp_yaml_config):
    config = {
        "param_space": {"only": [42]},
        "n_jobs": 1,
        "base_run_name": "one",
        "to_train_test": True,
        "to_wfa": False,
        "to_bootstrap": False,
        "to_null_fdr": False,
        "default_params": {},
        "feature_path": "",
        "hloc_path": "",
        "res_save_path": "",
        "res_filetype": "csv",
        "verbose": False,
    }
    with open(temp_yaml_config, "w") as f:
        yaml.dump(config, f)
    results = grid_edge_research_pipeline(temp_yaml_config)
    assert len(results) == 1
    res, log = results[0]
    assert res["params"]["only"] == 42

