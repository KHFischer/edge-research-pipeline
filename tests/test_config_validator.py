import pytest
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict
import tempfile
from collections import ChainMap

from edge_research.config_validator import (
    config_validator,
    load_params
)

# Test for config_validator()
# Dummy helpers to replace external validators in test scope
def is_all_or_list_of_strings(params_dict: Dict[str, Any], param_key: str):
    val = params_dict[param_key]
    if val != "all" and not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
        raise ValueError("Must be 'all' or a list of strings.")

def is_in_choices(valid_options, *_):
    def validator(value, *_):
        if value not in valid_options:
            raise ValueError(f"Value must be one of {valid_options}, got {value}")
    return validator

def is_valid_custom_bins(value):
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
        raise ValueError("Invalid custom bins.")

def is_valid_quantile_bins(value):
    if not isinstance(value, list) or sorted(value) != value:
        raise ValueError("Quantile bins must be sorted list.")

def is_valid_bin_labels(labels, edges):
    if len(labels) != len(edges) - 1:
        raise ValueError("Labels length must be one less than edges length.")

# Monkeypatch for test scope
import builtins
builtins.is_all_or_list_of_strings = is_all_or_list_of_strings
builtins.is_in_choices = is_in_choices
builtins.is_valid_custom_bins = is_valid_custom_bins
builtins.is_valid_quantile_bins = is_valid_quantile_bins
builtins.is_valid_bin_labels = is_valid_bin_labels


@pytest.mark.parametrize("params_dict,groups", [
    (
        {"foo": 123, "bar": "hello"},
        {"int": ["foo"], "string": ["bar"]}
    ),
    (
        {"mylist": ["a", "b"]},
        {"list": ["mylist"]}
    ),
    (
        {"opt": "all"},
        {"all_or_list": ["opt"]}
    )
])
def test_valid_configs_pass(params_dict, groups):
    # Should not raise
    config_validator(params_dict, groups)


def test_invalid_type_raises():
    params = {"foo": "not_an_int"}
    groups = {"int": ["foo"]}
    with pytest.raises(ValueError, match=r"\[foo\] must be int"):
        config_validator(params, groups)


def test_unknown_param_strict_raises():
    params = {"unknown": 42}
    groups = {"int": ["foo"]}
    with pytest.raises(ValueError, match=r"\[unknown\] unexpected parameter"):
        config_validator(params, groups, strict=True)


def test_unknown_param_strict_false_passes():
    params = {"unknown": 42}
    groups = {"int": ["foo"]}
    # Should not raise
    config_validator(params, groups, strict=False)


def test_val_from_list_validation_fails():
    params = {"model_type": "xgboost"}
    groups = {"val_from_list": [{"model_type": ["rf", "svm"]}]}
    with pytest.raises(ValueError, match=r"\[model_type\]"):
        config_validator(params, groups)


def test_val_from_list_validation_passes():
    params = {"model_type": "rf"}
    groups = {"val_from_list": [{"model_type": ["rf", "svm"]}]}
    config_validator(params, groups)


def test_quantiles_edge_key_custom_valid():
    params = {
        "target_bins": [0, 0.5, 1],
        "target_binning_method": "custom"
    }
    groups = {"quantiles": [{"target_bins": "target_labels"}]}
    config_validator(params, groups)


def test_quantiles_labels_key_mismatch():
    params = {
        "target_bins": [0, 0.5, 1],
        "target_labels": ["low"]
    }
    groups = {"quantiles": [{"target_bins": "target_labels"}]}
    with pytest.raises(ValueError, match=r"\[target_labels\]"):
        config_validator(params, groups)


def test_empty_input_passes():
    config_validator({}, {})


def test_logger_is_used(caplog):
    params = {"foo": 123}
    groups = {"int": ["foo"]}
    logger = logging.getLogger("test_logger")
    with caplog.at_level(logging.DEBUG):
        config_validator(params, groups, logger=logger)
        assert any("Validating param: foo" in m for m in caplog.messages)
        assert any("Configuration validation passed" in m for m in caplog.messages)

# Test for load_params()
# --- Dummy test helpers ---
def dummy_validator(params_dict, logger=None, strict=True):
    pass

def dummy_pretty_printer(params_dict):
    print("Pretty print:", dict(params_dict))


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    monkeypatch.setattr("my_module.config_validator", dummy_validator)
    monkeypatch.setattr("my_module.pretty_print_params", dummy_pretty_printer)


def test_dict_inputs_merge_correctly():
    defaults = {"a": 1, "b": 2}
    custom = {"b": 20, "c": 30}
    result = load_params(defaults, custom)
    assert isinstance(result, ChainMap)
    assert result["a"] == 1
    assert result["b"] == 20
    assert result["c"] == 30


def test_yaml_file_inputs_merge_correctly(tmp_path):
    default_path = tmp_path / "defaults.yaml"
    custom_path = tmp_path / "custom.yaml"
    default_yaml = {"foo": "bar", "x": 1}
    custom_yaml = {"x": 99, "y": 2}

    default_path.write_text(yaml.dump(default_yaml))
    custom_path.write_text(yaml.dump(custom_yaml))

    result = load_params(str(default_path), str(custom_path))
    assert result["foo"] == "bar"
    assert result["x"] == 99
    assert result["y"] == 2


def test_missing_yaml_file_fallback(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    defaults = {"x": 1}
    result = load_params(defaults, str(missing_path))
    assert result["x"] == 1  # fallback still loads default


def test_none_inputs_returns_empty_chainmap():
    result = load_params(None, None)
    assert isinstance(result, ChainMap)
    assert result.maps == [{}, {}]


@pytest.mark.parametrize("invalid_obj", [123, 5.6, ["list"], set([1])])
def test_invalid_input_type_raises(invalid_obj):
    with pytest.raises(ValueError, match="Unsupported parameter type"):
        load_params(invalid_obj, None)


def test_verbose_flag_triggers_pretty_print(capsys):
    defaults = {"x": 1}
    custom = {"x": 2}
    result = load_params(defaults, custom, verbose=True)
    captured = capsys.readouterr()
    assert "Pretty print" in captured.out
    assert result["x"] == 2
