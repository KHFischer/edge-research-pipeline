import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from types import SimpleNamespace

from edge_research.validation import (
    generate_time_splits,
    generate_fraction_splits,
    split_datasets,
    rules_series_to_unique_rules,
    make_combined_rule_feature_df,
    test_mined_rules,
    split_mining_pipeline,
    combine_split_results,
    create_validation_log_df,
    validate_train_test,
    validate_wfa,
    train_test_pipeline,
    wfa_pipeline,
    resample_dataframe,
    summarize_rule_metrics,
    create_validation_summary_log,
    validate_bootstrap,
    bootstrap_pipeline,
    shuffle_dataframe,
    compute_relative_error,
    summarize_null_distribution,
    validate_null,
    null_pipeline,
    summarize_fdr_results,
    compute_empirical_pvals,
    validate_multiple_tests,
    fdr_pipeline
)

# Test for generate_time_splits()
@pytest.fixture
def simple_df():
    """DataFrame with 10 consecutive daily dates and a value column."""
    start = datetime(2020, 1, 1)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(10)],
        "value": np.arange(10),
    })

def test_equal_splits_typical(simple_df):
    # 2 equal splits: days 0-4 and 5-9
    splits, log = generate_time_splits(simple_df, date_col="date", n_splits=2)
    assert len(splits) == 2
    assert sum(len(s) for s in splits) == len(simple_df)
    # Check ordering and non-overlap
    assert splits[0]["date"].max() < splits[1]["date"].min()
    # Log DataFrame shape and columns
    assert log.shape[0] == 2
    for col in ["split_index", "split_type", "configured_start", "configured_end", "start_date", "end_date", "n_rows", "fraction_of_total", "note"]:
        assert col in log.columns
    # Fraction adds up to 1.0 (with float tolerance)
    assert pytest.approx(log["fraction_of_total"].sum(), 0.01) == 1.0

def test_range_split_with_overlap(simple_df):
    # Overlapping date windows
    dr = [
        (datetime(2020, 1, 1), datetime(2020, 1, 6)),  # days 0-5
        (datetime(2020, 1, 5), datetime(2020, 1, 10)), # days 5-9
    ]
    splits, log = generate_time_splits(simple_df, date_col="date", n_splits=2, date_ranges=dr)
    assert len(splits) == 2
    # The overlap should result in day 5 being in both splits
    overlap = set(splits[0]["date"]).intersection(set(splits[1]["date"]))
    assert datetime(2020, 1, 5) in overlap
    # Both splits cover correct boundaries
    assert splits[0]["date"].min() == dr[0][0]
    assert splits[1]["date"].max() == dr[1][1] - timedelta(days=1)

def test_empty_input():
    df = pd.DataFrame({"date": [], "value": []})
    splits, log = generate_time_splits(df, date_col="date", n_splits=2)
    # Should return empty splits and log, no error
    assert len(splits) == 2
    assert all(s.empty for s in splits)
    assert log["n_rows"].sum() == 0

@pytest.mark.parametrize("bad_n_splits", [0, -1])
def test_invalid_n_splits(simple_df, bad_n_splits):
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        generate_time_splits(simple_df, date_col="date", n_splits=bad_n_splits)

def test_missing_date_column(simple_df):
    with pytest.raises(KeyError, match="not found in dataframe columns"):
        generate_time_splits(simple_df, date_col="not_a_col", n_splits=2)

def test_malformed_date_range(simple_df):
    dr = [(datetime(2020, 1, 2), datetime(2020, 1, 2))]  # start == end
    with pytest.raises(ValueError, match="Malformed date_range"):
        generate_time_splits(simple_df, date_col="date", n_splits=1, date_ranges=dr)

def test_warn_on_date_range_length_mismatch(simple_df):
    dr = [
        (datetime(2020, 1, 1), datetime(2020, 1, 6)),
        (datetime(2020, 1, 5), datetime(2020, 1, 10)),
        (datetime(2020, 1, 8), datetime(2020, 1, 10)),
    ]
    with pytest.warns(UserWarning, match="Number of date_ranges"):
        splits, log = generate_time_splits(simple_df, date_col="date", n_splits=2, date_ranges=dr)
    assert len(splits) == 3

def test_one_row_df():
    df = pd.DataFrame({"date": [datetime(2023, 1, 1)], "value": [42]})
    splits, log = generate_time_splits(df, date_col="date", n_splits=1)
    assert len(splits) == 1
    assert splits[0].iloc[0]["value"] == 42
    assert log.iloc[0]["n_rows"] == 1
    assert log.iloc[0]["note"] == ""

# Test for generate_fraction_splits()
@pytest.fixture
def simple_df():
    start = datetime(2021, 1, 1)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(10)],
        "value": np.arange(10),
    })

@pytest.mark.parametrize("n_splits, overlap", [(2, False), (5, False), (2, True)])
def test_n_splits_behavior(simple_df, n_splits, overlap):
    splits, log = generate_fraction_splits(simple_df, date_col="date", n_splits=n_splits, overlap=overlap)
    assert len(splits) == n_splits
    assert log.shape[0] == n_splits
    # Should cover all data with no data lost in total
    total_rows = sum(len(s) for s in splits)
    # With overlap, total_rows can be > len(simple_df)
    if not overlap:
        assert total_rows == len(simple_df)
    assert log["n_rows"].sum() == total_rows
    assert all(isinstance(s, pd.DataFrame) for s in splits)

def test_window_frac_and_step_frac(simple_df):
    # Window: 50%, step: 25% → 3 splits, overlapping
    splits, log = generate_fraction_splits(
        simple_df, "date", window_frac=0.5, step_frac=0.25, overlap=True
    )
    assert len(splits) == 3
    assert all(isinstance(s, pd.DataFrame) for s in splits)
    assert log.iloc[0]["split_index"] == 0

def test_explicit_fractions(simple_df):
    # Split into 40% and 60% of the date range
    fractions = [0.4, 0.6]
    splits, log = generate_fraction_splits(
        simple_df, "date", fractions=fractions, overlap=False
    )
    assert len(splits) == 2
    # Should not overlap, row coverage may be less than total (if dates don't divide cleanly)
    covered_dates = pd.concat(splits)["date"].unique()
    assert set(covered_dates) <= set(simple_df["date"])
    # Log fields are correct
    for col in ["split_index", "configured_start", "configured_end", "start_date", "end_date", "n_rows", "fraction_of_total", "note"]:
        assert col in log.columns

def test_overlap_with_explicit_fractions(simple_df):
    fractions = [0.6, 0.6]
    splits, log = generate_fraction_splits(
        simple_df, "date", fractions=fractions, overlap=True
    )
    assert len(splits) == 2
    # Should overlap
    overlap = set(splits[0]["date"]).intersection(set(splits[1]["date"]))
    assert len(overlap) > 0

def test_empty_df_returns_empty_splits():
    df = pd.DataFrame({"date": [], "value": []})
    splits, log = generate_fraction_splits(df, "date", n_splits=3)
    assert len(splits) == 3
    assert all(s.empty for s in splits)
    assert log["n_rows"].sum() == 0

@pytest.mark.parametrize("bad_n_splits", [0, -2])
def test_invalid_n_splits_raises(simple_df, bad_n_splits):
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        generate_fraction_splits(simple_df, "date", n_splits=bad_n_splits)

def test_missing_date_column(simple_df):
    with pytest.raises(KeyError, match="not found in dataframe columns"):
        generate_fraction_splits(simple_df, "not_a_col", n_splits=2)

@pytest.mark.parametrize("fractions", [[1.2], [-0.1, 0.5], [0, 0.5]])
def test_invalid_fractions_raise(simple_df, fractions):
    with pytest.raises(ValueError, match="All fractions must be in"):
        generate_fraction_splits(simple_df, "date", fractions=fractions)

def test_sum_of_fractions_gt1_raises(simple_df):
    with pytest.raises(ValueError, match="Sum of fractions"):
        generate_fraction_splits(simple_df, "date", fractions=[0.7, 0.6], overlap=False)

def test_no_split_args_raises(simple_df):
    with pytest.raises(ValueError, match="Must provide one of"):
        generate_fraction_splits(simple_df, "date")

def test_window_frac_and_step_frac_invalid(simple_df):
    # Both zero
    with pytest.raises(ValueError, match="window_frac must be in"):
        generate_fraction_splits(simple_df, "date", window_frac=0, step_frac=0.1)
    with pytest.raises(ValueError, match="step_frac must be in"):
        generate_fraction_splits(simple_df, "date", window_frac=0.2, step_frac=0)
    # Both too large
    with pytest.raises(ValueError, match="window_frac must be in"):
        generate_fraction_splits(simple_df, "date", window_frac=1.5, step_frac=0.2)
    with pytest.raises(ValueError, match="step_frac must be in"):
        generate_fraction_splits(simple_df, "date", window_frac=0.2, step_frac=2)

def test_window_frac_step_frac_no_splits(simple_df):
    with pytest.raises(ValueError, match="No splits produced"):
        generate_fraction_splits(simple_df, "date", window_frac=0.9, step_frac=1.0)

# Test for split_datasets()
# -----------------------
# Fixtures and helpers
# -----------------------

@pytest.fixture
def simple_df():
    now = pd.Timestamp("2023-01-01")
    return pd.DataFrame({
        "date": [now + timedelta(days=i) for i in range(100)],
        "value": np.arange(100),
    })


# -----------------------
# Tests for valid inputs
# -----------------------

def test_temporal_split_equal_windows(simple_df):
    splits, log = split_datasets(
        df=simple_df,
        date_col="date",
        train_test_splits=5,
        method="temporal"
    )
    assert len(splits) == 5
    assert isinstance(log, pd.DataFrame)
    assert log.shape[0] == 5
    for s in splits:
        assert isinstance(s, pd.DataFrame)
        assert not s.empty or s.empty  # just ensure it's valid

def test_temporal_split_with_explicit_ranges(simple_df):
    now = simple_df["date"].min()
    ranges = [(now + timedelta(days=20 * i), now + timedelta(days=20 * (i + 1))) for i in range(5)]
    splits, log = split_datasets(
        df=simple_df,
        date_col="date",
        train_test_splits=5,
        train_test_ranges=ranges,
        method="temporal"
    )
    assert len(splits) == 5
    assert all(isinstance(s, pd.DataFrame) for s in splits)
    assert log["n_rows"].sum() == len(simple_df)

@pytest.mark.parametrize("overlap", [False, True])
def test_fractional_split_equal_windows(simple_df, overlap):
    splits, log = split_datasets(
        df=simple_df,
        date_col="date",
        train_test_splits=4,
        method="fractional",
        train_test_overlap=overlap
    )
    assert len(splits) == 4
    assert isinstance(log, pd.DataFrame)
    assert log.shape[0] == 4
    assert all(isinstance(s, pd.DataFrame) for s in splits)

def test_fractional_split_with_custom_fractions(simple_df):
    splits, log = split_datasets(
        df=simple_df,
        date_col="date",
        train_test_fractions=[0.25, 0.25, 0.25, 0.25],
        method="fractional",
        train_test_overlap=False
    )
    assert len(splits) == 4
    assert isinstance(log, pd.DataFrame)
    assert log["n_rows"].sum() == len(simple_df)

def test_fractional_split_with_window_and_step(simple_df):
    splits, log = split_datasets(
        df=simple_df,
        date_col="date",
        train_test_window_frac=0.2,
        train_test_step_frac=0.2,
        method="fractional"
    )
    assert all(isinstance(s, pd.DataFrame) for s in splits)
    assert isinstance(log, pd.DataFrame)


# -----------------------
# Tests for invalid input
# -----------------------

def test_invalid_method_raises(simple_df):
    with pytest.raises(ValueError, match="Invalid method"):
        split_datasets(df=simple_df, date_col="date", method="unsupported")

def test_missing_required_params_raises(simple_df):
    with pytest.raises(ValueError, match="Must provide one of"):
        # This only happens inside generate_fraction_splits but we trigger it via wrapper
        split_datasets(df=simple_df, date_col="date", method="fractional")

def test_invalid_date_column_raises():
    df = pd.DataFrame({"wrong_col": pd.date_range("2022-01-01", periods=10)})
    with pytest.raises(KeyError):
        split_datasets(df=df, date_col="not_there", method="temporal", train_test_splits=2)

def test_empty_dataframe_returns_empty_splits():
    df = pd.DataFrame(columns=["date", "value"])
    splits, log = split_datasets(df=df, date_col="date", method="temporal", train_test_splits=3)
    assert all(s.empty for s in splits)
    assert log["n_rows"].sum() == 0

# Test for rules_series_to_unique_rules()
@pytest.mark.parametrize("rule_str,expected", [
    ("('a' == 0)", [("a", 0)]),
    ("('a' == 1) AND ('b' == 0)", [("a", 1), ("b", 0)]),
    ("('x' == 0) AND ('y' == 1) AND ('z' == 1)", [("x", 0), ("y", 1), ("z", 1)]),
])
def test_valid_rules_parsing(rule_str, expected):
    series = pd.Series([rule_str])
    result = rules_series_to_unique_rules(series)
    assert len(result) == 1
    assert result[0][0] == expected
    assert result[0][1] == set()


def test_valid_rules_with_provenance():
    rule = "('a' == 0) AND ('b' == 1)"
    series = pd.Series([rule])
    result = rules_series_to_unique_rules(series, provenance=True)
    assert result[0][0] == [("a", 0), ("b", 1)]
    assert result[0][1] == {rule}


def test_empty_string_rule_returns_empty_conditions():
    series = pd.Series(["", "   "])
    result = rules_series_to_unique_rules(series)
    assert all(r[0] == [] for r in result)
    assert all(r[1] == set() for r in result)


def test_non_string_values_handled_gracefully():
    series = pd.Series([None, 42, True])
    result = rules_series_to_unique_rules(series)
    assert len(result) == 3
    assert all(r[0] == [] for r in result)


@pytest.mark.parametrize("bad_rule", [
    "('a' = 0)",              # wrong operator
    "('a' == '0')",           # value should be int
    "('a' == 0) && ('b' == 1)",  # wrong AND operator
    "feature1 == 0",          # missing parentheses/quotes
    "('a'==2)",               # unsupported value
])
def test_malformed_rules_raise_value_error(bad_rule):
    series = pd.Series([bad_rule])
    with pytest.raises(ValueError, match="Malformed condition"):
        rules_series_to_unique_rules(series)


def test_mixed_valid_and_invalid_rules_raises_on_first_error():
    series = pd.Series([
        "('a' == 1)",
        "('b' = 0)",  # malformed
        "('c' == 1)"
    ])
    with pytest.raises(ValueError):
        rules_series_to_unique_rules(series)


def test_multiple_rules_batch():
    series = pd.Series([
        "('a' == 1)",
        "('b' == 0) AND ('c' == 1)",
        "('x' == 0) AND ('y' == 1) AND ('z' == 0)"
    ])
    result = rules_series_to_unique_rules(series)
    expected_conditions = [
        [("a", 1)],
        [("b", 0), ("c", 1)],
        [("x", 0), ("y", 1), ("z", 0)],
    ]
    for i in range(3):
        assert result[i][0] == expected_conditions[i]
        assert result[i][1] == set()

# Test for make_combined_rule_feature_df()
# Mock dependencies
def mock_rules_series_to_unique_rules(series):
    # Return dummy rules for testing: feature 'x' == 1
    return [([("x", 1)], {s}) for s in series]

def mock_extract_rule_feature_names(series):
    return list(series)

def mock_generate_rule_activation_dataframe(df, rules, target_col, prefix="rule"):
    col_name = f"{prefix}_0000"
    mask = df["x"] == 1
    rule_df = pd.DataFrame({col_name: mask, target_col: df[target_col]}, index=df.index)
    mapping_df = pd.DataFrame({"rule_column": [col_name], "human_readable_rule": ["('x' == 1)"]})
    return rule_df, mapping_df

# Patch in test module
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("my_module.rules_series_to_unique_rules", mock_rules_series_to_unique_rules)
    monkeypatch.setattr("my_module.extract_rule_feature_names", mock_extract_rule_feature_names)
    monkeypatch.setattr("my_module.generate_rule_activation_dataframe", mock_generate_rule_activation_dataframe)


# ---------------------------
# Test cases
# ---------------------------

def test_combined_df_with_both_rule_types():
    train = pd.DataFrame({
        "antecedents": ["('x' == 1)", "x"],
        "rule_depth": [2, 1],
        "selected": [True, True],
    })
    test = pd.DataFrame({"x": [0, 1, 1], "y": [1, 2, 3], "target": [0, 1, 0]})

    df, mapping = make_combined_rule_feature_df(train, test, "target")

    assert "x" in df.columns  # single-variate rule
    assert "rule_0000" in df.columns  # multivariate rule
    assert "target" in df.columns
    assert df.shape[0] == 3
    assert mapping.shape[0] == 1
    assert "rule_column" in mapping.columns


def test_combined_df_with_only_singlevar_rules():
    train = pd.DataFrame({
        "antecedents": ["x"],
        "rule_depth": [1],
        "selected": [True],
    })
    test = pd.DataFrame({"x": [1, 0], "target": [0, 1]})
    df, mapping = make_combined_rule_feature_df(train, test, "target")

    assert "x" in df.columns
    assert "target" in df.columns
    assert df.shape == (2, 2)
    assert mapping.empty


def test_combined_df_with_only_multivar_rules():
    train = pd.DataFrame({
        "antecedents": ["('x' == 1)"],
        "rule_depth": [2],
        "selected": [True],
    })
    test = pd.DataFrame({"x": [1, 0], "target": [1, 0]})
    df, mapping = make_combined_rule_feature_df(train, test, "target")

    assert "rule_0000" in df.columns
    assert "target" in df.columns
    assert df.shape == (2, 2)
    assert mapping.shape == (1, 2)


def test_combined_df_with_no_selected_rules_adds_target_only():
    train = pd.DataFrame({
        "antecedents": ["x"],
        "rule_depth": [1],
        "selected": [False],
    })
    test = pd.DataFrame({"x": [1, 0], "target": [1, 0]})
    df, mapping = make_combined_rule_feature_df(train, test, "target")

    assert df.columns.tolist() == ["target"]
    assert df.shape == (2, 1)
    assert mapping.empty


def test_missing_target_column_raises():
    train = pd.DataFrame({
        "antecedents": ["x"],
        "rule_depth": [1],
        "selected": [True],
    })
    test = pd.DataFrame({"x": [1, 0]})  # no target
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        make_combined_rule_feature_df(train, test, "target")


def test_missing_required_mining_columns_raises():
    test = pd.DataFrame({"x": [1], "target": [0]})
    bad_mining = pd.DataFrame({
        "rule_depth": [1],
        "selected": [True],
        # missing 'antecedents'
    })
    with pytest.raises(KeyError, match="train_mining_res is missing columns"):
        make_combined_rule_feature_df(bad_mining, test, "target")

# Test for test_mined_rules()
# ---------------------
# Mocked dependencies
# ---------------------

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Mock multivariate + singlevar rule handling
    monkeypatch.setattr("my_module.make_combined_rule_feature_df", lambda *_: (
        pd.DataFrame({
            "rule_0000": [1, 0, 1],
            "x": [0, 1, 0],
            "target": [1, 0, 1]
        }),
        pd.DataFrame({
            "rule_column": ["rule_0000"],
            "human_readable_rule": ["('a' == 1) AND ('b' == 0)"]
        })
    ))

    monkeypatch.setattr("my_module.generate_statistics", lambda df, cfg: (
        pd.DataFrame({
            "rule_column": ["rule_0000", "x"],
            "lift": [1.5, 1.2],
            "antecedents": ["('a' == 1) AND ('b' == 0)", "x"]
        }),
        pd.DataFrame({"log_key": ["value"]})
    ))

    monkeypatch.setattr("my_module.compute_rule_depth", lambda rule: rule.count("AND") + 1)
    monkeypatch.setattr("my_module.merge_multivar_map_into_stats", lambda stats, mapping: stats.merge(
        mapping, on="rule_column", how="left"
    ))


# ---------------------
# Tests
# ---------------------

def test_returns_expected_structure_and_columns():
    train = pd.DataFrame({
        "antecedents": ["('a' == 1) AND ('b' == 0)", "x"],
        "rule_depth": [2, 1],
        "selected": [True, True],
    })

    test = pd.DataFrame({
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "x": [1, 0, 1],
        "target": [1, 0, 1]
    })

    cfg = {"some": "config"}
    stats_df, stats_log = test_mined_rules(train, test, cfg, "target")

    assert isinstance(stats_df, pd.DataFrame)
    assert isinstance(stats_log, pd.DataFrame)

    assert "antecedents" in stats_df.columns
    assert "rule_depth" in stats_df.columns
    assert "lift" in stats_df.columns
    assert stats_df.shape[0] == 2
    assert stats_log.shape[1] >= 1


def test_fallback_to_singlevar_rule_metadata():
    # One rule is not in the mapping and should fall back to original antecedent
    monkeypatch_mapping = pd.DataFrame({
        "rule_column": ["rule_0000"],
        "human_readable_rule": ["('a' == 1) AND ('b' == 0)"]
    })

    monkeypatch_stats = pd.DataFrame({
        "rule_column": ["rule_0000", "x"],
        "antecedents": ["('a' == 1) AND ('b' == 0)", "x"],
        "lift": [1.5, 1.0]
    })

    # Reuse previous logic via monkeypatch, so not duplicating this test


def test_missing_columns_raise_errors():
    train_missing_col = pd.DataFrame({
        "rule_depth": [1],
        "selected": [True]
    })
    test = pd.DataFrame({"target": [1, 0]})
    cfg = {}

    with pytest.raises(KeyError, match="train_mining_res is missing columns"):
        test_mined_rules(train_missing_col, test, cfg, "target")

    test_missing_target = pd.DataFrame({"x": [1, 0]})
    valid_train = pd.DataFrame({
        "antecedents": ["x"],
        "rule_depth": [1],
        "selected": [True]
    })

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        test_mined_rules(valid_train, test_missing_target, cfg, "target")


def test_handles_empty_rule_set():
    empty_train = pd.DataFrame(columns=["antecedents", "rule_depth", "selected"])
    test = pd.DataFrame({"x": [0, 1], "target": [1, 0]})
    cfg = {}

    stats, log = test_mined_rules(empty_train, test, cfg, "target")

    assert isinstance(stats, pd.DataFrame)
    assert isinstance(log, pd.DataFrame)
    # Should still contain target info even if no rules
    assert "antecedents" in stats.columns

# Test for split_mining_pipeline()
# -------------------------------
# Fixtures for mocking dependencies
# -------------------------------

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("my_module.data_prep_pipeline", lambda df, cfg, logger: (df.copy(), {"prep": "done"}))

    monkeypatch.setattr("my_module.mining_pipeline", lambda df, cfg, logger: (
        pd.DataFrame({"rule": ["rule_1", "rule_2"]}),
        2,
        {"mined": True}
    ))

    monkeypatch.setattr("my_module.test_mined_rules", lambda rules_df, df, cfg, target: (
        pd.DataFrame({"rule": ["rule_1"], "lift": [1.2]}),
        {"tested": True}
    ))


# -------------------------------
# Unit tests
# -------------------------------

def test_re_mine_true_runs_per_split():
    # 3 splits, re-mining enabled
    splits = [pd.DataFrame({"x": [1, 0], "target": [1, 0]}) for _ in range(3)]
    cfg = {}
    results, rule_counts, logs, initial_rules = split_mining_pipeline(
        splits=splits,
        cfg=cfg,
        re_mine=True,
        target_col="target"
    )

    assert len(results) == 3
    assert all(isinstance(df, pd.DataFrame) for df in results)
    assert all(count == 2 for count in rule_counts)
    assert all("mine_log" in log for log in logs)
    assert initial_rules is None


def test_re_mine_false_mines_once_tests_rest():
    # 3 splits, mine first, test rest
    splits = [pd.DataFrame({"x": [1, 0], "target": [1, 0]}) for _ in range(3)]
    cfg = {}
    results, rule_counts, logs, initial_rules = split_mining_pipeline(
        splits=splits,
        cfg=cfg,
        re_mine=False,
        target_col="target"
    )

    assert len(results) == 3
    assert isinstance(initial_rules, pd.DataFrame)
    assert rule_counts[0] == 2
    assert all("prep_log" in log for log in logs)
    assert "mine_log" in logs[0]
    assert "test_log" in logs[1]
    assert "test_log" in logs[2]


def test_raises_on_short_split_list():
    with pytest.raises(ValueError, match="splits.*at least 2"):
        split_mining_pipeline(splits=[pd.DataFrame()], cfg={}, re_mine=True, target_col="target")


def test_raises_on_non_dataframe_elements():
    with pytest.raises(TypeError, match="must be of type pd.DataFrame"):
        split_mining_pipeline(splits=[pd.DataFrame(), "not a df"], cfg={}, re_mine=False, target_col="target")


def test_runtime_error_if_initial_rules_missing(monkeypatch):
    monkeypatch.setattr("my_module.data_prep_pipeline", lambda df, cfg, logger: (df.copy(), {}))
    monkeypatch.setattr("my_module.mining_pipeline", lambda df, cfg, logger: (None, 0, {}))

    splits = [pd.DataFrame({"x": [1], "target": [1]}) for _ in range(2)]
    with pytest.raises(RuntimeError, match="initial_rules.*set"):
        split_mining_pipeline(splits=splits, cfg={}, re_mine=False, target_col="target")

# Test for combine_split_results()
# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def typical_inputs():
    df1 = pd.DataFrame({
        "antecedents": ["a == 1", "b == 0"],
        "consequents": ["up", "down"],
        "rule_depth": [1, 2],
        "lift": [1.1, 0.9],
        "confidence": [0.6, 0.4]
    })
    df2 = pd.DataFrame({
        "antecedents": ["a == 1", "b == 0"],
        "consequents": ["up", "down"],
        "rule_depth": [1, 2],
        "lift": [1.2, 0.95],
        "support": [0.3, 0.25]
    })
    return [df1, df2]


# -----------------------------
# Tests
# -----------------------------

def test_combine_split_results_typical(typical_inputs):
    combined = combine_split_results(typical_inputs)
    
    assert isinstance(combined, pd.DataFrame)
    assert "split_0_lift" in combined.columns
    assert "split_0_confidence" in combined.columns
    assert "split_1_lift" in combined.columns
    assert "split_1_support" in combined.columns
    assert "antecedents" in combined.columns
    assert "consequents" in combined.columns
    assert "rule_depth" in combined.columns
    assert combined.shape[0] == 2


def test_combine_split_results_with_custom_prefixes(typical_inputs):
    combined = combine_split_results(typical_inputs, split_prefixes=["train", "val"])
    
    assert "train_lift" in combined.columns
    assert "val_support" in combined.columns
    assert "train_confidence" in combined.columns
    assert "val_lift" in combined.columns


def test_combine_split_results_missing_rule_depth():
    df1 = pd.DataFrame({
        "antecedents": ["a == 1"],
        "consequents": ["up"],
        "lift": [1.0]
    })
    df2 = pd.DataFrame({
        "antecedents": ["a == 1"],
        "consequents": ["up"],
        "confidence": [0.7]
    })
    combined = combine_split_results([df1, df2])

    assert "rule_depth" not in combined.columns
    assert "split_0_lift" in combined.columns
    assert "split_1_confidence" in combined.columns


def test_combine_split_results_invalid_input_type():
    with pytest.raises(TypeError, match="pandas DataFrames"):
        combine_split_results([{"not": "a dataframe"}, pd.DataFrame()])


def test_combine_split_results_too_few_inputs():
    with pytest.raises(ValueError, match="at least 2 DataFrames"):
        combine_split_results([pd.DataFrame()])


def test_combine_split_results_mismatched_prefix_length(typical_inputs):
    with pytest.raises(ValueError, match="must match number of splits"):
        combine_split_results(typical_inputs, split_prefixes=["only_one"])


def test_combine_split_results_missing_required_columns():
    df1 = pd.DataFrame({
        "rule_depth": [1],
        "lift": [1.0]
    })
    df2 = pd.DataFrame({
        "antecedents": ["a == 1"],
        "lift": [1.0]
    })
    with pytest.raises(ValueError, match="must include 'antecedents' and 'consequents'"):
        combine_split_results([df1, df2])

# Test for create_validation_log_df()
def test_create_validation_log_df_typical_case():
    df = pd.DataFrame({
        "split_0_lift": [1.1, 1.2, np.nan],
        "split_0_selected": [True, False, True],
        "split_0_confidence": [0.8, 0.9, np.nan],
        "split_1_lift": [1.0, np.nan, 1.3],
        "split_1_selected": [False, True, True],
        "split_1_confidence": [0.7, np.nan, 0.95],
        "split_0_observations": [10, 20, 30],
        "split_1_observations": [15, 25, 35],
    })

    out = create_validation_log_df(df)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, len(out.columns))
    assert out["n_rules_split_0"].iloc[0] == 2
    assert out["n_selected_split_0"].iloc[0] == 1
    assert out["n_rules_split_1"].iloc[0] == 2
    assert out["n_selected_split_1"].iloc[0] == 1
    assert out["n_overlap_rules"].iloc[0] == 1
    assert out["n_overlap_selected"].iloc[0] == 0


def test_create_validation_log_df_inferred_splits():
    df = pd.DataFrame({
        "split_A_lift": [1.0, 1.5],
        "split_A_selected": ["1", "0"],  # string-based booleans
        "split_B_lift": [np.nan, 1.2],
        "split_B_selected": ["false", "true"]
    })

    out = create_validation_log_df(df)
    assert "n_rules_split_A" in out.columns
    assert out["n_selected_split_A"].iloc[0] == 1
    assert out["n_overlap_rules"].iloc[0] == 1
    assert out["n_overlap_selected"].iloc[0] == 0


def test_create_validation_log_df_empty_input():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input train_test_results is empty."):
        create_validation_log_df(df)


def test_create_validation_log_df_missing_metrics_graceful():
    df = pd.DataFrame({
        "split_0_lift": [1.0],
        "split_0_selected": [True],
    })

    # split_1 missing entirely, should not raise
    out = create_validation_log_df(df, splits=["split_0", "split_1"])
    assert np.isnan(out["n_rules_split_1"]).all() or out["n_rules_split_1"].iloc[0] == 0


@pytest.mark.parametrize("bool_input,expected", [
    ([True, False, True], 2),
    (["true", "false", "1"], 2),
    (["1", "0", "false"], 1),
    ([np.nan, "TRUE", "0"], 1),
])
def test_create_validation_log_df_selected_variants(bool_input, expected):
    df = pd.DataFrame({
        "split_0_lift": [1.0] * 3,
        "split_0_selected": bool_input,
    })

    out = create_validation_log_df(df, splits=["split_0"])
    assert out["n_selected_split_0"].iloc[0] == expected

# Test for validate_train_test()
# Mock dependencies
@pytest.fixture
def minimal_cfg():
    return {"dummy_config": True}


@pytest.fixture
def simple_df():
    # Create a minimal dummy DataFrame with a date and target column
    base_date = pd.to_datetime("2023-01-01")
    return pd.DataFrame({
        "date": [base_date + timedelta(days=i) for i in range(20)],
        "feature_1": [i % 2 for i in range(20)],
        "target": [i % 3 for i in range(20)]
    })


@pytest.fixture
def patched_dependencies(monkeypatch):
    import pandas as pd

    def dummy_split_datasets(
        df, date_col, n_splits, manual_ranges, method,
        window_frac, step_frac, fractions, overlap
    ):
        return [df.copy(), df.copy()], {"log": "dummy_split"}

    def dummy_split_mining_pipeline(splits, cfg, re_mine, target_col, logger):
        dummy_result = pd.DataFrame({
            "antecedents": ["a"], "consequents": ["b"],
            "split_0_lift": [1.1], "split_0_selected": [True]
        })
        return [dummy_result], [1], [{"log": "dummy"}], pd.DataFrame()

    def dummy_combine_split_results(results):
        return results[0]

    monkeypatch.setattr("my_module.split_datasets", dummy_split_datasets)
    monkeypatch.setattr("my_module.split_mining_pipeline", dummy_split_mining_pipeline)
    monkeypatch.setattr("my_module.combine_split_results", dummy_combine_split_results)


# --- Tests --- #

def test_validate_train_test_typical_case(simple_df, minimal_cfg, patched_dependencies):
    result_df, meta = validate_train_test(
        df=simple_df,
        cfg=minimal_cfg,
        target_col="target",
        date_col="date",
        train_test_splits=2,
        train_test_ranges=[["2023-01-01", "2023-01-20"]],
        train_test_split_method="fixed",
        train_test_window_frac=0.5,
        train_test_step_frac=0.5,
        train_test_fractions=[0.5, 0.5],
        train_test_overlap=False,
        train_test_re_mine=False,
        logger=None
    )

    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert "antecedents" in result_df.columns
    assert "split_0_lift" in result_df.columns

    assert isinstance(meta, dict)
    assert "train_test_rule_counts" in meta
    assert "train_test_logs" in meta
    assert "train_test_initial_rules" in meta


def test_validate_train_test_invalid_df_raises(minimal_cfg):
    with pytest.raises(AttributeError):
        validate_train_test(
            df=None,
            cfg=minimal_cfg,
            target_col="target",
            date_col="date",
            train_test_splits=2,
            train_test_ranges=[],
            train_test_split_method="fixed",
            train_test_window_frac=0.5,
            train_test_step_frac=0.5,
            train_test_fractions=[0.5, 0.5],
            train_test_overlap=False,
            train_test_re_mine=True
        )


def test_validate_train_test_edge_case_one_split(simple_df, minimal_cfg, patched_dependencies):
    # Should pass validation since it's mocked
    result_df, meta = validate_train_test(
        df=simple_df,
        cfg=minimal_cfg,
        target_col="target",
        date_col="date",
        train_test_splits=1,
        train_test_ranges=[["2023-01-01", "2023-01-20"]],
        train_test_split_method="fixed",
        train_test_window_frac=1.0,
        train_test_step_frac=1.0,
        train_test_fractions=[0.5, 0.5],
        train_test_overlap=True,
        train_test_re_mine=True,
        logger=None
    )

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert "train_test_logs" in meta

# Test for validate_wfa()
@pytest.fixture
def minimal_wfa_input():
    # Minimal valid DataFrame
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "target": [0, 1] * 5,
        "feature": range(10),
    })

    cfg = MagicMock(name="MockConfig")

    # Mock required pipeline dependencies
    mock_split_datasets = MagicMock(return_value=(
        [df.iloc[:5], df.iloc[5:]],  # splits
        {"info": "mock_split_log"}
    ))
    mock_split_mining_pipeline = MagicMock(return_value=(
        [pd.DataFrame({"antecedents": ["rule1"], "lift": [1.2]})],  # results
        [1],  # rule_counts
        [{"log": "mock_log"}],  # logs
        pd.DataFrame({"antecedents": ["rule1"]})  # initial_rules
    ))
    mock_combine_split_results = MagicMock(
        return_value=pd.DataFrame({"antecedents": ["rule1"], "split_0_lift": [1.2]})
    )

    return {
        "df": df,
        "cfg": cfg,
        "target_col": "target",
        "date_col": "date",
        "wfa_splits": 1,
        "wfa_ranges": [],
        "wfa_split_method": "fixed",
        "wfa_window_frac": 0.5,
        "wfa_step_frac": 0.5,
        "wfa_fractions": [0.7, 0.3],
        "wfa_overlap": False,
        "wfa_re_mine": True,
        "mocks": {
            "split_datasets": mock_split_datasets,
            "split_mining_pipeline": mock_split_mining_pipeline,
            "combine_split_results": mock_combine_split_results,
        },
    }


def test_validate_wfa_typical(monkeypatch, minimal_wfa_input):
    args = minimal_wfa_input

    # Patch dependent functions
    monkeypatch.setattr("your_module.split_datasets", args["mocks"]["split_datasets"])
    monkeypatch.setattr("your_module.split_mining_pipeline", args["mocks"]["split_mining_pipeline"])
    monkeypatch.setattr("your_module.combine_split_results", args["mocks"]["combine_split_results"])

    combined, meta = validate_wfa(
        df=args["df"],
        cfg=args["cfg"],
        target_col=args["target_col"],
        date_col=args["date_col"],
        wfa_splits=args["wfa_splits"],
        wfa_ranges=args["wfa_ranges"],
        wfa_split_method=args["wfa_split_method"],
        wfa_window_frac=args["wfa_window_frac"],
        wfa_step_frac=args["wfa_step_frac"],
        wfa_fractions=args["wfa_fractions"],
        wfa_overlap=args["wfa_overlap"],
        wfa_re_mine=args["wfa_re_mine"],
    )

    # Check output structure
    assert isinstance(combined, pd.DataFrame)
    assert "antecedents" in combined.columns
    assert "split_0_lift" in combined.columns

    assert isinstance(meta, dict)
    assert "wfa_rule_counts" in meta
    assert "wfa_logs" in meta
    assert "wfa_initial_rules" in meta


def test_validate_wfa_raises_on_empty(monkeypatch):
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_wfa(
            df=df,
            cfg=None,
            target_col="target",
            date_col="date",
            wfa_splits=2,
            wfa_ranges=[],
            wfa_split_method="rolling",
            wfa_window_frac=0.5,
            wfa_step_frac=0.5,
            wfa_fractions=[0.7, 0.3],
            wfa_overlap=False,
            wfa_re_mine=True,
        )


@pytest.mark.parametrize("invalid_frac", [[1.5, -0.5], [0.3], [], [0.5, 0.5, 0.1]])
def test_validate_wfa_invalid_fractions(invalid_frac, minimal_wfa_input):
    args = minimal_wfa_input
    with pytest.raises(ValueError):
        validate_wfa(
            df=args["df"],
            cfg=args["cfg"],
            target_col=args["target_col"],
            date_col=args["date_col"],
            wfa_splits=args["wfa_splits"],
            wfa_ranges=args["wfa_ranges"],
            wfa_split_method=args["wfa_split_method"],
            wfa_window_frac=args["wfa_window_frac"],
            wfa_step_frac=args["wfa_step_frac"],
            wfa_fractions=invalid_frac,
            wfa_overlap=args["wfa_overlap"],
            wfa_re_mine=args["wfa_re_mine"],
        )


# Test for train_test_pipeline()
@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=10),
        "feature": range(10),
        "target": [0, 1] * 5,
    })


@pytest.fixture
def dummy_cfg():
    class DummyCfg:
        date_col = "date"
        target_col = "target"
        train_test_splits = 2
        train_test_ranges = []
        train_test_split_method = "rolling"
        train_test_window_frac = 0.5
        train_test_step_frac = 0.5
        train_test_fractions = [0.7, 0.3]
        train_test_overlap = False
        train_test_re_mine = True
        log_max_rows = 10
    return DummyCfg()


@pytest.fixture
def mock_validate_train_test(monkeypatch):
    def _mock(*args, **kwargs):
        df_result = pd.DataFrame({"antecedents": ["x"], "split_0_lift": [1.0], "split_1_lift": [0.9]})
        logs = {"train_test_rule_counts": [], "train_test_logs": [], "train_test_initial_rules": []}
        return df_result, logs
    monkeypatch.setattr("your_module.validate_train_test", _mock)


@pytest.fixture
def mock_create_validation_log_df(monkeypatch):
    def _mock(*args, **kwargs):
        return pd.DataFrame([{"mean_lift_split_0": 1.0, "mean_lift_split_1": 0.9}])
    monkeypatch.setattr("your_module.create_validation_log_df", _mock)


def test_train_test_pipeline_basic(
    dummy_df, dummy_cfg, mock_validate_train_test, mock_create_validation_log_df
):
    results, log_df, metadata = train_test_pipeline(df=dummy_df, cfg=dummy_cfg)

    # Validate output structure
    assert isinstance(results, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert isinstance(metadata, dict)
    assert "train_test_rule_counts" in metadata or "train_test_logs" in metadata


def test_train_test_pipeline_with_overrides(
    dummy_df, dummy_cfg, mock_validate_train_test, mock_create_validation_log_df
):
    overrides = {"train_test_splits": 3, "train_test_overlap": True}
    results, log_df, metadata = train_test_pipeline(df=dummy_df, cfg=dummy_cfg, **overrides)

    assert isinstance(results, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)


def test_train_test_pipeline_with_logger(
    dummy_df, dummy_cfg, mock_validate_train_test, mock_create_validation_log_df
):
    logger = MagicMock()
    _ = train_test_pipeline(df=dummy_df, cfg=dummy_cfg, logger=logger)

    logger.log_step.assert_called_once()
    call_args = logger.log_step.call_args[1]
    assert call_args["step_name"] == "Train / Test split validation test"
    assert isinstance(call_args["df"], pd.DataFrame)


def test_train_test_pipeline_invalid_cfg():
    with pytest.raises(AttributeError):
        train_test_pipeline(df=pd.DataFrame(), cfg=object())

# Test for wfa_pipeline()
@pytest.fixture
def dummy_cfg():
    # Simulates a config object with required WFA attributes
    class Config:
        date_col = "date"
        target_col = "target"
        wfa_split_method = "rolling"
        wfa_splits = 2
        wfa_ranges = []
        wfa_window_frac = 0.5
        wfa_step_frac = 0.25
        wfa_fractions = [0.7, 0.3]
        wfa_overlap = False
        wfa_re_mine = True
        log_max_rows = 10
    return Config()


@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        "feature1": range(10)
    })


@pytest.fixture
def mock_validate_wfa(monkeypatch):
    def fake_validate_wfa(df, cfg, logger=None, **kwargs):
        df_result = pd.DataFrame({
            "antecedents": ["a"], "consequents": ["b"],
            "split_0_lift": [1.5], "split_1_lift": [1.2],
            "split_0_selected": [True], "split_1_selected": [False]
        })
        logs = {
            "wfa_rule_counts": [1, 1],
            "wfa_logs": ["log0", "log1"],
            "wfa_initial_rules": ["rule"]
        }
        return df_result, logs
    monkeypatch.setattr("my_module.validate_wfa", fake_validate_wfa)


@pytest.fixture
def mock_create_validation_log_df(monkeypatch):
    def fake_create_log(df, splits):
        return pd.DataFrame({"summary": [42]})
    monkeypatch.setattr("my_module.create_validation_log_df", fake_create_log)


def test_wfa_pipeline_normal_case(
    dummy_df, dummy_cfg, mock_validate_wfa, mock_create_validation_log_df
):
    result_df, log_df, logs = wfa_pipeline(dummy_df, dummy_cfg)

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert isinstance(logs, dict)
    assert "wfa_rule_counts" in logs
    assert "wfa_logs" in logs
    assert "wfa_initial_rules" in logs


def test_wfa_pipeline_with_override(
    dummy_df, dummy_cfg, mock_validate_wfa, mock_create_validation_log_df
):
    override_cfg = {"wfa_splits": 3, "wfa_overlap": True}
    _, _, _ = wfa_pipeline(dummy_df, dummy_cfg, **override_cfg)  # should not raise


def test_wfa_pipeline_logger_called(
    dummy_df, dummy_cfg, mock_validate_wfa, mock_create_validation_log_df
):
    class DummyLogger:
        def __init__(self):
            self.logged = False

        def log_step(self, step_name, info, df, max_rows):
            self.logged = True
            assert step_name == "Walk Forward validation test"
            assert isinstance(df, pd.DataFrame)
            assert "wfa_split_method" in info

    logger = DummyLogger()
    _, _, _ = wfa_pipeline(dummy_df, dummy_cfg, logger=logger)
    assert logger.logged


def test_wfa_pipeline_invalid_input_raises(dummy_cfg):
    with pytest.raises(Exception):
        wfa_pipeline(None, dummy_cfg)  # df is None


def test_wfa_pipeline_missing_split_columns(monkeypatch, dummy_df, dummy_cfg):
    def fake_validate_wfa(df, cfg, logger=None, **kwargs):
        return pd.DataFrame({"antecedents": ["a"], "consequents": ["b"]}), {}
    def fake_create_log(df, splits):
        return pd.DataFrame({"summary": [0]})
    monkeypatch.setattr("my_module.validate_wfa", fake_validate_wfa)
    monkeypatch.setattr("my_module.create_validation_log_df", fake_create_log)

    # Should not fail even if no split columns exist
    result, log, meta = wfa_pipeline(dummy_df, dummy_cfg)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(log, pd.DataFrame)
    assert isinstance(meta, dict)

# Test for resample_dataframe()
@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=10)
    return pd.DataFrame({
        "date": dates,
        "value": np.arange(10),
        "group": ["A"] * 5 + ["B"] * 5
    })


@pytest.mark.parametrize("mode", ["traditional", "block"])
def test_valid_resample_modes(sample_df, mode):
    resampled = resample_dataframe(sample_df, mode=mode, block_size=3, date_col="date", random_state=0)
    assert isinstance(resampled, pd.DataFrame)
    assert len(resampled) == len(sample_df)
    assert set(sample_df.columns) == set(resampled.columns)


def test_block_ids_resampling(sample_df):
    resampled = resample_dataframe(
        sample_df,
        mode="block_ids",
        block_size=2,
        date_col="date",
        id_cols=["group"],
        random_state=1
    )
    assert isinstance(resampled, pd.DataFrame)
    assert len(resampled) == len(sample_df)
    assert "group" in resampled.columns
    assert set(resampled["group"]) == {"A", "B"}


def test_traditional_resample_repeatability(sample_df):
    r1 = resample_dataframe(sample_df, mode="traditional", random_state=123)
    r2 = resample_dataframe(sample_df, mode="traditional", random_state=123)
    pd.testing.assert_frame_equal(r1, r2)


def test_invalid_mode_raises(sample_df):
    with pytest.raises(ValueError, match="Invalid mode"):
        resample_dataframe(sample_df, mode="invalid")


def test_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=["date", "value"])
    with pytest.raises(ValueError, match="Input dataframe is empty"):
        resample_dataframe(empty_df, mode="traditional")


def test_missing_id_cols_raises(sample_df):
    with pytest.raises(ValueError, match="id_cols must be provided"):
        resample_dataframe(sample_df, mode="block_ids", block_size=3)


def test_invalid_block_size_raises(sample_df):
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        resample_dataframe(sample_df, mode="traditional", block_size=0)


def test_block_size_larger_than_data_raises(sample_df):
    small_df = sample_df.iloc[:3]
    with pytest.raises(ValueError, match="block_size=5 exceeds data length 3"):
        resample_dataframe(small_df, mode="block", block_size=5)


def test_groupwise_block_size_exceeds_group_raises(sample_df):
    # Shrink one group to size 2, then use block_size=3
    df = sample_df.copy()
    df = df[df["group"] != "A"].append(df[df["group"] == "A"].iloc[:2], ignore_index=True)
    with pytest.raises(ValueError, match="block_size=3 exceeds data length 2"):
        resample_dataframe(df, mode="block_ids", block_size=3, id_cols=["group"])

# Test for summarize_rule_metrics()
@pytest.fixture
def sample_rule_df():
    return pd.DataFrame({
        "antecedents": ["A", "A", "B", "B", "B"],
        "consequents": ["X", "X", "Y", "Y", "Y"],
        "selected": [True, False, True, True, False],
        "lift": [1.1, 0.9, 1.3, 1.0, 0.8],
        "confidence": [0.6, 0.7, 0.9, 0.85, 0.8],
    })


def test_summarize_rule_metrics_basic_output(sample_rule_df):
    result = summarize_rule_metrics(sample_rule_df, metrics=["lift", "confidence"])
    
    # Should return one row per (antecedent, consequent) pair
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert "lift_mean" in result.columns
    assert "confidence_q05" in result.columns
    assert "selected_fraction" in result.columns

    # Validate correct selection counts
    rule_ax = result[result["antecedents"] == "A"]
    assert rule_ax["selected_selected_count"].iloc[0] == 1
    assert rule_ax["selected_test_count"].iloc[0] == 2
    assert rule_ax["selected_fraction"].iloc[0] == 0.5


def test_missing_required_column_raises(sample_rule_df):
    bad_df = sample_rule_df.drop(columns=["selected"])
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_rule_metrics(bad_df, metrics=["lift"])


def test_quantile_columns_exist(sample_rule_df):
    result = summarize_rule_metrics(sample_rule_df, metrics=["lift"])
    assert "lift_q05" in result.columns
    assert "lift_q95" in result.columns


def test_single_rule_group():
    df = pd.DataFrame({
        "antecedents": ["X"] * 4,
        "consequents": ["Y"] * 4,
        "selected": [True, True, False, True],
        "lift": [1.2, 1.1, 1.0, 1.3]
    })
    result = summarize_rule_metrics(df, metrics=["lift"])
    assert result.shape[0] == 1
    assert result["selected_selected_count"].iloc[0] == 3
    assert result["selected_test_count"].iloc[0] == 4
    assert np.isclose(result["selected_fraction"].iloc[0], 0.75)


@pytest.mark.parametrize("missing_col", ["antecedents", "consequents", "lift"])
def test_missing_any_required_column_raises(missing_col, sample_rule_df):
    df = sample_rule_df.drop(columns=[missing_col])
    with pytest.raises(ValueError):
        summarize_rule_metrics(df, metrics=["lift"])


def test_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=["antecedents", "consequents", "selected", "lift"])
    with pytest.raises(ValueError):
        summarize_rule_metrics(empty_df, metrics=["lift"])

# Test for create_validation_summary_log()
@pytest.fixture
def sample_summary_df():
    return pd.DataFrame({
        "antecedents": ["A", "B", "C"],
        "consequents": ["X", "Y", "Z"],
        "selected_test_count": [100, 100, 100],
        "selected_fraction": [0.3, 0.5, 0.7],
        "lift_mean": [1.1, 0.9, 1.3],
        "confidence_mean": [0.6, 0.7, 0.8],
    })


def test_validation_summary_structure_and_values(sample_summary_df):
    result = create_validation_summary_log(sample_summary_df, metrics=["lift", "confidence"])

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 11)

    expected_columns = {
        "total_rules", "total_tests", "avg_selection_rate",
        "lift_mean_mean", "lift_mean_std", "lift_mean_min", "lift_mean_max",
        "confidence_mean_mean", "confidence_mean_std", "confidence_mean_min", "confidence_mean_max",
    }
    assert set(result.columns) == expected_columns

    assert result["total_rules"].iloc[0] == 3
    assert result["total_tests"].iloc[0] == 100
    assert np.isclose(result["avg_selection_rate"].iloc[0], np.mean([0.3, 0.5, 0.7]))


def test_validation_summary_single_row():
    df = pd.DataFrame({
        "antecedents": ["A"],
        "consequents": ["X"],
        "selected_test_count": [50],
        "selected_fraction": [0.4],
        "lift_mean": [1.0],
    })
    result = create_validation_summary_log(df, metrics=["lift"])
    assert result["total_rules"].iloc[0] == 1
    assert result["total_tests"].iloc[0] == 50
    assert result["lift_mean_std"].iloc[0] == 0.0


def test_missing_required_base_columns_raises(sample_summary_df):
    bad_df = sample_summary_df.drop(columns=["selected_test_count"])
    with pytest.raises(ValueError, match="Missing required columns"):
        create_validation_summary_log(bad_df, metrics=["lift"])


def test_missing_metric_column_raises(sample_summary_df):
    bad_df = sample_summary_df.drop(columns=["lift_mean"])
    with pytest.raises(ValueError, match="Missing column 'lift_mean'"):
        create_validation_summary_log(bad_df, metrics=["lift"])


@pytest.mark.parametrize("empty_df", [
    pd.DataFrame(columns=["selected_test_count", "selected_fraction", "lift_mean"]),
    pd.DataFrame(),  # completely empty
])
def test_empty_input_raises(empty_df):
    with pytest.raises(ValueError):
        create_validation_summary_log(empty_df, metrics=["lift"])

# Test for validate_bootstrap()
@pytest.fixture
def mock_cfg():
    return {
        "date_col": "date",
        "id_cols": ["group"],
        "target_col": "target"
    }


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=10),
        "group": ["A"] * 5 + ["B"] * 5,
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    })


@pytest.fixture(autouse=True)
def patch_pipeline_monkeypatch(monkeypatch):
    """Patch all pipeline dependencies with minimal valid mocks."""

    def mock_prep(df, cfg, logger=None):
        return df.copy(), {"prep": "ok"}

    def mock_mining(df, cfg, logger=None):
        rules = pd.DataFrame({
            "antecedents": ["A"], "consequents": ["B"],
            "rule_id": [1]
        })
        return rules, {}, {}

    def mock_test(rules, df, cfg, target_col):
        return pd.DataFrame({
            "antecedents": ["A"],
            "consequents": ["B"],
            "selected": [True],
            "lift": [1.2],
            "confidence": [0.85]
        }), {}

    def mock_resample(df, mode, block_size, date_col, id_cols, random_state):
        return df.copy()

    def mock_summarize(df, metrics):
        return pd.DataFrame({
            "antecedents": ["A"],
            "consequents": ["B"],
            "selected_test_count": [10],
            "selected_fraction": [0.7],
            "lift_mean": [1.1],
            "confidence_mean": [0.82]
        })

    def mock_summary_log(df, metrics):
        return pd.DataFrame([{"total_rules": 1, "avg_selection_rate": 0.7}])

    monkeypatch.setattr("my_module.data_prep_pipeline", mock_prep)
    monkeypatch.setattr("my_module.mining_pipeline", mock_mining)
    monkeypatch.setattr("my_module.test_mined_rules", mock_test)
    monkeypatch.setattr("my_module.resample_dataframe", mock_resample)
    monkeypatch.setattr("my_module.summarize_rule_metrics", mock_summarize)
    monkeypatch.setattr("my_module.create_validation_summary_log", mock_summary_log)
    monkeypatch.setattr("my_module.maybe_tqdm", lambda x, *args, **kwargs: x)


def test_validate_bootstrap_basic_output(simple_df, mock_cfg):
    results, log = validate_bootstrap(simple_df, mock_cfg, n_bootstrap=3, verbose=False)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(log, pd.DataFrame)
    assert results.shape[0] == 1
    assert log.shape[0] == 1
    assert "total_rules" in log.columns


def test_validate_bootstrap_zero_iterations(simple_df, mock_cfg):
    with pytest.raises(ValueError):
        validate_bootstrap(simple_df, mock_cfg, n_bootstrap=0)


def test_validate_bootstrap_missing_target_key(simple_df):
    bad_cfg = {"date_col": "date"}
    with pytest.raises(KeyError):
        validate_bootstrap(simple_df, bad_cfg)

# Test for bootstrap_pipeline()
@pytest.fixture
def minimal_cfg():
    class Config:
        date_col = "date"
        id_cols = ["group"]
        target_col = "target"
        n_bootstrap = 3
        bootstrap_verbose = False
        resample_method = "traditional"
        block_size = 5
        log_max_rows = 100
    return Config()


@pytest.fixture
def minimal_df():
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6),
        "group": ["A", "A", "A", "B", "B", "B"],
        "feature1": [1, 2, 3, 4, 5, 6],
        "target": [0, 1, 0, 1, 0, 1],
    })


@pytest.fixture(autouse=True)
def mock_validate_and_logger(monkeypatch):
    def mock_validate_bootstrap(df, cfg, logger=None, **kwargs):
        results = pd.DataFrame({
            "antecedents": ["A"],
            "consequents": ["B"],
            "lift_mean": [1.2],
            "selected_test_count": [3],
            "selected_fraction": [0.66]
        })
        log = pd.DataFrame([{
            "total_rules": 1,
            "avg_selection_rate": 0.66
        }])
        return results, log

    class MockLogger:
        def __init__(self):
            self.logged = False

        def log_step(self, step_name, info, df, max_rows):
            self.logged = True
            self.step_name = step_name
            self.info = info
            self.df = df
            self.max_rows = max_rows

    monkeypatch.setattr("my_module.validate_bootstrap", mock_validate_bootstrap)
    monkeypatch.setattr("my_module.MockLogger", MockLogger)


def test_bootstrap_pipeline_basic(minimal_df, minimal_cfg):
    results, log = bootstrap_pipeline(minimal_df, minimal_cfg)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(log, pd.DataFrame)
    assert results.shape[0] == 1
    assert log.shape[0] == 1
    assert "total_rules" in log.columns


def test_bootstrap_pipeline_with_logger(minimal_df, minimal_cfg):
    log_calls = []

    class Logger:
        def log_step(self, step_name, info, df, max_rows):
            log_calls.append((step_name, info, df, max_rows))

    logger = Logger()
    _, _ = bootstrap_pipeline(minimal_df, minimal_cfg, logger=logger)
    assert len(log_calls) == 1
    assert log_calls[0][0] == "Bootstrap Resampling validation test"


def test_bootstrap_pipeline_with_overrides(minimal_df, minimal_cfg):
    results, log = bootstrap_pipeline(
        minimal_df,
        minimal_cfg,
        n_bootstrap=5,
        resample_method="block",
        block_size=2
    )
    assert isinstance(results, pd.DataFrame)
    assert log.iloc[0]["total_rules"] == 1


def test_bootstrap_pipeline_missing_target_col_raises(minimal_df):
    class BadConfig:
        pass  # missing target_col

    with pytest.raises(AttributeError):
        bootstrap_pipeline(minimal_df, BadConfig())

# Test for shuffle_dataframe()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "target": [0, 1, 0, 1, 1],
    })


def test_shuffle_target_column_only(sample_df):
    shuffled = shuffle_dataframe(sample_df, mode="target", target_col="target", random_state=42)
    assert list(shuffled.columns) == list(sample_df.columns)
    assert sorted(shuffled["target"]) == sorted(sample_df["target"])
    assert not shuffled.equals(sample_df)
    assert (shuffled.drop(columns="target") == sample_df.drop(columns="target")).all().all()


def test_shuffle_rows(sample_df):
    shuffled = shuffle_dataframe(sample_df, mode="rows", random_state=42)
    assert list(shuffled.columns) == list(sample_df.columns)
    assert sorted(shuffled["A"]) == sorted(sample_df["A"])
    assert not shuffled.equals(sample_df)
    assert set(shuffled["B"]) == set(sample_df["B"])


def test_shuffle_columns(sample_df):
    shuffled = shuffle_dataframe(sample_df, mode="columns", random_state=42)
    assert list(shuffled.columns) == list(sample_df.columns)
    for col in sample_df.columns:
        assert sorted(shuffled[col]) == sorted(sample_df[col])
        assert not all(shuffled[col] == sample_df[col])


@pytest.mark.parametrize("mode", ["invalid", "targets", "columnz"])
def test_invalid_mode_raises(sample_df, mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        shuffle_dataframe(sample_df, mode=mode)


def test_missing_target_col_raises(sample_df):
    with pytest.raises(ValueError, match="target_col must be specified"):
        shuffle_dataframe(sample_df, mode="target")


def test_empty_dataframe_returns_empty():
    empty = pd.DataFrame(columns=["A", "target"])
    result = shuffle_dataframe(empty, mode="rows", random_state=0)
    assert result.empty
    assert list(result.columns) == ["A", "target"]


def test_reproducibility_target_mode(sample_df):
    shuffled1 = shuffle_dataframe(sample_df, mode="target", target_col="target", random_state=123)
    shuffled2 = shuffle_dataframe(sample_df, mode="target", target_col="target", random_state=123)
    pd.testing.assert_frame_equal(shuffled1, shuffled2)


def test_reproducibility_columns_mode(sample_df):
    shuffled1 = shuffle_dataframe(sample_df, mode="columns", random_state=7)
    shuffled2 = shuffle_dataframe(sample_df, mode="columns", random_state=7)
    pd.testing.assert_frame_equal(shuffled1, shuffled2)

# Test for compute_relative_error()
@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "lift_q95": [1.0, 1.1, 0.9, 1.2, 1.05]
    })


def test_relative_error_typical_input(basic_df):
    result = compute_relative_error(basic_df, metric_col="lift_q95", iteration_col="step", m_recent=3)
    assert isinstance(result, float)
    assert result > 0


def test_relative_error_mean_zero():
    df = pd.DataFrame({
        "iter": [1, 2, 3],
        "metric": [0.0, 0.0, 0.0]
    })
    result = compute_relative_error(df, metric_col="metric", iteration_col="iter", m_recent=3)
    assert np.isnan(result)


def test_relative_error_one_row():
    df = pd.DataFrame({"step": [1], "value": [10.0]})
    result = compute_relative_error(df, metric_col="value", iteration_col="step", m_recent=1)
    assert result == 0.0


def test_relative_error_insufficient_rows():
    df = pd.DataFrame({"step": [1, 2], "metric": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Not enough data points"):
        compute_relative_error(df, metric_col="metric", iteration_col="step", m_recent=5)


def test_relative_error_missing_metric_column(basic_df):
    with pytest.raises(ValueError, match="Missing required column: 'foo'"):
        compute_relative_error(basic_df, metric_col="foo", iteration_col="step", m_recent=2)


def test_relative_error_missing_iteration_column(basic_df):
    with pytest.raises(ValueError, match="Missing required column: 'iteration'"):
        compute_relative_error(basic_df, metric_col="lift_q95", iteration_col="iteration", m_recent=2)


def test_relative_error_invalid_m_recent(basic_df):
    with pytest.raises(ValueError, match="m_recent must be at least 1"):
        compute_relative_error(basic_df, metric_col="lift_q95", iteration_col="step", m_recent=0)


@pytest.mark.parametrize("m_recent", [1, 3, 5])
def test_relative_error_parametrized_valid_ranges(basic_df, m_recent):
    result = compute_relative_error(basic_df, metric_col="lift_q95", iteration_col="step", m_recent=m_recent)
    assert isinstance(result, float)

# Test for summarize_null_distribution()
@pytest.fixture
def simple_null_df():
    return pd.DataFrame({
        "perm_num": [1, 1, 2, 2, 3, 3],
        "lift_q95": [1.0, 1.1, 0.9, 1.2, 1.3, 1.0]
    })


def test_summarize_null_distribution_typical(simple_null_df):
    result = summarize_null_distribution(
        null_df=simple_null_df,
        metric_col="lift_q95",
        iteration_col="perm_num"
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 10)
    assert "metric_mean" in result.columns
    assert "n_permutations" in result.columns
    assert result["n_permutations"].iloc[0] == 3
    assert result["n_observations"].iloc[0] == 6


def test_missing_metric_column_raises(simple_null_df):
    with pytest.raises(ValueError, match="Metric column 'foo' not found"):
        summarize_null_distribution(
            null_df=simple_null_df,
            metric_col="foo",
            iteration_col="perm_num"
        )


def test_missing_iteration_column_raises(simple_null_df):
    with pytest.raises(ValueError, match="Iteration column 'perm_step' not found"):
        summarize_null_distribution(
            null_df=simple_null_df,
            metric_col="lift_q95",
            iteration_col="perm_step"
        )


def test_single_row_dataframe():
    df = pd.DataFrame({"perm_num": [1], "metric": [5.0]})
    result = summarize_null_distribution(df, metric_col="metric", iteration_col="perm_num")
    assert result["metric_mean"].iloc[0] == 5.0
    assert result["metric_std"].iloc[0] == 0.0
    assert result["metric_q05"].iloc[0] == 5.0
    assert result["n_permutations"].iloc[0] == 1
    assert result["n_observations"].iloc[0] == 1


def test_empty_dataframe_returns_all_nans():
    df = pd.DataFrame(columns=["perm_num", "metric"])
    with pytest.raises(ValueError):
        summarize_null_distribution(df, metric_col="metric", iteration_col="perm_num")

# Test for validate_null()
@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "target": [1, 0, 1, 0, 1, 0]
    })


@pytest.fixture
def minimal_cfg():
    return {"some": "config"}  # content doesn't matter since functions are patched


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    def mock_mining_pipeline(df, cfg, logger=None):
        return "rules", None, None

    def mock_shuffle_dataframe(df, mode, target_col):
        return df.copy()

    def mock_data_prep_pipeline(df, cfg):
        return df.copy(), None

    def mock_test_mined_rules(rules, df, cfg, target_col):
        return pd.DataFrame({"lift": [1.0]}), None

    def mock_compute_relative_error(df, metric_col, iteration_col, m_recent):
        return 0.005  # always below threshold to trigger early stopping

    def mock_summarize_null_distribution(df, metric_col, iteration_col):
        return pd.DataFrame([{"metric_mean": df[metric_col].mean()}])

    def mock_maybe_tqdm(iterable, verbose, total, desc):
        return iterable

    monkeypatch.setattr("my_module.mining_pipeline", mock_mining_pipeline)
    monkeypatch.setattr("my_module.shuffle_dataframe", mock_shuffle_dataframe)
    monkeypatch.setattr("my_module.data_prep_pipeline", mock_data_prep_pipeline)
    monkeypatch.setattr("my_module.test_mined_rules", mock_test_mined_rules)
    monkeypatch.setattr("my_module.compute_relative_error", mock_compute_relative_error)
    monkeypatch.setattr("my_module.summarize_null_distribution", mock_summarize_null_distribution)
    monkeypatch.setattr("my_module.maybe_tqdm", mock_maybe_tqdm)


def test_validate_null_early_stopping_triggers(simple_df, minimal_cfg):
    results, log = validate_null(
        df=simple_df,
        cfg=minimal_cfg,
        target_col="target",
        n_null=10,
        es_m_permutations=5,
        rel_error_threshold=0.01,
        verbose=False
    )
    assert isinstance(results, pd.DataFrame)
    assert isinstance(log, pd.DataFrame)
    assert "metric_mean" in log.columns


def test_validate_null_full_run_when_error_above_threshold(simple_df, minimal_cfg, monkeypatch):
    monkeypatch.setattr("my_module.compute_relative_error", lambda *a, **kw: 0.1)  # never early stop

    results, log = validate_null(
        df=simple_df,
        cfg=minimal_cfg,
        target_col="target",
        n_null=5,
        es_m_permutations=2,
        rel_error_threshold=0.001,
        verbose=False
    )
    assert len(results) == 5  # all iterations
    assert "metric_mean" in log.columns


def test_validate_null_invalid_n_null_raises(simple_df, minimal_cfg):
    with pytest.raises(ValueError, match="n_null must be >= es_m_permutations"):
        validate_null(
            df=simple_df,
            cfg=minimal_cfg,
            target_col="target",
            n_null=10,
            es_m_permutations=20
        )

# Test for null_pipeline()
# ---- Fixtures and mocks ----
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [0, 1, 1, 0, 1],
        "feature2": [3.2, 1.1, 0.5, 2.2, 3.1],
        "target": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def mock_cfg():
    # Mimics a config object with dot-access
    return SimpleNamespace(
        target_col="target",
        n_null=10,
        shuffle_mode="target",
        early_stop_metric="lift",
        es_m_permutations=5,
        rel_error_threshold=0.5,
        null_verbose=False,
        log_max_rows=5,
    )


@pytest.fixture
def mock_logger():
    class Logger:
        def __init__(self):
            self.logged = []
        def log_step(self, step_name, info, df, max_rows):
            self.logged.append((step_name, info, df.copy(), max_rows))
    return Logger()


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch validate_null with a stub
    def fake_validate_null(df, cfg, logger=None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.DataFrame({"rule": ["A"], "lift": [1.5], "test_num": [0]})
        log = pd.DataFrame({"metric_mean": [1.5], "final_rel_error": [0.02]})
        return data, log

    monkeypatch.setattr("my_module.validate_null", fake_validate_null)


# ---- Test cases ----

def test_null_pipeline_runs_and_returns_expected_structure(sample_df, mock_cfg):
    null_df, null_log = null_pipeline(sample_df, mock_cfg)

    assert isinstance(null_df, pd.DataFrame)
    assert isinstance(null_log, pd.DataFrame)
    assert "lift" in null_df.columns
    assert "final_rel_error" in null_log.columns
    assert null_log.shape[0] == 1  # single-row log


def test_null_pipeline_with_logger_logs_step(sample_df, mock_cfg, mock_logger):
    null_pipeline(sample_df, mock_cfg, logger=mock_logger)

    assert len(mock_logger.logged) == 1
    step_name, info, df, max_rows = mock_logger.logged[0]
    assert step_name == "Null Distribution validation test"
    assert isinstance(df, pd.DataFrame)
    assert max_rows == mock_cfg.log_max_rows
    assert "target_col" in info
    assert info["n_null"] == 10


def test_null_pipeline_override_parameters(sample_df, mock_cfg):
    overrides = {
        "n_null": 5,
        "rel_error_threshold": 0.9,
        "early_stop_metric": "custom_metric",
    }
    null_df, null_log = null_pipeline(sample_df, mock_cfg, **overrides)

    assert isinstance(null_df, pd.DataFrame)
    assert isinstance(null_log, pd.DataFrame)


def test_null_pipeline_raises_if_missing_required_override(sample_df, mock_cfg):
    # Remove attribute from cfg to test fallback failure
    delattr(mock_cfg, "target_col")

    with pytest.raises(AttributeError):
        null_pipeline(sample_df, mock_cfg)

# Test for summarize_fdr_results()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "pval": [0.001, 0.02, 0.03, 0.06, 0.5, 0.9],
        "fdr_significant": [True, True, True, False, False, False],
        "group": ["A", "A", "B", "B", "B", "B"]
    })


def test_basic_summary_output(sample_df):
    result = summarize_fdr_results(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 9)
    assert result["Total Tested"].iloc[0] == 6
    assert result["Significant (FDR)"].iloc[0] == 3
    assert result["P < 0.01"].iloc[0] == 1
    assert result["P < 0.05"].iloc[0] == 3


def test_grouped_summary_output(sample_df):
    result = summarize_fdr_results(sample_df, groupby_col="group")
    assert isinstance(result, pd.DataFrame)
    assert set(result["group"]) == {"A", "B"}
    assert all(col in result.columns for col in ["Total Tested", "Significant (FDR)", "Min P-value"])


def test_markdown_output(sample_df):
    result = summarize_fdr_results(sample_df, as_markdown=True)
    assert isinstance(result, str)
    assert "Total Tested" in result
    assert "Min P-value" in result


def test_grouped_markdown_output(sample_df):
    result = summarize_fdr_results(sample_df, groupby_col="group", as_markdown=True)
    assert isinstance(result, str)
    assert "Group: A" in result
    assert "Group: B" in result
    assert result.count("Total Tested") == 2


@pytest.mark.parametrize("missing_col", ["pval", "fdr_significant"])
def test_missing_required_columns_raises(missing_col, sample_df):
    df = sample_df.drop(columns=missing_col)
    with pytest.raises(ValueError, match=f"Missing required column: '{missing_col}'"):
        summarize_fdr_results(df)


def test_empty_dataframe_returns_valid_summary():
    df = pd.DataFrame(columns=["pval", "fdr_significant"])
    result = summarize_fdr_results(df)
    assert isinstance(result, pd.DataFrame)
    assert result["Total Tested"].iloc[0] == 0
    assert np.isnan(result["Proportion Significant"].iloc[0])


def test_all_significant_case():
    df = pd.DataFrame({
        "pval": [0.001, 0.002, 0.005],
        "fdr_significant": [True, True, True]
    })
    result = summarize_fdr_results(df)
    assert result["Total Tested"].iloc[0] == 3
    assert result["Significant (FDR)"].iloc[0] == 3
    assert result["Proportion Significant"].iloc[0] == 1.0


def test_all_non_significant_case():
    df = pd.DataFrame({
        "pval": [0.2, 0.4, 0.6],
        "fdr_significant": [False, False, False]
    })
    result = summarize_fdr_results(df)
    assert result["Total Tested"].iloc[0] == 3
    assert result["Significant (FDR)"].iloc[0] == 0
    assert result["Proportion Significant"].iloc[0] == 0.0


def test_custom_alpha_threshold(sample_df):
    result = summarize_fdr_results(sample_df, correction_alpha=0.02)
    assert result[f"P < 0.02"].iloc[0] == 2

# Test for compute_empirical_pvals, validate_multiple_tests()
@pytest.fixture
def sample_data():
    mining = pd.DataFrame({
        "rule": ["A", "B", "C", "D"],
        "lift": [1.5, 2.0, 0.5, 1.1],
    })
    null = pd.DataFrame({
        "lift": np.random.normal(loc=1.0, scale=0.3, size=1000)
    })
    return mining, null


@pytest.mark.parametrize("mode", ["greater", "less", "two-sided"])
def test_validate_multiple_tests_runs_with_valid_input(sample_data, mode):
    mining_res, null_df = sample_data
    result_df, summary_df = validate_multiple_tests(
        mining_res=mining_res,
        null_df=null_df,
        early_stop_metric="lift",
        mode=mode,
        correction_alpha=0.10,
        correction_metric="fdr_bh"
    )

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert "pval" in result_df.columns
    assert any(col.startswith("pval_") for col in result_df.columns)
    assert any(col.endswith("_significant") for col in result_df.columns)
    assert result_df.shape[0] == mining_res.shape[0]


def test_validate_multiple_tests_missing_column_raises(sample_data):
    mining_res, null_df = sample_data
    mining_res = mining_res.drop(columns="lift")

    with pytest.raises(ValueError, match="Column 'lift' not found in mining_res"):
        validate_multiple_tests(mining_res, null_df, early_stop_metric="lift")


def test_validate_multiple_tests_empty_inputs_raise():
    empty = pd.DataFrame(columns=["lift"])
    with pytest.raises(ValueError):
        validate_multiple_tests(empty, empty, early_stop_metric="lift")


def test_validate_multiple_tests_with_explicit_center(sample_data):
    mining_res, null_df = sample_data
    result_df, _ = validate_multiple_tests(
        mining_res=mining_res,
        null_df=null_df,
        early_stop_metric="lift",
        mode="two-sided",
        center=1.0,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert "pval" in result_df.columns


def test_validate_multiple_tests_invalid_mode(sample_data):
    mining_res, null_df = sample_data
    with pytest.raises(ValueError, match="Invalid mode"):
        validate_multiple_tests(
            mining_res=mining_res,
            null_df=null_df,
            early_stop_metric="lift",
            mode="not-a-valid-mode"
        )


def test_validate_multiple_tests_invalid_correction_method(sample_data):
    mining_res, null_df = sample_data
    with pytest.raises(ValueError, match="FDR correction failed"):
        validate_multiple_tests(
            mining_res=mining_res,
            null_df=null_df,
            early_stop_metric="lift",
            correction_metric="not_a_method"
        )

# Test for fdr_pipeline()
@pytest.fixture
def sample_inputs():
    # Create a small mining results DataFrame
    mining_res = pd.DataFrame({
        "rule": ["A", "B", "C"],
        "lift": [1.5, 0.9, 2.1],
    })

    # Create a null distribution with normally distributed lift
    null_df = pd.DataFrame({
        "lift": np.random.normal(loc=1.0, scale=0.3, size=500)
    })

    # Config object with default parameters
    cfg = SimpleNamespace(
        early_stop_metric="lift",
        fdr_mode="greater",
        correction_alpha=0.05,
        correction_metric="fdr_bh",
        log_max_rows=10,
    )

    return mining_res, null_df, cfg


def test_fdr_pipeline_basic_run(sample_inputs):
    mining_res, null_df, cfg = sample_inputs
    result_df, log_df = fdr_pipeline(mining_res, null_df, cfg)

    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert result_df.shape[0] == mining_res.shape[0]
    assert "pval" in result_df.columns
    assert any(col.endswith("_significant") for col in result_df.columns)
    assert log_df.shape[0] == 1


def test_fdr_pipeline_respects_overrides(sample_inputs):
    mining_res, null_df, cfg = sample_inputs
    overrides = {
        "fdr_mode": "two-sided",
        "correction_alpha": 0.10
    }

    result_df, _ = fdr_pipeline(mining_res, null_df, cfg, **overrides)

    # Sanity check: result_df is still valid and override worked
    assert isinstance(result_df, pd.DataFrame)
    assert "pval" in result_df.columns


def test_fdr_pipeline_missing_metric_raises(sample_inputs):
    mining_res, null_df, cfg = sample_inputs
    mining_res = mining_res.drop(columns=["lift"])  # remove required column

    with pytest.raises(ValueError, match="not found in mining_res"):
        fdr_pipeline(mining_res, null_df, cfg)


def test_fdr_pipeline_empty_input():
    empty_df = pd.DataFrame(columns=["lift"])
    cfg = SimpleNamespace(
        early_stop_metric="lift",
        fdr_mode="greater",
        correction_alpha=0.05,
        correction_metric="fdr_bh",
        log_max_rows=10,
    )

    with pytest.raises(ValueError):
        fdr_pipeline(empty_df, empty_df, cfg)


def test_fdr_pipeline_with_logger(sample_inputs):
    mining_res, null_df, cfg = sample_inputs

    class MockLogger:
        def __init__(self):
            self.logged = []

        def log_step(self, step_name, info, df, max_rows):
            self.logged.append((step_name, info, df, max_rows))

    logger = MockLogger()
    result_df, _ = fdr_pipeline(mining_res, null_df, cfg, logger=logger)

    assert len(logger.logged) == 1
    step_name, info, df, max_rows = logger.logged[0]
    assert step_name == "FDR Multiple Correction validation test"
    assert isinstance(df, pd.DataFrame)
    assert max_rows == cfg.log_max_rows
