import numpy as np
import pandas as pd
import pytest
from edge_research.target import (
    compute_forward_return,
    merge_features_with_returns,
    summarize_merge,
    bin_target_column,
)

# Test for compute_forward_return()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ticker": ["A"] * 5,
        "date": pd.date_range("2023-01-01", periods=5),
        "adj_close": [100, 105, 110, 115, 120]
    })


@pytest.mark.parametrize("return_mode", ["pct_change", "log_return", "smoothed"])
def test_basic_modes_run_without_error(sample_df, return_mode):
    df = compute_forward_return(sample_df, n_periods=1, return_mode=return_mode)
    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns
    assert "date" in df.columns
    assert "forward_return" in df.columns
    assert not df.empty


def test_smoothed_modes(sample_df):
    for method in ["median", "mean", "max", "min"]:
        df = compute_forward_return(sample_df, n_periods=2, return_mode="smoothed", smoothing_method=method)
        assert "forward_return" in df.columns
        assert not df.empty


def test_pct_change_values(sample_df):
    df = compute_forward_return(sample_df, n_periods=1, return_mode="pct_change")
    expected_returns = (sample_df["adj_close"].shift(-1) - sample_df["adj_close"]) / sample_df["adj_close"]
    # Drop nan to align
    expected_returns = expected_returns.dropna().values
    assert pytest.approx(df["forward_return"].values, rel=1e-6) == expected_returns


def test_log_return_values(sample_df):
    df = compute_forward_return(sample_df, n_periods=1, return_mode="log_return")
    expected_returns = np.log(sample_df["adj_close"].shift(-1) / sample_df["adj_close"])
    expected_returns = expected_returns.dropna().values
    assert pytest.approx(df["forward_return"].values, rel=1e-6) == expected_returns


def test_vol_adjusted_requires_window(sample_df):
    with pytest.raises(ValueError, match="vol_window must be provided"):
        compute_forward_return(sample_df, n_periods=1, return_mode="vol_adjusted")


def test_vol_adjusted_returns(sample_df):
    df = compute_forward_return(sample_df, n_periods=1, return_mode="vol_adjusted", vol_window=2)
    assert "forward_return" in df.columns
    assert not df.empty


def test_invalid_mode(sample_df):
    with pytest.raises(ValueError, match="Unsupported return_mode"):
        compute_forward_return(sample_df, n_periods=1, return_mode="invalid_mode")


def test_invalid_smoothing_method(sample_df):
    with pytest.raises(KeyError):
        compute_forward_return(sample_df, n_periods=1, return_mode="smoothed", smoothing_method="invalid_method")


def test_invalid_missing_columns():
    df = pd.DataFrame({"ticker": ["A"], "date": ["2023-01-01"]})  # Missing adj_close
    with pytest.raises(ValueError, match="missing required columns"):
        compute_forward_return(df, n_periods=1)


def test_invalid_n_periods(sample_df):
    with pytest.raises(ValueError, match="n_periods must be positive"):
        compute_forward_return(sample_df, n_periods=0)


def test_empty_dataframe():
    df = pd.DataFrame(columns=["ticker", "date", "adj_close"])
    result = compute_forward_return(df, n_periods=1)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

# Test for merge_features_with_returns()
@pytest.fixture
def sample_feature_df():
    return pd.DataFrame({
        "ticker": ["a", "b"],
        "feature_date": ["2024-01-01", "2024-01-02"],
        "feature_val": [1.0, 2.0]
    })


@pytest.fixture
def sample_returns_df():
    return pd.DataFrame({
        "ticker": ["A", "B"],
        "returns_date": ["2024-01-01", "2024-01-03"],
        "return": [0.05, 0.10]
    })


def test_basic_forward_merge(sample_feature_df, sample_returns_df):
    merged = merge_features_with_returns(
        feature_df=sample_feature_df,
        returns_df=sample_returns_df,
        id_cols=["ticker"],
        feature_date_col="feature_date",
        returns_date_col="returns_date"
    )
    assert isinstance(merged, pd.DataFrame)
    assert "return" in merged.columns
    assert merged.shape[0] == sample_feature_df.shape[0]
    # A's return should match, B should merge forward to its next available return
    assert merged.loc[0, "return"] == 0.05
    assert merged.loc[1, "return"] == 0.10


@pytest.mark.parametrize("id_case", ["upper", "lower", "original"])
def test_id_case_normalization(sample_feature_df, sample_returns_df, id_case):
    merged = merge_features_with_returns(
        feature_df=sample_feature_df,
        returns_df=sample_returns_df,
        id_cols=["ticker"],
        feature_date_col="feature_date",
        returns_date_col="returns_date",
        id_case=id_case
    )
    assert "return" in merged.columns
    assert merged.shape[0] == sample_feature_df.shape[0]


@pytest.mark.parametrize("direction", ["forward", "backward", "nearest"])
def test_merge_direction_modes(sample_feature_df, sample_returns_df, direction):
    merged = merge_features_with_returns(
        feature_df=sample_feature_df,
        returns_df=sample_returns_df,
        id_cols=["ticker"],
        feature_date_col="feature_date",
        returns_date_col="returns_date",
        direction=direction
    )
    assert merged.shape[0] == sample_feature_df.shape[0]


def test_invalid_direction_raises(sample_feature_df, sample_returns_df):
    with pytest.raises(ValueError, match="direction must be one of"):
        merge_features_with_returns(
            feature_df=sample_feature_df,
            returns_df=sample_returns_df,
            id_cols=["ticker"],
            feature_date_col="feature_date",
            returns_date_col="returns_date",
            direction="invalid"
        )


def test_invalid_id_case_raises(sample_feature_df, sample_returns_df):
    with pytest.raises(ValueError, match="id_case must be one of"):
        merge_features_with_returns(
            feature_df=sample_feature_df,
            returns_df=sample_returns_df,
            id_cols=["ticker"],
            feature_date_col="feature_date",
            returns_date_col="returns_date",
            id_case="bad_case"
        )


def test_unparsable_dates_raise():
    df = pd.DataFrame({"ticker": ["A"], "feature_date": ["not_a_date"], "feature_val": [1]})
    returns_df = pd.DataFrame({"ticker": ["A"], "returns_date": ["2024-01-01"], "return": [0.1]})
    with pytest.raises(ValueError, match="feature_date contains unparsable dates"):
        merge_features_with_returns(
            feature_df=df,
            returns_df=returns_df,
            id_cols=["ticker"],
            feature_date_col="feature_date",
            returns_date_col="returns_date"
        )


def test_empty_inputs():
    feature_df = pd.DataFrame(columns=["ticker", "feature_date", "feature_val"])
    returns_df = pd.DataFrame(columns=["ticker", "returns_date", "return"])
    merged = merge_features_with_returns(
        feature_df=feature_df,
        returns_df=returns_df,
        id_cols=["ticker"],
        feature_date_col="feature_date",
        returns_date_col="returns_date"
    )
    assert merged.empty


def test_tolerance_parameter(sample_feature_df, sample_returns_df):
    merged = merge_features_with_returns(
        feature_df=sample_feature_df,
        returns_df=sample_returns_df,
        id_cols=["ticker"],
        feature_date_col="feature_date",
        returns_date_col="returns_date",
        tolerance=pd.Timedelta("2D")
    )
    assert "return" in merged.columns

# Test for summarize_merge()
@pytest.fixture
def sample_merged_df():
    return pd.DataFrame({
        "ticker": ["A", "A", "B", "B", "C"],
        "feature_date": pd.date_range("2024-01-01", periods=5),
        "forward_return": [0.05, None, 0.10, None, None]
    })


def test_typical_summary(sample_merged_df):
    summary, counts_per_id, unmatched_sample = summarize_merge(
        merged=sample_merged_df,
        id_cols=["ticker"],
        returns_date_col="feature_date",  # Not used in logic but required in signature
        feature_date_col="feature_date",
        target_col="forward_return",
        sample_size=2
    )

    # Check summary dictionary
    assert summary["total_rows"] == 5
    assert summary["matched_rows"] == 2
    assert summary["unmatched_rows"] == 3

    # Check counts per ID
    assert set(counts_per_id.columns) == {"ticker", "total", "matched", "unmatched"}
    assert counts_per_id["total"].sum() == 5
    assert counts_per_id["matched"].sum() == 2
    assert counts_per_id["unmatched"].sum() == 3

    # Check unmatched sample
    assert isinstance(unmatched_sample, pd.DataFrame)
    assert len(unmatched_sample) <= 2
    assert set(unmatched_sample.columns) == {"ticker", "feature_date"}


def test_empty_input():
    empty_df = pd.DataFrame(columns=["ticker", "feature_date", "forward_return"])
    summary, counts_per_id, unmatched_sample = summarize_merge(
        merged=empty_df,
        id_cols=["ticker"],
        returns_date_col="feature_date",
        feature_date_col="feature_date",
        target_col="forward_return"
    )

    assert summary["total_rows"] == 0
    assert summary["matched_rows"] == 0
    assert summary["unmatched_rows"] == 0
    assert counts_per_id.empty
    assert unmatched_sample.empty


def test_missing_required_columns_raises():
    bad_df = pd.DataFrame({
        "ticker": ["A", "B"],
        "feature_date": pd.date_range("2024-01-01", periods=2),
        # Missing 'forward_return'
    })

    with pytest.raises(ValueError, match="Merged dataframe missing required columns"):
        summarize_merge(
            merged=bad_df,
            id_cols=["ticker"],
            returns_date_col="feature_date",
            feature_date_col="feature_date",
            target_col="forward_return"
        )


def test_single_row_input():
    df = pd.DataFrame({
        "ticker": ["A"],
        "feature_date": ["2024-01-01"],
        "forward_return": [None]
    })

    summary, counts_per_id, unmatched_sample = summarize_merge(
        merged=df,
        id_cols=["ticker"],
        returns_date_col="feature_date",
        feature_date_col="feature_date",
        target_col="forward_return"
    )

    assert summary["total_rows"] == 1
    assert summary["matched_rows"] == 0
    assert summary["unmatched_rows"] == 1
    assert len(unmatched_sample) == 1
    assert counts_per_id.iloc[0]["total"] == 1
    assert counts_per_id.iloc[0]["matched"] == 0
    assert counts_per_id.iloc[0]["unmatched"] == 1


@pytest.mark.parametrize("sample_size", [0, 1, 5, 10])
def test_sample_size_respected(sample_merged_df, sample_size):
    _, _, unmatched_sample = summarize_merge(
        merged=sample_merged_df,
        id_cols=["ticker"],
        returns_date_col="feature_date",
        feature_date_col="feature_date",
        target_col="forward_return",
        sample_size=sample_size
    )
    assert len(unmatched_sample) <= sample_size

# Test for bin_target_column()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": ["A", "A", "B", "B", "C"],
        "date": pd.date_range("2024-01-01", periods=5),
        "return": [-0.1, 0.0, 0.05, 0.1, 0.2]
    })


@pytest.mark.parametrize("binning_method", ["binary", "custom", "quantile"])
def test_returns_dataframe_and_log(sample_df, binning_method):
    bins = [0] if binning_method == "binary" else [-0.05, 0.05] if binning_method == "custom" else [0.5]
    labels = ["pos", "neg"] if binning_method == "binary" else ["low", "high"]

    df_binned, log_df = bin_target_column(
        sample_df,
        binning_method=binning_method,
        bins=bins,
        labels=labels,
        target_col="return",
        id_cols=["id"],
        date_col="date",
        grouping="none",
        n_datetime_units=None,
        nan_placeholder="missing"
    )

    assert isinstance(df_binned, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert "return" in df_binned.columns
    assert not log_df.empty


def test_binary_binning_logic(sample_df):
    bins = [0.05]
    labels = ["above", "below"]

    df_binned, log_df = bin_target_column(
        sample_df,
        binning_method="binary",
        bins=bins,
        labels=labels,
        target_col="return"
    )

    assert set(df_binned["return"].unique()) <= set(labels)
    assert log_df.iloc[0]["method"] == "binary"


def test_custom_binning_works(sample_df):
    bins = [-np.inf, -0.05, 0.05, np.inf]
    labels = ["down", "flat", "up"]

    df_binned, log_df = bin_target_column(
        sample_df,
        binning_method="custom",
        bins=bins,
        labels=labels,
        target_col="return"
    )

    assert set(df_binned["return"].unique()) <= set(labels)
    assert log_df.iloc[0]["method"] == "custom"


def test_custom_binning_label_mismatch_fills_placeholder(sample_df):
    # Incorrect: 2 edges define 1 interval but 3 labels provided
    bins = [-0.05, 0.05]
    labels = ["down", "up", "sideways"]  # mismatch

    df_binned, _ = bin_target_column(
        sample_df,
        binning_method="custom",
        bins=bins,
        labels=labels,
        target_col="return",
        nan_placeholder="missing"
    )

    assert all(df_binned["return"] == "missing")


def test_quantile_binning_grouping_none(sample_df):
    bins = [0.5]  # Median split
    labels = ["low", "high"]

    df_binned, log_df = bin_target_column(
        sample_df,
        binning_method="quantile",
        bins=bins,
        labels=labels,
        target_col="return",
        id_cols=["id"],
        date_col="date",
        grouping="none"
    )

    assert set(df_binned["return"].unique()).issubset(set(labels) | {"no_data"})
    assert "quantile" in log_df["method"].values[0]


def test_invalid_binning_method_raises(sample_df):
    with pytest.raises(ValueError, match="binning_method must be"):
        bin_target_column(
            sample_df,
            binning_method="invalid",
            bins=[0],
            labels=["label"]
        )


def test_invalid_grouping_raises(sample_df):
    with pytest.raises(ValueError, match="Invalid grouping"):
        bin_target_column(
            sample_df,
            binning_method="quantile",
            bins=[0.5],
            labels=["low", "high"],
            target_col="return",
            id_cols=["id"],
            date_col="date",
            grouping="bad_grouping"
        )


def test_quantile_binning_missing_ids_raises(sample_df):
    with pytest.raises(ValueError, match="id_cols must be provided"):
        bin_target_column(
            sample_df,
            binning_method="quantile",
            bins=[0.5],
            labels=["low", "high"],
            target_col="return",
            grouping="ids"
        )


def test_quantile_binning_missing_date_col_raises(sample_df):
    with pytest.raises(ValueError, match="date_col and n_datetime_units must be provided"):
        bin_target_column(
            sample_df,
            binning_method="quantile",
            bins=[0.5],
            labels=["low", "high"],
            target_col="return",
            id_cols=["id"],
            grouping="datetime",
            n_datetime_units=None
        )


def test_empty_dataframe_returns_empty_log():
    empty_df = pd.DataFrame(columns=["id", "date", "return"])

    df_binned, log_df = bin_target_column(
        empty_df,
        binning_method="binary",
        bins=[0],
        labels=["pos", "neg"],
        target_col="return"
    )

    assert df_binned.empty
    assert not log_df.empty
