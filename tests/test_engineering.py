import pytest
import pandas as pd
import numpy as np

from edge_research.engineering import (
    generate_ratio_features, 
    MAX_REPLACEMENT_DEFAULT,
    generate_temporal_pct_change,
    extract_date_features,
    bin_columns_flexible,
    sweep_low_count_bins,
    one_hot_encode_features,
    generate_and_encode_temporal_trends,
    engineer_features,
    encode_data,
    engineer_pipeline,
    validate_pipeline_input
)

# Test for generate_ratio_features()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 0],   # Contains zero to trigger inf replacement
        'C': [7, 8, 9],
        'non_numeric': ['x', 'y', 'z']
    })


def test_generate_ratios_with_all_columns(sample_df):
    df_new, log_df = generate_ratio_features(sample_df, columns="all")

    # Expect 3 columns: A/B, A/C, B/C (3 choose 2)
    assert log_df.shape[0] == 3
    assert all(col in df_new.columns for col in log_df['new_column'])

    # Check that infs replaced correctly
    ratio_col = log_df.query("numerator == 'B' and denominator == 'C'")['new_column'].iloc[0]
    assert df_new[ratio_col].isnull().sum() == 0
    assert (df_new[ratio_col].abs() <= MAX_REPLACEMENT_DEFAULT).all()


def test_generate_ratios_with_manual_columns(sample_df):
    df_new, log_df = generate_ratio_features(sample_df, columns=['A', 'B'])

    assert log_df.shape[0] == 1  # Only A/B
    assert log_df.iloc[0]['numerator'] == 'A'
    assert log_df.iloc[0]['denominator'] == 'B'
    assert log_df.iloc[0]['new_column'] in df_new.columns


def test_invalid_columns_argument(sample_df):
    with pytest.raises(TypeError):
        generate_ratio_features(sample_df, columns=123)


def test_single_column_input_raises(sample_df):
    with pytest.raises(ValueError):
        generate_ratio_features(sample_df, columns=['A'])


def test_nonexistent_column_in_manual_list(sample_df):
    # Should silently ignore missing columns but fail gracefully
    with pytest.raises(ValueError):
        generate_ratio_features(sample_df, columns=['A', 'missing_col'])


@pytest.mark.parametrize("replacement_val", [42.0, 9999.0])
def test_custom_max_replacement_value(sample_df, replacement_val):
    df_new, log_df = generate_ratio_features(
        sample_df, columns="all", max_replacement=replacement_val
    )
    # Ensure no infinities remain and max replacement applied
    assert (df_new[log_df.iloc[0]['new_column']].abs() <= replacement_val).all()


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        generate_ratio_features(empty_df, columns="all")


def test_non_numeric_columns_only():
    df = pd.DataFrame({'A': ['x', 'y', 'z']})
    with pytest.raises(ValueError):
        generate_ratio_features(df, columns="all")

# Test for generate_temporal_pct_change()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'entity': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03',
                                '2024-01-01', '2024-01-02', '2024-01-03']),
        'feature1': [10, 20, 30, 100, 200, 300],
        'feature2': [1, 2, 3, 10, 20, 30],
        'non_numeric': ['x', 'y', 'z', 'x', 'y', 'z']
    })


def test_generate_pct_change_with_all_columns(sample_df):
    df_out, log_df = generate_temporal_pct_change(
        sample_df,
        columns="all",
        id_cols=['entity'],
        datetime_col='date',
        n_dt=1
    )

    # Should generate two new columns (feature1, feature2)
    assert log_df.shape[0] == 2
    assert all(new_col in df_out.columns for new_col in log_df['new_column'])

    # Resulting new columns should contain NaNs after group shifts
    for new_col in log_df['new_column']:
        assert df_out[new_col].isnull().any()


def test_generate_pct_change_with_manual_columns(sample_df):
    df_out, log_df = generate_temporal_pct_change(
        sample_df,
        columns=['feature1'],
        id_cols=['entity'],
        datetime_col='date',
        n_dt=1
    )

    # Only one new column expected
    assert log_df.shape[0] == 1
    assert log_df.iloc[0]['original_column'] == 'feature1'
    assert log_df.iloc[0]['new_column'] in df_out.columns


def test_invalid_columns_argument_raises(sample_df):
    with pytest.raises(TypeError):
        generate_temporal_pct_change(sample_df, columns=123, datetime_col='date')


def test_missing_datetime_column_raises(sample_df):
    with pytest.raises(ValueError):
        generate_temporal_pct_change(sample_df, columns='all')


def test_empty_dataframe():
    df_empty = pd.DataFrame(columns=['id', 'date'])
    with pytest.raises(ValueError):
        generate_temporal_pct_change(df_empty, columns="all", datetime_col='date')


def test_single_numeric_column(sample_df):
    # Should succeed with single column
    df_out, log_df = generate_temporal_pct_change(
        sample_df,
        columns=['feature1'],
        id_cols=['entity'],
        datetime_col='date'
    )
    assert log_df.shape[0] == 1


@pytest.mark.parametrize("n_dt", [1, 2, 3])
def test_pct_change_lags(sample_df, n_dt):
    df_out, log_df = generate_temporal_pct_change(
        sample_df,
        columns=['feature1'],
        id_cols=['entity'],
        datetime_col='date',
        n_dt=n_dt
    )
    assert df_out.shape[0] == sample_df.shape[0]
    assert all(new_col in df_out.columns for new_col in log_df['new_column'])

# Test for extract_date_features()
@pytest.fixture
def simple_dates_df():
    return pd.DataFrame({
        'date_col': pd.date_range(start='2024-01-01', periods=5),
        'value': [1, 2, 3, 4, 5]
    })


def test_basic_extraction(simple_dates_df):
    df_out = extract_date_features(simple_dates_df, date_col='date_col')
    assert isinstance(df_out, pd.DataFrame)

    expected_columns = [
        'dt_year', 'dt_quarter', 'dt_month', 'dt_week', 'dt_weekday',
        'dt_is_month_end', 'dt_is_month_start',
        'dt_is_quarter_end', 'dt_is_quarter_start',
        'dt_is_year_end', 'dt_is_year_start'
    ]

    for col in expected_columns:
        assert col in df_out.columns
        assert pd.api.types.is_integer_dtype(df_out[col]) or pd.api.types.is_numeric_dtype(df_out[col])


def test_custom_prefix(simple_dates_df):
    df_out = extract_date_features(simple_dates_df, date_col='date_col', prefix='time_')
    assert 'time_year' in df_out.columns
    assert 'time_is_month_end' in df_out.columns


def test_non_datetime_column_conversion():
    df = pd.DataFrame({'some_col': ['2024-01-01', '2024-01-02']})
    df_out = extract_date_features(df, date_col='some_col')
    assert 'dt_year' in df_out.columns
    assert df_out['dt_year'].iloc[0] == 2024


def test_missing_date_column_raises():
    df = pd.DataFrame({'other_col': [1, 2, 3]})
    with pytest.raises(ValueError, match="column not found"):
        extract_date_features(df, date_col='nonexistent')


def test_invalid_date_column_raises():
    df = pd.DataFrame({'bad_dates': ['foo', 'bar', 'baz']})
    with pytest.raises(ValueError, match="Cannot convert"):
        extract_date_features(df, date_col='bad_dates')


def test_empty_dataframe_handled_gracefully():
    df_empty = pd.DataFrame(columns=['timestamp'])
    with pytest.raises(ValueError):
        extract_date_features(df_empty, date_col='timestamp')


@pytest.mark.parametrize("edge_date", [
    pd.Timestamp('2024-01-01'),
    pd.Timestamp('2024-12-31'),
    pd.Timestamp('2024-06-30'),
])
def test_flags_are_binary(edge_date):
    df = pd.DataFrame({'date_col': [edge_date]})
    df_out = extract_date_features(df, date_col='date_col')

    flag_cols = [c for c in df_out.columns if 'is_' in c]
    for col in flag_cols:
        assert df_out[col].isin([0, 1]).all()

# Test for bin_columns_flexible()
@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'entity': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': pd.date_range('2024-01-01', periods=6),
        'feature1': [10, 20, 30, 5, 15, 25],
        'feature2': [0, 1, 2, 3, 4, 5],
        'text_column': ['x', 'y', 'z', 'x', 'y', 'z']
    })


def test_no_grouping_auto_columns(simple_df):
    df_binned, log_df = bin_columns_flexible(simple_df, columns="all")

    assert isinstance(df_binned, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)

    # Should produce bins for numeric columns (feature1, feature2)
    assert log_df['column'].isin(['feature1', 'feature2']).all()

    # All binned columns should be strings after binning
    for col in ['feature1', 'feature2']:
        assert df_binned[col].dtype == object
        assert df_binned[col].isnull().sum() == 0


def test_manual_columns_binning(simple_df):
    df_binned, log_df = bin_columns_flexible(
        simple_df,
        columns=['feature1'],
        quantiles=[0, 0.5, 1.0],
        quantile_labels=["low", "high"]
    )

    assert 'feature1' in df_binned.columns
    assert df_binned['feature1'].isin(["low", "high", "no_data"]).any()


@pytest.mark.parametrize("grouping_mode", ["none", "ids", "datetime", "datetime+ids"])
def test_grouping_modes(simple_df, grouping_mode):
    kwargs = {"columns": ["feature1"], "grouping": grouping_mode}

    if grouping_mode in {"ids", "datetime+ids"}:
        kwargs["id_cols"] = ["entity"]
    if grouping_mode in {"datetime", "datetime+ids"}:
        kwargs["date_col"] = "date"
        kwargs["n_datetime_units"] = 2

    df_binned, log_df = bin_columns_flexible(simple_df, **kwargs)

    assert 'feature1' in df_binned.columns
    assert isinstance(df_binned, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)


def test_invalid_grouping_raises(simple_df):
    with pytest.raises(ValueError):
        bin_columns_flexible(simple_df, grouping="invalid")


def test_missing_id_cols_for_grouping_raises(simple_df):
    with pytest.raises(ValueError):
        bin_columns_flexible(simple_df, grouping="ids")


def test_missing_datetime_args_for_grouping_raises(simple_df):
    with pytest.raises(ValueError):
        bin_columns_flexible(simple_df, grouping="datetime", date_col=None)


def test_nan_placeholder_behavior(simple_df):
    # Use quantiles that force ValueError (e.g., constant column)
    df_const = simple_df.copy()
    df_const['feature1'] = 1

    df_binned, _ = bin_columns_flexible(df_const, columns=["feature1"], nan_placeholder="missing_bin")

    assert df_binned['feature1'].isin(["missing_bin"]).all()


def test_invalid_columns_argument_raises(simple_df):
    with pytest.raises(TypeError):
        bin_columns_flexible(simple_df, columns=123)


def test_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=['id', 'value'])
    with pytest.raises(ValueError):
        bin_columns_flexible(empty_df, columns="all")

# Test for sweep_low_count_bins()
@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'cat_col': ['A', 'A', 'B', 'C', 'C', 'D', 'E', 'E', 'E', 'E'],
        'other_col': ['x', 'y', 'x', 'y', 'z', 'x', 'x', 'y', 'z', 'z']
    })


def test_basic_sweeping_min_count(simple_df):
    df_out, log_df = sweep_low_count_bins(
        simple_df,
        columns=['cat_col'],
        min_count=3,
        reserved_labels=['D']
    )

    # A, B, C should be swept (all < 3), except D (reserved), E remains
    assert (df_out['cat_col'] == 'others').sum() >= 1
    assert 'D' in df_out['cat_col'].unique()
    assert 'E' in df_out['cat_col'].unique()

    # Log should report swept bins, excluding D
    swept_bins = log_df['bin_swept'].tolist()
    assert 'A' in swept_bins
    assert 'B' in swept_bins
    assert 'C' in swept_bins
    assert 'D' not in swept_bins


def test_fraction_threshold_sweeping(simple_df):
    df_out, log_df = sweep_low_count_bins(
        simple_df,
        columns=['cat_col'],
        min_fraction=0.2  # Anything < 20% swept
    )
    # E dominates - others should be swept
    assert 'others' in df_out['cat_col'].unique()
    assert 'E' in df_out['cat_col'].unique()


def test_combined_threshold_maximum_logic(simple_df):
    df_out, log_df = sweep_low_count_bins(
        simple_df,
        columns=['cat_col'],
        min_count=2,
        min_fraction=0.1  # Should pick max threshold between count and fraction
    )
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert 'others' in df_out['cat_col'].values or len(log_df) > 0


def test_columns_all_infers_categoricals(simple_df):
    df_copy = simple_df.copy()
    df_copy['cat_col'] = pd.Categorical(df_copy['cat_col'])

    df_out, log_df = sweep_low_count_bins(
        df_copy,
        columns="all",
        min_count=3
    )
    assert isinstance(df_out, pd.DataFrame)
    assert 'others' in df_out['cat_col'].unique()


def test_reserved_label_preserved(simple_df):
    df_out, log_df = sweep_low_count_bins(
        simple_df,
        columns=['cat_col'],
        min_count=10,
        reserved_labels=['E']
    )
    assert 'E' in df_out['cat_col'].unique()
    assert 'E' not in log_df['bin_swept'].unique()


def test_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=['col'])
    with pytest.raises(ValueError):
        sweep_low_count_bins(empty_df, columns=['col'], min_count=1)


def test_invalid_columns_argument_raises(simple_df):
    with pytest.raises(TypeError):
        sweep_low_count_bins(simple_df, columns=123, min_count=1)


@pytest.mark.parametrize("colspec", ["all", ["cat_col"], ["other_col"]])
def test_columns_handling_variations(simple_df, colspec):
    df_out, log_df = sweep_low_count_bins(
        simple_df,
        columns=colspec,
        min_count=3
    )
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)


def test_sweep_label_custom(simple_df):
    df_out, _ = sweep_low_count_bins(
        simple_df,
        columns=['cat_col'],
        min_count=3,
        sweep_label='RARE_BIN'
    )
    assert 'RARE_BIN' in df_out['cat_col'].unique()

# Test for one_hot_encode_features()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'feature': ['A', 'B', 'no_data'],
        'drop_this': ['x', 'y', 'z']
    })


def test_basic_encoding(sample_df):
    df_out = one_hot_encode_features(
        sample_df,
        id_cols=['id'],
        date_col='date',
        drop_cols=['drop_this']
    )

    # ID, date, and drop_this should remain unchanged
    for col in ['id', 'date', 'drop_this']:
        assert col in df_out.columns

    # feature should be one-hot encoded
    assert any('feature=' in col for col in df_out.columns)

    # Result should have same number of rows
    assert len(df_out) == len(sample_df)


def test_drop_no_data_column(sample_df):
    df_out = one_hot_encode_features(
        sample_df,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        no_data_label='no_data',
        drop_no_data_columns=True
    )

    no_data_columns = [col for col in df_out.columns if "=no_data" in col]
    assert len(no_data_columns) == 0


def test_retain_no_data_column(sample_df):
    df_out = one_hot_encode_features(
        sample_df,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        no_data_label='no_data',
        drop_no_data_columns=False
    )

    no_data_columns = [col for col in df_out.columns if "=no_data" in col]
    assert len(no_data_columns) >= 1


def test_all_columns_excluded(sample_df):
    # Remove all columns from encoding
    df_out = one_hot_encode_features(
        sample_df,
        id_cols=['id', 'feature'],
        date_col='date',
        drop_cols=['drop_this']
    )

    # Should contain only retained columns
    assert set(df_out.columns) == {'id', 'date', 'drop_this'}


def test_empty_dataframe():
    empty_df = pd.DataFrame(columns=['id', 'date', 'feature'])
    df_out = one_hot_encode_features(
        empty_df,
        id_cols=['id'],
        date_col='date',
        drop_cols=[]
    )
    assert df_out.empty


@pytest.mark.parametrize("drop_flag", [True, False])
def test_no_data_drop_toggle(sample_df, drop_flag):
    df_out = one_hot_encode_features(
        sample_df,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        drop_no_data_columns=drop_flag
    )
    assert isinstance(df_out, pd.DataFrame)
    # Basic structure should always hold
    assert len(df_out) == len(sample_df)

# Test for generate_and_encode_temporal_trends()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'ticker': ['A', 'A', 'A', 'B', 'B'],
        'date': pd.date_range(start='2022-01-01', periods=5),
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })


@pytest.mark.parametrize("return_mode", ["combined_only", "encoded_and_combined", "raw_and_combined"])
def test_typical_usage(sample_df, return_mode):
    df_out, log = generate_and_encode_temporal_trends(
        df=sample_df,
        n_dt_list=[1, 2],
        columns=["feature1", "feature2"],
        id_cols=["ticker"],
        datetime_col="date",
        flat_threshold=[-0.01, 0.01],
        return_mode=return_mode
    )

    # Output dataframe should match input row count
    assert len(df_out) == len(sample_df)

    # Log dataframe should have expected structure
    assert set(log.columns) == {"original_column", "new_column", "n_lag"}
    assert len(log) > 0

    # Combined columns must exist
    for col in ["feature1", "feature2"]:
        combined_col = f"{col}_combined_trend"
        assert combined_col in df_out.columns

    if return_mode == "combined_only":
        for col in ["feature1", "feature2"]:
            assert all(c not in df_out.columns for c in df_out.columns if c.startswith(f"{col}_") and "combined" not in c)

    elif return_mode == "encoded_and_combined":
        for col in ["feature1", "feature2"]:
            assert any(c.startswith(f"{col}_") and "_trend" in c for c in df_out.columns)

    elif return_mode == "raw_and_combined":
        for col in ["feature1", "feature2"]:
            assert any(c.startswith(f"{col}_") and "_pctchange" in c for c in df_out.columns)


def test_empty_input():
    empty_df = pd.DataFrame(columns=['ticker', 'date', 'feature1'])
    df_out, log = generate_and_encode_temporal_trends(
        df=empty_df,
        n_dt_list=[1],
        columns=["feature1"],
        id_cols=["ticker"],
        datetime_col="date"
    )
    assert df_out.empty
    assert log.empty


def test_invalid_return_mode(sample_df):
    with pytest.raises(AssertionError, match="Invalid return_mode"):
        generate_and_encode_temporal_trends(
            df=sample_df,
            n_dt_list=[1],
            columns="all",
            id_cols=["ticker"],
            datetime_col="date",
            return_mode="bad_mode"
        )


def test_invalid_threshold(sample_df):
    with pytest.raises(AssertionError, match="flat_threshold must be a list of two values"):
        generate_and_encode_temporal_trends(
            df=sample_df,
            n_dt_list=[1],
            columns="all",
            id_cols=["ticker"],
            datetime_col="date",
            flat_threshold=[0.0]  # Invalid: only one value
        )


def test_edge_case_single_row():
    df = pd.DataFrame({
        'ticker': ['A'],
        'date': pd.to_datetime(['2022-01-01']),
        'feature1': [42]
    })
    df_out, log = generate_and_encode_temporal_trends(
        df=df,
        n_dt_list=[1, 2, 5],
        columns=["feature1"],
        id_cols=["ticker"],
        datetime_col="date"
    )
    assert len(df_out) == 1
    assert len(log) == 3  # One entry per lag


@pytest.mark.parametrize("n_dt_list", [[1], [1, 2, 3], [5, 10]])
def test_various_lag_intervals(sample_df, n_dt_list):
    df_out, log = generate_and_encode_temporal_trends(
        df=sample_df,
        n_dt_list=n_dt_list,
        columns=["feature1"],
        id_cols=["ticker"],
        datetime_col="date"
    )
    assert len(log) == len(n_dt_list)

# Test for engineer_features()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0]
    })


@pytest.mark.parametrize("engineer_cols", ["base", "all"])
@pytest.mark.parametrize("to_engineer_dates", [True, False])
@pytest.mark.parametrize("to_engineer_ratios", [True, False])
@pytest.mark.parametrize("to_engineer_lags", [True, False])
@pytest.mark.parametrize("lag_mode", ["raw_only", "combined_only", "encoded_and_combined", "raw_and_combined"])
def test_engineer_features_typical_usage(
    sample_df,
    engineer_cols,
    to_engineer_dates,
    to_engineer_ratios,
    to_engineer_lags,
    lag_mode
):
    df_out, logs = engineer_features(
        df=sample_df,
        date_col='date',
        id_cols=['id'],
        engineer_cols=engineer_cols,
        to_engineer_dates=to_engineer_dates,
        to_engineer_ratios=to_engineer_ratios,
        to_engineer_lags=to_engineer_lags,
        lag_mode=lag_mode,
        n_dt_list=[1, 2],
        flat_threshold=[-0.01, 0.01]
    )

    # Output dataframe should retain row count
    assert len(df_out) == len(sample_df)

    # Logs should be consistent with requested features
    expected_keys = []
    if to_engineer_dates:
        expected_keys.append('date_log')
    if to_engineer_ratios:
        expected_keys.append('ratio_log')
    if to_engineer_lags:
        expected_keys.append('lag_log')

    assert set(logs.keys()) == set(expected_keys)


def test_engineer_features_empty_dataframe():
    empty_df = pd.DataFrame(columns=['id', 'date', 'feature1'])
    df_out, logs = engineer_features(
        df=empty_df,
        date_col='date',
        id_cols=['id'],
        engineer_cols='base',
        to_engineer_dates=True,
        to_engineer_ratios=True,
        to_engineer_lags=True,
        lag_mode='raw_only',
        n_dt_list=[1],
        flat_threshold=[-0.01, 0.01]
    )
    assert df_out.empty
    for log_df in logs.values():
        assert log_df.empty


@pytest.mark.parametrize("bad_engineer_cols", ["invalid", 123, None])
def test_invalid_engineer_cols_raises(sample_df, bad_engineer_cols):
    with pytest.raises(ValueError, match="Invalid engineer_cols"):
        engineer_features(
            df=sample_df,
            date_col='date',
            id_cols=['id'],
            engineer_cols=bad_engineer_cols,
            to_engineer_dates=True,
            to_engineer_ratios=False,
            to_engineer_lags=False,
            lag_mode='raw_only',
            n_dt_list=[1],
            flat_threshold=[-0.01, 0.01]
        )


@pytest.mark.parametrize("bad_lag_mode", ["not_a_mode", 42, None])
def test_invalid_lag_mode_raises(sample_df, bad_lag_mode):
    with pytest.raises(ValueError, match="Invalid lag_mode"):
        engineer_features(
            df=sample_df,
            date_col='date',
            id_cols=['id'],
            engineer_cols='base',
            to_engineer_dates=False,
            to_engineer_ratios=False,
            to_engineer_lags=True,
            lag_mode=bad_lag_mode,
            n_dt_list=[1],
            flat_threshold=[-0.01, 0.01]
        )


def test_single_row_edge_case():
    df = pd.DataFrame({
        'id': ['X'],
        'date': pd.to_datetime(['2024-01-01']),
        'feature1': [100]
    })
    df_out, logs = engineer_features(
        df=df,
        date_col='date',
        id_cols=['id'],
        engineer_cols='base',
        to_engineer_dates=True,
        to_engineer_ratios=True,
        to_engineer_lags=True,
        lag_mode='combined_only',
        n_dt_list=[1],
        flat_threshold=[-0.01, 0.01]
    )
    assert len(df_out) == 1
    assert isinstance(logs, dict)
    assert all(isinstance(log_df, pd.DataFrame) for log_df in logs.values())

# Test for encode_data()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0]
    })


@pytest.mark.parametrize("to_sweep", [True, False])
@pytest.mark.parametrize("to_drop_no_data", [True, False])
@pytest.mark.parametrize("bin_cols", ["all", ["feature1"]])
def test_encode_data_typical_usage(sample_df, to_sweep, to_drop_no_data, bin_cols):
    ohe_df, logs = encode_data(
        df=sample_df,
        bin_cols=bin_cols,
        bin_quantiles=[0, 0.25, 0.5, 0.75, 1.0],
        bin_quantile_labels=None,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        bin_grouping='none',
        bin_dt_units=4,
        to_sweep=to_sweep,
        to_drop_no_data=to_drop_no_data,
        min_bin_obs=1,
        min_bin_fraction=0.01,
        lag_num_missing=1
    )

    # Output dataframe must match row count
    assert isinstance(ohe_df, pd.DataFrame)
    assert len(ohe_df) == len(sample_df)

    # Logs must always contain bin_log
    assert 'bin_log' in logs
    assert isinstance(logs['bin_log'], pd.DataFrame)

    if to_sweep:
        assert 'sweep_log' in logs
        assert isinstance(logs['sweep_log'], pd.DataFrame)
    else:
        assert 'sweep_log' not in logs


def test_encode_data_empty_df():
    empty_df = pd.DataFrame(columns=['id', 'date', 'feature1'])
    ohe_df, logs = encode_data(
        df=empty_df,
        bin_cols='all',
        bin_quantiles=[0, 0.25, 0.5, 0.75, 1.0],
        bin_quantile_labels=None,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        bin_grouping='none',
        bin_dt_units=1,
        to_sweep=True,
        to_drop_no_data=True,
        min_bin_obs=1,
        min_bin_fraction=0.01,
        lag_num_missing=0
    )
    assert ohe_df.empty
    assert 'bin_log' in logs
    assert logs['bin_log'].empty


@pytest.mark.parametrize("bad_bin_cols", [123, None, 3.14])
def test_encode_data_invalid_bin_cols_raises(sample_df, bad_bin_cols):
    with pytest.raises(TypeError):
        encode_data(
            df=sample_df,
            bin_cols=bad_bin_cols,
            bin_quantiles=[0, 0.5, 1.0],
            bin_quantile_labels=None,
            id_cols=['id'],
            date_col='date',
            drop_cols=[],
            bin_grouping='none',
            bin_dt_units=1,
            to_sweep=False,
            to_drop_no_data=False,
            min_bin_obs=1,
            min_bin_fraction=0.01,
            lag_num_missing=0
        )


def test_encode_data_edge_case_single_row():
    df = pd.DataFrame({
        'id': ['X'],
        'date': pd.to_datetime(['2024-01-01']),
        'feature1': [42]
    })
    ohe_df, logs = encode_data(
        df=df,
        bin_cols='all',
        bin_quantiles=[0, 0.5, 1.0],
        bin_quantile_labels=None,
        id_cols=['id'],
        date_col='date',
        drop_cols=[],
        bin_grouping='none',
        bin_dt_units=1,
        to_sweep=False,
        to_drop_no_data=False,
        min_bin_obs=1,
        min_bin_fraction=0.01,
        lag_num_missing=0
    )
    assert len(ohe_df) == 1
    assert 'bin_log' in logs

# Test for engineer_pipeline()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0]
    })


@pytest.fixture
def dummy_cfg():
    # Returns a simple config object with required attributes
    class DummyCfg:
        id_cols = ['id']
        date_col = 'date'
        log_max_rows = 10
        engineer_cols = 'base'
        to_engineer_dates = True
        to_engineer_ratios = True
        to_engineer_lags = True
        lag_mode = 'raw_only'
        n_dt_list = [1]
        flat_threshold = [-0.01, 0.01]
        bin_cols = 'all'
        bin_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        bin_quantile_labels = None
        drop_cols = []
        bin_grouping = 'none'
        bin_dt_units = 1
        to_sweep = True
        to_drop_no_data = True
        min_bin_obs = 1
        min_bin_fraction = 0.01
        lag_num_missing = 1

    return DummyCfg()


def test_engineer_pipeline_typical_usage(sample_df, dummy_cfg):
    df_out, logs = engineer_pipeline(df=sample_df, cfg=dummy_cfg)

    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == len(sample_df)
    assert isinstance(logs, dict)
    assert 'engineer_features_logs' in logs
    assert 'encode_data_logs' in logs


def test_engineer_pipeline_empty_dataframe(dummy_cfg):
    empty_df = pd.DataFrame(columns=['id', 'date', 'feature1'])
    df_out, logs = engineer_pipeline(df=empty_df, cfg=dummy_cfg)

    assert df_out.empty
    assert isinstance(logs, dict)
    assert all(isinstance(log, dict) for log in logs.values())


def test_engineer_pipeline_single_row(sample_df, dummy_cfg):
    single_row_df = sample_df.iloc[[0]]
    df_out, logs = engineer_pipeline(df=single_row_df, cfg=dummy_cfg)

    assert len(df_out) == 1
    assert 'engineer_features_logs' in logs
    assert 'encode_data_logs' in logs


def test_engineer_pipeline_logger_called(sample_df, dummy_cfg):
    logger = MagicMock()

    df_out, logs = engineer_pipeline(df=sample_df, cfg=dummy_cfg, logger=logger)

    assert logger.log_step.call_count == 2  # Should log for engineering + encoding


@pytest.mark.parametrize("override_key,override_value", [
    ('engineer_cols', 'all'),
    ('to_engineer_dates', False),
    ('lag_mode', 'combined_only')
])
def test_engineer_pipeline_parameter_overrides(sample_df, dummy_cfg, override_key, override_value):
    overrides = {override_key: override_value}
    df_out, logs = engineer_pipeline(df=sample_df, cfg=dummy_cfg, **overrides)

    assert isinstance(df_out, pd.DataFrame)
    assert 'engineer_features_logs' in logs
    assert 'encode_data_logs' in logs

# Test for validate_pipeline_input()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'feature_bin1': [1, 0, 1, 0],
        'feature_bin2': [True, False, True, False],
        'feature_ohe=value1': [1, 0, 1, 0],
        'feature_continuous': [0.5, 1.2, 3.4, 4.5]
    })


def test_typical_pipeline_ready(sample_df):
    report = validate_pipeline_input(
        df=sample_df,
        id_cols=['id'],
        date_col='date',
        skip_cols=['feature_continuous']
    )
    assert report['pipeline_ready'] is True
    assert "binary_columns" in report['column_summary']
    assert "report_text" in report


def test_pipeline_not_ready_due_to_missing_id(sample_df):
    report = validate_pipeline_input(
        df=sample_df.drop(columns=['id']),
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is False
    assert any("Missing ID columns" in warning for warning in report['warnings'])


def test_pipeline_not_ready_due_to_non_datetime_date(sample_df):
    broken_df = sample_df.copy()
    broken_df['date'] = ['not_a_date'] * len(broken_df)
    report = validate_pipeline_input(
        df=broken_df,
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is False
    assert any("cannot be parsed as datetime" in warning for warning in report['warnings'])


def test_pipeline_not_ready_due_to_non_binary_features(sample_df):
    report = validate_pipeline_input(
        df=sample_df,
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is False
    assert any("non-binary" in warning for warning in report['warnings'])


def test_empty_dataframe():
    df = pd.DataFrame()
    report = validate_pipeline_input(
        df=df,
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is False
    assert "Missing ID columns" in report['warnings'][0]


def test_skip_cols_behavior(sample_df):
    report = validate_pipeline_input(
        df=sample_df,
        id_cols=['id'],
        date_col='date',
        skip_cols=['feature_continuous']
    )
    assert "feature_continuous" not in report['column_summary']
    assert report['pipeline_ready'] is True


def test_duplicate_rows_detection(sample_df):
    duplicated_df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    report = validate_pipeline_input(
        df=duplicated_df,
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is False
    assert any("duplicate rows detected" in warning for warning in report['warnings'])


@pytest.mark.parametrize("binary_col", [
    [1, 0, 1, 0],
    [True, False, True, False]
])
def test_binary_detection_varied_types(binary_col):
    df = pd.DataFrame({
        'id': ['X', 'Y', 'Z', 'Q'],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'feature': binary_col
    })
    report = validate_pipeline_input(
        df=df,
        id_cols=['id'],
        date_col='date'
    )
    assert report['pipeline_ready'] is True
