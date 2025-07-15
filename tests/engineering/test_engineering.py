import pytest
import pandas as pd
import numpy as np

from edge_research import (
    generate_ratio_features, 
    MAX_REPLACEMENT_DEFAULT,
    generate_temporal_pct_change,
    extract_date_features,
    bin_columns_flexible,
    sweep_low_count_bins,
    one_hot_encode_features
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
