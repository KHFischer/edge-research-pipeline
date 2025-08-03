import pandas as pd
import pytest

from edge_research.calculator import (
    _get_binary_counts,
    calculate_association_metrics,
    apply_statistic_filters,
    generate_statistics,
    extend_stat_registry, 
    get_stat_registry
)

# Test for _get_binary_counts()
@pytest.mark.parametrize("feature_values", [
    ([1, 0, 1, 1]),   # strictly 0/1 integers
    ([True, False, True, True])  # boolean features
])
def test_get_binary_counts_normal_case(feature_values):
    df = pd.DataFrame({
        'feature1': feature_values,
        'feature2': [0, 1, 0, 1],
        'target': ['A', 'B', 'A', 'A']
    })

    counts_df = _get_binary_counts(df, feature_cols=['feature1', 'feature2'], target_col='target')

    # Validate output structure
    assert set(counts_df.columns) == {'feature', 'feature_value', 'target', 'count'}

    # Validate counts are correct (feature1=1 with target='A' should be counted)
    assert counts_df['count'].sum() == 2 * len(df)  # Each feature contributes 4 rows


def test_get_binary_counts_raises_on_non_binary_feature():
    df = pd.DataFrame({
        'feature1': [0, 1, 2, 1],  # Invalid: contains 2
        'target': ['X', 'Y', 'X', 'X']
    })

    with pytest.raises(ValueError, match="Feature 'feature1' must be strictly binary"):
        _get_binary_counts(df, feature_cols=['feature1'], target_col='target')


def test_get_binary_counts_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'target'])

    counts_df = _get_binary_counts(df, feature_cols=['feature1'], target_col='target')

    # Should return empty dataframe with correct columns
    assert isinstance(counts_df, pd.DataFrame)
    assert set(counts_df.columns) == {'feature', 'feature_value', 'target', 'count'}
    assert counts_df.empty


def test_get_binary_counts_single_row():
    df = pd.DataFrame({
        'feature1': [1],
        'target': ['A']
    })

    counts_df = _get_binary_counts(df, feature_cols=['feature1'], target_col='target')

    # Should produce exactly one count row (feature1==1 with target A)
    assert len(counts_df) == 1
    assert counts_df.iloc[0]['count'] == 1
    assert counts_df.iloc[0]['feature'] == 'feature1'
    assert counts_df.iloc[0]['feature_value'] == 1
    assert counts_df.iloc[0]['target'] == 'A'


def test_get_binary_counts_output_counts_correct():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 1],
        'target': ['A', 'A', 'B', 'A']
    })

    counts_df = _get_binary_counts(df, feature_cols=['feature1'], target_col='target')

    # Expected: feature1 == 1 occurs 2 times with A, 1 time with B
    result = counts_df[
        (counts_df['feature'] == 'feature1') &
        (counts_df['feature_value'] == 1) &
        (counts_df['target'] == 'A')
    ]['count'].values[0]

    assert result == 2

# Test for calculate_association_metrics()
def test_calculate_association_metrics_normal_case():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 1],
        'feature2': [0, 1, 0, 0],
        'target': ['A', 'B', 'A', 'A']
    })

    result = calculate_association_metrics(df, target_col='target')

    # Check output columns exist
    assert "antecedents" in result.columns
    assert "consequents" in result.columns
    assert "support" in result.columns  # Registry-driven metric

    # Check output row count is correct (features × target classes)
    assert len(result) >= 1

    # Check that antecedents descriptions are formatted
    assert all(result["antecedents"].str.contains("== 1"))

    # Check consequents values come from original target column
    assert set(result["consequents"].unique()).issubset(set(df['target'].unique()))


def test_calculate_association_metrics_single_row():
    df = pd.DataFrame({
        'feature1': [1],
        'target': ['A']
    })

    result = calculate_association_metrics(df, target_col='target')

    # Should return exactly one row
    assert len(result) == 1
    assert result.iloc[0]["support"] == pytest.approx(1.0)


def test_calculate_association_metrics_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'target'])

    with pytest.raises(KeyError):
        calculate_association_metrics(df, target_col='target')


def test_calculate_association_metrics_invalid_feature_values():
    df = pd.DataFrame({
        'feature1': [0, 1, 2],  # Invalid: contains 2
        'target': ['A', 'B', 'A']
    })

    # Should fail inside _get_binary_counts
    with pytest.raises(ValueError):
        calculate_association_metrics(df, target_col='target')


def test_calculate_association_metrics_output_correctness():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 1],
        'target': ['X', 'Y', 'X', 'Y']
    })

    result = calculate_association_metrics(df, target_col='target')

    # Known: feature1 == 1 occurs 2 times with X, 1 time with Y
    x_support = result[
        (result["antecedents"] == "feature1 == 1") &
        (result["consequents"] == "X")
    ]["support"].values[0]

    assert x_support == pytest.approx(2 / 4)


def test_calculate_association_metrics_target_missing():
    df = pd.DataFrame({
        'feature1': [1, 0, 1],
        'feature2': [1, 1, 0]
    })

    with pytest.raises(KeyError):
        calculate_association_metrics(df, target_col='nonexistent_target')

# Test for apply_statistic_filters()
@pytest.mark.parametrize("filter_key, threshold, expected_selected", [
    ("stat_min_score", 0.5, [True, True, False, False]),
    ("stat_max_score", 0.5, [True, False, False, True]),
    ("stat_upper_score", 0.5, [True, False, False, True]),
    ("stat_lower_score", 0.5, [True, True, False, False]),
    ("stat_bounds_score", [0.2, 0.8], [True, False, False, True]),  # outside 0.2–0.8
    ("stat_range_score", [0.2, 0.8], [False, True, True, False])    # inside 0.2–0.8
])
def test_apply_statistic_filters_conditions(filter_key, threshold, expected_selected):
    df = pd.DataFrame({
        'score': [0.1, 0.4, 0.6, 0.95]
    })

    filter_config = {filter_key: threshold}

    result = apply_statistic_filters(df, filter_config)

    # Verify 'selected' column reflects the filter correctly
    assert list(result['selected']) == expected_selected


def test_apply_statistic_filters_empty_dataframe():
    df = pd.DataFrame(columns=['score'])
    filter_config = {"stat_min_score": 0.5}

    result = apply_statistic_filters(df, filter_config)

    # Should return empty dataframe with 'selected' column
    assert 'selected' in result.columns
    assert result.empty


def test_apply_statistic_filters_metric_not_found():
    df = pd.DataFrame({'score': [0.1, 0.2]})
    filter_config = {"stat_min_missing_metric": 0.5}

    with pytest.raises(ValueError, match="Metric 'missing_metric'"):
        apply_statistic_filters(df, filter_config)


def test_apply_statistic_filters_invalid_key_format():
    df = pd.DataFrame({'score': [0.1, 0.2]})
    filter_config = {"invalid_key_format": 0.5}

    with pytest.raises(ValueError, match="Invalid filter key format"):
        apply_statistic_filters(df, filter_config)


def test_apply_statistic_filters_invalid_condition():
    df = pd.DataFrame({'score': [0.1, 0.2]})
    filter_config = {"stat_unknown_score": 0.5}

    with pytest.raises(ValueError, match="Unsupported filter condition"):
        apply_statistic_filters(df, filter_config)


@pytest.mark.parametrize("invalid_threshold", [
    "not_a_number",
    [1, "bad"],
    [1],
    [1, 2, 3]
])
def test_apply_statistic_filters_invalid_threshold_type(invalid_threshold):
    df = pd.DataFrame({'score': [0.1, 0.2]})
    filter_config = {"stat_bounds_score": invalid_threshold}

    with pytest.raises(ValueError):
        apply_statistic_filters(df, filter_config)

# Test for generate_statistics()
class DummyConfig:
    """Minimal config stub for testing."""
    id_cols = ['id']
    date_col = 'date'
    drop_cols = ['meta']
    target_col = 'target'


def test_generate_statistics_normal_case():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'date': pd.date_range('2024-01-01', periods=4),
        'meta': ['x', 'y', 'x', 'y'],
        'feature1': [1, 0, 1, 1],
        'feature2': [0, 1, 0, 1],
        'target': ['A', 'B', 'A', 'A']
    })

    cfg = DummyConfig()

    result_df, summary_df = generate_statistics(df, cfg)

    # Verify result structure
    assert isinstance(result_df, pd.DataFrame)
    assert "selected" in result_df.columns
    assert "support" in result_df.columns

    # Verify summary structure
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape[0] == 1  # Single-row summary


def test_generate_statistics_target_missing():
    df = pd.DataFrame({
        'id': [1, 2],
        'date': ['2024-01-01', '2024-01-02'],
        'meta': ['x', 'y'],
        'feature1': [1, 0]
    })

    cfg = DummyConfig()

    with pytest.raises(ValueError, match="Target column 'target' missing"):
        generate_statistics(df, cfg)


def test_generate_statistics_empty_dataframe():
    df = pd.DataFrame(columns=['id', 'date', 'meta', 'feature1', 'target'])
    cfg = DummyConfig()

    result_df, summary_df = generate_statistics(df, cfg)

    # Should return empty results but still structured properly
    assert isinstance(result_df, pd.DataFrame)
    assert "selected" in result_df.columns
    assert result_df.empty

    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape[0] == 1


def test_generate_statistics_with_overrides():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'date': pd.date_range('2024-01-01', periods=4),
        'meta': ['x', 'y', 'x', 'y'],
        'feature1': [1, 0, 1, 1],
        'feature2': [0, 1, 0, 1],
        'target': ['A', 'B', 'A', 'A']
    })

    cfg = DummyConfig()

    overrides = {"stat_min_support": 0.5}  # Force stricter filter

    result_df, summary_df = generate_statistics(df, cfg, overrides=overrides)

    # All rules should be unselected due to high threshold
    assert result_df["selected"].sum() == 0

# Test for extend_stat_registry() and get_stat_registry()
def dummy_statistic(m):
    return m["support"] * 2


def failing_statistic(m):
    raise RuntimeError("Intentional failure")


def non_scalar_statistic(m):
    return [m["support"]]


def test_extend_stat_registry_adds_custom_statistic():
    custom_stats = {"double_support": dummy_statistic}
    extended = extend_stat_registry(custom_stats=custom_stats)
    assert "double_support" in extended
    assert callable(extended["double_support"])


def test_extend_stat_registry_excludes_standard_metrics():
    standard_keys = set(get_stat_registry().keys())
    exclude_keys = list(standard_keys)[:2]  # exclude first two metrics
    extended = extend_stat_registry(exclude_stats=exclude_keys)
    for key in exclude_keys:
        assert key not in extended


def test_extend_stat_registry_prevents_overwrite_by_default():
    standard_metric = next(iter(get_stat_registry().keys()))
    custom_stats = {standard_metric: dummy_statistic}
    with pytest.raises(ValueError, match="already exists"):
        extend_stat_registry(custom_stats=custom_stats)


def test_extend_stat_registry_allows_override_if_enabled():
    standard_metric = next(iter(get_stat_registry().keys()))
    custom_stats = {standard_metric: dummy_statistic}
    extended = extend_stat_registry(custom_stats=custom_stats, allow_override=True)
    assert extended[standard_metric] is dummy_statistic


def test_extend_stat_registry_invalid_function_raises_error():
    custom_stats = {"failing_stat": failing_statistic}
    with pytest.raises(ValueError, match="raised an error on test input"):
        extend_stat_registry(custom_stats=custom_stats)


def test_extend_stat_registry_non_scalar_function_raises_error():
    custom_stats = {"non_scalar_stat": non_scalar_statistic}
    with pytest.raises(ValueError, match="must return a numeric scalar"):
        extend_stat_registry(custom_stats=custom_stats)


@pytest.mark.parametrize("exclude_stats", [[], None])
def test_extend_stat_registry_with_no_exclusions(exclude_stats):
    original = get_stat_registry()
    extended = extend_stat_registry(exclude_stats=exclude_stats)
    assert set(extended.keys()) == set(original.keys())


def test_extend_stat_registry_empty_custom_stats_returns_original():
    original = get_stat_registry()
    extended = extend_stat_registry(custom_stats={})
    assert set(extended.keys()) == set(original.keys())


def test_extend_stat_registry_custom_and_exclude_combined():
    custom_stats = {"double_support": dummy_statistic}
    exclude_keys = list(get_stat_registry().keys())[:3]
    extended = extend_stat_registry(custom_stats=custom_stats, exclude_stats=exclude_keys)
    for key in exclude_keys:
        assert key not in extended
    assert "double_support" in extended
