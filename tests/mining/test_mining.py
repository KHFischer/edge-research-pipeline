import numpy as np
import pandas as pd
import pytest

from edge_research.mining import (
    prepare_dataframe_for_mining,
    parse_apriori_rules,
    perform_rulefit,
    parse_rule_string_to_tuples,
    parse_rulefit_rules,
    perform_subgroup_discovery,
    parse_subgroup_rule_to_tuples,
    parse_subgroup_rules,
    normalize_and_dedup_rules,
    normalize_rule,
    deduplicate_rules_with_provenance,
    count_rules_per_algorithm,
    generate_rule_activation_dataframe
)

# Test for prepare_dataframe_for_mining()
@pytest.fixture
def sample_dataframe():
    data = {
        'date': pd.date_range('2025-01-01', periods=10),
        'id': range(10),
        'drop_me': [0]*10,
        'feature1': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'feature2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'target': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_typical_input_processing(sample_dataframe):
    df_out, log_df = prepare_dataframe_for_mining(
        df=sample_dataframe,
        date_col='date',
        id_cols=['id'],
        drop_cols=['drop_me'],
        target_col='target'
    )

    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(log_df, pd.DataFrame)
    assert df_out.shape[1] == 3  # feature1, feature2, target
    assert set(df_out.columns) == {'feature1', 'feature2', 'target'}
    assert df_out.dtypes['feature1'] == 'uint8'
    assert log_df.shape == (1, len(log_df.columns))


def test_target_column_missing_raises(sample_dataframe):
    with pytest.raises(ValueError, match="Target column 'not_present' missing"):
        prepare_dataframe_for_mining(
            df=sample_dataframe,
            date_col='date',
            id_cols=['id'],
            drop_cols=['drop_me'],
            target_col='not_present'
        )


def test_sampling_reduces_rows(sample_dataframe):
    df_large = pd.concat([sample_dataframe] * 2000, ignore_index=True)
    df_out, log_df = prepare_dataframe_for_mining(
        df=df_large,
        date_col='date',
        id_cols=['id'],
        drop_cols=['drop_me'],
        target_col='target',
        to_sample=True,
        sample_size=1000
    )

    assert len(df_out) <= 1000
    assert log_df.iloc[0]['sampling_applied'] is True
    assert log_df.iloc[0]['rows_after_sampling'] <= 1000


def test_no_sampling_when_smaller_than_sample_size(sample_dataframe):
    df_out, log_df = prepare_dataframe_for_mining(
        df=sample_dataframe,
        date_col='date',
        id_cols=['id'],
        drop_cols=['drop_me'],
        target_col='target',
        to_sample=True,
        sample_size=100
    )

    assert len(df_out) == len(sample_dataframe)
    assert log_df.iloc[0]['rows_after_sampling'] == len(sample_dataframe)


def test_drop_duplicates_removes_rows():
    df = pd.DataFrame({
        'date': ['2025-01-01'] * 4,
        'id': [1, 2, 3, 4],
        'drop_me': [0, 0, 0, 0],
        'feature1': [1, 1, 1, 1],
        'target': [1, 1, 1, 1]
    })

    df_out, log_df = prepare_dataframe_for_mining(
        df=df,
        date_col='date',
        id_cols=['id'],
        drop_cols=['drop_me'],
        target_col='target',
        drop_duplicates=True
    )

    assert len(df_out) == 1
    assert log_df.iloc[0]['duplicates_dropped'] == 3


@pytest.mark.parametrize("to_sample,drop_duplicates", [
    (True, False),
    (False, True),
    (True, True),
    (False, False)
])
def test_log_dataframe_columns_present(sample_dataframe, to_sample, drop_duplicates):
    df_out, log_df = prepare_dataframe_for_mining(
        df=sample_dataframe,
        date_col='date',
        id_cols=['id'],
        drop_cols=['drop_me'],
        target_col='target',
        to_sample=to_sample,
        drop_duplicates=drop_duplicates
    )

    expected_columns = [
        'initial_rows', 'initial_columns', 'initial_ram_mb',
        'columns_dropped', 'features_retained', 'duplicates_dropped',
        'rows_after_drop_duplicates', 'sampling_applied',
        'rows_after_sampling', 'final_rows', 'final_ram_mb'
    ]

    assert all(col in log_df.columns for col in expected_columns)
    assert log_df.shape == (1, len(expected_columns))

# Test for parse_apriori_rules()
def test_parse_apriori_rules_typical_case():
    df = pd.DataFrame({'antecedents': [
        frozenset({'featA', 'featB'}),
        frozenset({'featC'}),
    ]})
    result = parse_apriori_rules(df)

    assert isinstance(result, list)
    assert all(isinstance(rule, list) for rule in result)
    assert all(isinstance(cond, tuple) and len(cond) == 2 for rule in result for cond in rule)
    assert all(cond[1] == 1 for rule in result for cond in rule)

    flat_features = [cond[0] for rule in result for cond in rule]
    assert set(flat_features).issuperset({'featA', 'featB', 'featC'})


def test_parse_apriori_rules_empty_dataframe():
    df = pd.DataFrame({'antecedents': []})
    result = parse_apriori_rules(df)
    assert result == []


def test_parse_apriori_rules_missing_column_raises():
    df = pd.DataFrame({'wrong_column': [frozenset({'featA'})]})
    with pytest.raises(ValueError, match="Column 'antecedents' not found"):
        parse_apriori_rules(df)


@pytest.mark.parametrize("bad_value", [
    ['featA', 'featB'],  # list instead of frozenset
    ('featA',),          # tuple instead of frozenset
    123,                 # int
    None,                # NoneType
    frozenset([123]),    # frozenset but non-string feature
])
def test_parse_apriori_rules_invalid_types_raise(bad_value):
    df = pd.DataFrame({'antecedents': [bad_value]})
    with pytest.raises(ValueError, match="Expected frozenset"):
        parse_apriori_rules(df)


def test_parse_apriori_rules_single_feature_rule():
    df = pd.DataFrame({'antecedents': [frozenset({'only_feature'})]})
    result = parse_apriori_rules(df)
    assert result == [[('only_feature', 1)]]

# Test for perform_rulefit()
@pytest.fixture
def simple_binary_dataset():
    """Creates a minimal valid dataset with binary features and binary target."""
    df = pd.DataFrame({
        'featA': [0, 1, 0, 1],
        'featB': [1, 0, 1, 0],
        'target': [1, 0, 1, 0]
    })
    return df


def test_perform_rulefit_basic_output(simple_binary_dataset):
    rules_df, summary_df = perform_rulefit(
        df=simple_binary_dataset,
        target_col='target',
        tree_size=2,
        min_rule_depth=2
    )

    assert isinstance(rules_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert 'consequents' in rules_df.columns
    assert 'rule' in rules_df.columns
    assert 'support' in rules_df.columns
    assert 'depth' in rules_df.columns

    assert summary_df.shape[0] == len(simple_binary_dataset['target'].unique())
    assert 'target_class' in summary_df.columns
    assert 'total_extracted_rules' in summary_df.columns


def test_perform_rulefit_missing_target_raises(simple_binary_dataset):
    with pytest.raises(ValueError, match="Target column 'not_found' not found"):
        perform_rulefit(
            df=simple_binary_dataset,
            target_col='not_found'
        )


def test_perform_rulefit_nan_in_features_raises(simple_binary_dataset):
    df_with_nan = simple_binary_dataset.copy()
    df_with_nan.loc[0, 'featA'] = np.nan

    with pytest.raises(ValueError, match="Feature matrix contains missing values"):
        perform_rulefit(
            df=df_with_nan,
            target_col='target'
        )


@pytest.mark.parametrize("tree_size,min_rule_depth", [
    (1, 1),
    (5, 2),
    (3, 3),
])
def test_perform_rulefit_parameter_variations(simple_binary_dataset, tree_size, min_rule_depth):
    rules_df, summary_df = perform_rulefit(
        df=simple_binary_dataset,
        target_col='target',
        tree_size=tree_size,
        min_rule_depth=min_rule_depth
    )

    assert isinstance(rules_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)


def test_perform_rulefit_empty_dataframe_returns_empty_rules():
    df_empty = pd.DataFrame(columns=['featA', 'featB', 'target'])
    with pytest.raises(ValueError):
        perform_rulefit(df_empty, target_col='target')

# Test for parse_rule_string_to_tuples() and parse_rulefit_rules()
@pytest.mark.parametrize("rule_str,expected", [
    ("feature1 <= 0.5", [("feature1", 0)]),
    ("feature2 > 0.5", [("feature2", 1)]),
    ("featureA <= 0.5 and featureB > 0.5", [("featureA", 0), ("featureB", 1)]),
    ("feat1 > 0.5 and feat2 <= 0.5", [("feat1", 1), ("feat2", 0)]),
])
def test_parse_rule_string_to_tuples_valid_cases(rule_str, expected):
    result = parse_rule_string_to_tuples(rule_str)
    assert result == expected


@pytest.mark.parametrize("rule_str", [
    "featureX <= 1.0",       # Unsupported value
    "featureY > 1.0",        # Unsupported value
    "featureZ != 0.5",       # Unsupported operator
    "feature1 <= 0.5 or feature2 > 0.5",  # Unsupported split by 'or'
    "invalidrule",           # No operator
])
def test_parse_rule_string_to_tuples_invalid_cases(rule_str):
    with pytest.raises(ValueError):
        parse_rule_string_to_tuples(rule_str)


def test_parse_rulefit_rules_typical_case():
    df = pd.DataFrame({'rule': [
        "feature1 <= 0.5 and feature2 > 0.5",
        "feature3 > 0.5"
    ]})
    parsed = parse_rulefit_rules(df)
    expected = [
        [("feature1", 0), ("feature2", 1)],
        [("feature3", 1)]
    ]
    assert parsed == expected


def test_parse_rulefit_rules_missing_column_raises():
    df = pd.DataFrame({'wrong_column': ["feature1 <= 0.5"]})
    with pytest.raises(ValueError, match="Column 'rule' not found"):
        parse_rulefit_rules(df)


def test_parse_rulefit_rules_invalid_row_type_raises():
    df = pd.DataFrame({'rule': [None]})
    with pytest.raises(ValueError, match="Row 0: Expected rule string"):
        parse_rulefit_rules(df)


def test_parse_rulefit_rules_empty_dataframe_returns_empty():
    df = pd.DataFrame({'rule': []})
    result = parse_rulefit_rules(df)
    assert result == []

# Test for perform_subgroup_discovery()
@pytest.fixture
def simple_multiclass_dataset():
    return pd.DataFrame({
        'featA': [0, 1, 0, 1, 0, 1],
        'featB': [1, 0, 1, 1, 0, 0],
        'target': ['A', 'B', 'A', 'B', 'A', 'B']
    })


def test_perform_subgroup_discovery_basic_output(simple_multiclass_dataset):
    rules_df, summary_df = perform_subgroup_discovery(
        df=simple_multiclass_dataset,
        target_col='target',
        top_n=10,
        depth=3,
        beam_width=5
    )

    assert isinstance(rules_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)

    # rules_df must contain required columns
    assert 'rule' in rules_df.columns
    assert 'consequents' in rules_df.columns
    assert 'quality' in rules_df.columns
    assert 'depth' in rules_df.columns

    # summary_df must contain expected statistics
    expected_summary_columns = [
        'target_class', 'empty_rule_set', 'total_raw_rules',
        'rules_retained_multivar', 'rules_filtered_out',
        'avg_rule_depth', 'quality_min', 'quality_max', 'quality_mean'
    ]
    for col in expected_summary_columns:
        assert col in summary_df.columns


def test_perform_subgroup_discovery_missing_target_raises(simple_multiclass_dataset):
    with pytest.raises(ValueError, match="Target column 'not_found' not found"):
        perform_subgroup_discovery(
            df=simple_multiclass_dataset,
            target_col='not_found'
        )


def test_perform_subgroup_discovery_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=['featA', 'featB', 'target'])
    with pytest.raises(ValueError):
        perform_subgroup_discovery(
            df=empty_df,
            target_col='target'
        )


@pytest.mark.parametrize("beam_width,top_n", [
    (2, 5),
    (10, 5),
    (5, 10)
])
def test_perform_subgroup_discovery_param_variations(simple_multiclass_dataset, beam_width, top_n):
    rules_df, summary_df = perform_subgroup_discovery(
        df=simple_multiclass_dataset,
        target_col='target',
        top_n=top_n,
        depth=2,
        beam_width=beam_width
    )

    assert isinstance(rules_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)


def test_perform_subgroup_discovery_empty_class_handling():
    df = pd.DataFrame({
        'featA': [0, 0, 0],
        'featB': [1, 1, 1],
        'target': ['A', 'A', 'A']  # single class
    })
    rules_df, summary_df = perform_subgroup_discovery(df, target_col='target')

    # Expect single-class summary
    assert summary_df.shape[0] == 1
    assert 'A' in summary_df['target_class'].values

    # Expect rules_df to have only rules predicting class A
    assert all(rules_df['consequents'] == 'A')

# Test for parse_subgroup_rule_to_tuples() and parse_subgroup_rules()
@pytest.mark.parametrize("rule_str,expected", [
    ("featureA == True", [("featureA", 1)]),
    ("featureB == False", [("featureB", 0)]),
    ("featureA == True AND featureB == False", [("featureA", 1), ("featureB", 0)]),
    ("feature1 == False AND feature2 == True AND feature3 == True",
     [("feature1", 0), ("feature2", 1), ("feature3", 1)]),
    ("(featureX) == True", [("featureX", 1)]),
    ("", []),
    (None, []),  # Treat None as empty string
])
def test_parse_subgroup_rule_to_tuples_valid_cases(rule_str, expected):
    result = parse_subgroup_rule_to_tuples(rule_str)
    assert result == expected


@pytest.mark.parametrize("rule_str", [
    "featureA != True",                   # Unsupported operator
    "featureB == maybe",                  # Unsupported value
    "featureC >= True",                   # Unsupported operator
    "feature1 AND feature2 == True",      # Missing equality in first part
    "badly formatted rule",               # No equality operator
])
def test_parse_subgroup_rule_to_tuples_invalid_cases(rule_str):
    with pytest.raises(ValueError):
        parse_subgroup_rule_to_tuples(rule_str)


def test_parse_subgroup_rule_to_tuples_ignores_target_conditions():
    rule = "target_foo == True AND featureA == False AND target_bar == False"
    result = parse_subgroup_rule_to_tuples(rule, target_prefix="target_")
    assert result == [("featureA", 0)]  # Only featureA should be parsed


def test_parse_subgroup_rules_typical_dataframe():
    df = pd.DataFrame({'rule': [
        "featureA == True AND featureB == False",
        "featureC == True"
    ]})
    parsed = parse_subgroup_rules(df)
    expected = [
        [("featureA", 1), ("featureB", 0)],
        [("featureC", 1)]
    ]
    assert parsed == expected


def test_parse_subgroup_rules_missing_column_raises():
    df = pd.DataFrame({'wrong_column': ["featureA == True"]})
    with pytest.raises(ValueError, match="Column 'rule' not found"):
        parse_subgroup_rules(df)


def test_parse_subgroup_rules_invalid_rule_raises():
    df = pd.DataFrame({'rule': ["featureA != True"]})
    with pytest.raises(ValueError, match="Error parsing rule at row 0"):
        parse_subgroup_rules(df)


def test_parse_subgroup_rules_empty_dataframe_returns_empty():
    df = pd.DataFrame({'rule': []})
    result = parse_subgroup_rules(df)
    assert result == []

# Test for parse_subgroup_rules(), normalize_rule(), deduplicate_rules_with_provenance(), count_rules_per_algorithm()
def test_normalize_and_dedup_rules_basic_case():
    # Simulate overlapping rules across miners
    rule_sources = [
        ('apriori', [
            [('featA', 1), ('featB', 0)],
            [('featC', 1), ('featD', 0)],
        ]),
        ('rulefit', [
            [('featB', 0), ('featA', 1)],  # Same as first apriori rule, but unordered
            [('featE', 1)],
        ]),
        ('subgroup', [
            [('featE', 1)],  # Same as one from rulefit
        ]),
    ]

    dedup_rules, rule_count_df = normalize_and_dedup_rules(rule_sources)

    # Check structure
    assert isinstance(dedup_rules, list)
    assert all(isinstance(rule, tuple) and isinstance(rule[0], list) and isinstance(rule[1], set)
               for rule in dedup_rules)

    # Check output counts
    assert isinstance(rule_count_df, pd.DataFrame)
    assert set(rule_count_df.columns) == {'algorithm', 'unique_rule_count'}

    # Validate counts: apriori should contribute 2 unique rules, rulefit 2, subgroup 1
    algo_counts = dict(zip(rule_count_df['algorithm'], rule_count_df['unique_rule_count']))
    assert algo_counts['apriori'] == 2
    assert algo_counts['rulefit'] == 2
    assert algo_counts['subgroup'] == 1

    # Validate deduplication: only 3 unique rules should remain
    assert len(dedup_rules) == 3

    # Validate provenance: algorithms contributing to each unique rule
    provenance_sets = [provenance for _, provenance in dedup_rules]
    assert any({'apriori', 'rulefit'} == provenance for provenance in provenance_sets)
    assert any({'rulefit', 'subgroup'} == provenance for provenance in provenance_sets)
    assert any({'apriori'} == provenance for provenance in provenance_sets)


def test_normalize_and_dedup_rules_empty_input():
    dedup_rules, rule_count_df = normalize_and_dedup_rules([])

    assert dedup_rules == []
    assert isinstance(rule_count_df, pd.DataFrame)
    assert rule_count_df.empty


def test_normalize_rule_sorting():
    rule = [('featB', 0), ('featA', 1), ('featC', 1)]
    normalized = normalize_rule(rule)
    assert normalized == [('featA', 1), ('featB', 0), ('featC', 1)]


def test_deduplicate_rules_with_provenance_merges_equivalent_rules():
    rule_sources = [
        ('algo1', [[('A', 1), ('B', 0)]]),
        ('algo2', [[('B', 0), ('A', 1)]]),
    ]
    normalized_sources = [
        (algo, [normalize_rule(rule) for rule in rules])
        for algo, rules in rule_sources
    ]
    deduped = deduplicate_rules_with_provenance(normalized_sources)
    assert len(deduped) == 1
    assert deduped[0][1] == {'algo1', 'algo2'}


def test_count_rules_per_algorithm_correct_counts():
    dedup_rules = [
        ([('featA', 1)], {'apriori'}),
        ([('featB', 1)], {'apriori', 'rulefit'}),
        ([('featC', 1)], {'rulefit'}),
    ]
    df = count_rules_per_algorithm(dedup_rules)
    counts = dict(zip(df['algorithm'], df['unique_rule_count']))

    assert counts['apriori'] == 2  # featA and featB
    assert counts['rulefit'] == 2  # featB and featC

# Test for generate_rule_activation_dataframe()
def test_generate_rule_activation_dataframe_typical_case():
    df = pd.DataFrame({
        'featureA': [1, 0, 1, 1],
        'featureB': [0, 0, 1, 0],
        'featureC': [1, 1, 1, 0],
        'target': ['up', 'down', 'up', 'down']
    })

    unique_rules = [
        ([('featureA', 1), ('featureB', 0)], {'rulefit'}),
        ([('featureC', 1)], {'apriori'})
    ]

    rule_df, mapping_df = generate_rule_activation_dataframe(df, unique_rules, target_col='target')

    # Check dataframe shapes and columns
    assert isinstance(rule_df, pd.DataFrame)
    assert isinstance(mapping_df, pd.DataFrame)
    assert rule_df.shape[0] == df.shape[0]
    assert 'rule_0000' in rule_df.columns
    assert 'rule_0001' in rule_df.columns
    assert 'target' in rule_df.columns

    assert list(mapping_df.columns) == ['rule_column', 'human_readable_rule']
    assert len(mapping_df) == 2

    # Check rule activations
    # Rule 0: featureA == 1 AND featureB == 0
    expected_rule_0 = ((df['featureA'] == 1) & (df['featureB'] == 0)).tolist()
    assert rule_df['rule_0000'].tolist() == expected_rule_0

    # Rule 1: featureC == 1
    expected_rule_1 = (df['featureC'] == 1).tolist()
    assert rule_df['rule_0001'].tolist() == expected_rule_1


def test_generate_rule_activation_dataframe_missing_feature_raises():
    df = pd.DataFrame({
        'featureA': [1, 0, 1],
        'target': ['yes', 'no', 'yes']
    })
    unique_rules = [
        ([('featureB', 1)], {'rulefit'})  # featureB missing
    ]
    with pytest.raises(KeyError, match="Feature 'featureB' not found"):
        generate_rule_activation_dataframe(df, unique_rules, target_col='target')


def test_generate_rule_activation_dataframe_missing_target_raises():
    df = pd.DataFrame({
        'featureA': [1, 1, 0]
    })
    unique_rules = [
        ([('featureA', 1)], {'apriori'})
    ]
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        generate_rule_activation_dataframe(df, unique_rules, target_col='target')


def test_generate_rule_activation_dataframe_empty_rules_returns_no_columns():
    df = pd.DataFrame({
        'featureA': [1, 0, 1],
        'target': [1, 0, 1]
    })
    unique_rules = []  # No rules

    rule_df, mapping_df = generate_rule_activation_dataframe(df, unique_rules, target_col='target')

    # Should only return target column
    assert list(rule_df.columns) == ['target']
    assert mapping_df.empty


def test_generate_rule_activation_dataframe_rule_column_naming_consistency():
    df = pd.DataFrame({
        'featureX': [1, 1, 0],
        'target': ['A', 'B', 'A']
    })
    unique_rules = [
        ([('featureX', 1)], {'apriori'})
    ]

    rule_df, mapping_df = generate_rule_activation_dataframe(df, unique_rules, target_col='target', prefix='custom_rule')

    assert 'custom_rule_0000' in rule_df.columns
    assert mapping_df['rule_column'].iloc[0] == 'custom_rule_0000'
