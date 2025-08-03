import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from typing import Any

from skeLCS import eLCS
import Orange

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
    generate_rule_activation_dataframe,
    merge_multivar_map_into_stats, 
    compute_rule_depth,
    perform_elcs,
    df_to_orange_table,
    perform_cn2,
    perform_cart,
    generate_synthetic_data_sdv,
    generate_synthetic_data_synthcity,
    apply_class_imbalance, 
    generate_skewed_proportions, 
    flip_boolean_values, 
    flip_labels,
    generate_combined_synthetic_data,
    augment_dataset,
    mine_stats,
    coalesce_data,
    data_prep_pipeline,
    mining_pipeline
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

# Test for merge_multivar_map_into_stats() and compute_rule_depth()
def test_merge_multivar_map_into_stats_basic_case():
    stats_df = pd.DataFrame({
        'antecedents': ['rule_0001 == 1', 'rule_0002 == 1'],
        'support': [0.15, 0.23]
    })

    map_df = pd.DataFrame({
        'rule_column': ['rule_0001', 'rule_0002'],
        'human_readable_rule': [
            "('featureA' == 1)",
            "('featureB' == 0) AND ('featureC' == 1)"
        ]
    })

    merged_df = merge_multivar_map_into_stats(stats_df, map_df)

    assert 'rule_column' in merged_df.columns
    assert 'human_readable_rule' in merged_df.columns
    assert merged_df.shape[0] == stats_df.shape[0]

    # Check suffix removal and correct merge
    assert merged_df['rule_column'].iloc[0] == 'rule_0001'
    assert merged_df['human_readable_rule'].iloc[1] == "('featureB' == 0) AND ('featureC' == 1)"


def test_merge_multivar_map_missing_antecedents_col_raises():
    stats_df = pd.DataFrame({'wrong_col': ['rule_0001 == 1']})
    map_df = pd.DataFrame({'rule_column': ['rule_0001'], 'human_readable_rule': ["('featureA' == 1)"]})

    with pytest.raises(ValueError, match="Column 'antecedents' not found"):
        merge_multivar_map_into_stats(stats_df, map_df)


def test_merge_multivar_map_missing_required_columns_raises():
    stats_df = pd.DataFrame({'antecedents': ['rule_0001 == 1']})
    map_df = pd.DataFrame({'wrong_col': ['rule_0001']})

    with pytest.raises(ValueError, match="must contain 'rule_column' and 'human_readable_rule'"):
        merge_multivar_map_into_stats(stats_df, map_df)


@pytest.mark.parametrize("rule_str,expected_depth", [
    ("('featureA' == 1)", 1),
    ("('A' == 1) AND ('B' == 1)", 2),
    ("('A' == 0) AND ('B' == 1) AND ('C' == 1)", 3),
    ("", 0),
    (None, 0),
    ("   ", 0),
])
def test_compute_rule_depth_various_cases(rule_str, expected_depth):
    assert compute_rule_depth(rule_str) == expected_depth

# Test for perform_elcs()
def test_perform_elcs_typical_case():
    # Create a small binary classification dataset
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0],
        'feature2': [0, 1, 1, 0],
        'target': ['up', 'down', 'up', 'down']
    })

    rules, log_df = perform_elcs(df, 'target')

    # Basic checks
    assert isinstance(rules, list)
    assert all(isinstance(rule, list) for rule in rules)
    assert all(isinstance(cond, tuple) for rule in rules for cond in rule)
    assert isinstance(log_df, pd.DataFrame)
    assert set(log_df.columns) == {
        'target_class', 'n_rules', 'avg_depth', 'avg_fitness', 'avg_accuracy'
    }

    # Each rule is non-empty
    assert all(len(rule) > 0 for rule in rules)

    # Log entries correspond to target classes
    assert sorted(log_df['target_class'].tolist()) == sorted(['up', 'down'])

    # n_rules is non-negative
    assert (log_df['n_rules'] >= 0).all()


def test_perform_elcs_single_class():
    # Single-class target (edge case)
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0],
        'feature2': [0, 1, 1, 0],
        'target': ['up', 'up', 'up', 'up']
    })

    rules, log_df = perform_elcs(df, 'target')

    # Should produce rules but only for one class
    assert isinstance(rules, list)
    assert len(log_df) == 1
    assert log_df.iloc[0]['target_class'] == 'up'


def test_perform_elcs_raises_on_missing_target():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0],
        'feature2': [0, 1, 1, 0]
    })

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        perform_elcs(df, 'target')


@pytest.mark.parametrize("n_samples, n_features", [
    (10, 5),
    (100, 20),
    (2, 1)
])
def test_perform_elcs_varied_dataset_shapes(n_samples, n_features):
    # Random binary features and random binary target
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2, size=(n_samples, n_features))
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = rng.choice(['A', 'B'], size=n_samples)

    rules, log_df = perform_elcs(df, 'target')

    # Always expect a list of rules and dataframe
    assert isinstance(rules, list)
    assert isinstance(log_df, pd.DataFrame)
    assert not log_df.empty


def test_perform_elcs_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'feature2', 'target'])
    with pytest.raises(ValueError):
        perform_elcs(df, 'target')

# Test for df_to_orange_table()
def test_df_to_orange_table_typical_case():
    df = pd.DataFrame({
        'feature1': [1, 0, 1],
        'feature2': [0, 1, 1],
        'target': ['A', 'B', 'A']
    })

    table = df_to_orange_table(df, target_col='target')

    assert isinstance(table, Orange.data.Table)
    assert table.domain.class_var.name == 'target'
    assert table.X.shape == (3, 2)
    assert table.Y.shape == (3,)
    assert list(table.domain.attributes[0].values) == ['0', '1']


def test_df_to_orange_table_one_sample():
    df = pd.DataFrame({
        'feature1': [1],
        'feature2': [0],
        'target': ['A']
    })

    table = df_to_orange_table(df, target_col='target')

    assert table.X.shape == (1, 2)
    assert table.Y.shape == (1,)


def test_df_to_orange_table_raises_on_missing_target():
    df = pd.DataFrame({
        'feature1': [1, 0],
        'feature2': [0, 1]
    })

    with pytest.raises(KeyError, match="Target column 'target' not found"):
        df_to_orange_table(df, target_col='target')


@pytest.mark.parametrize("bad_values", [
    [2, 0, 1],         # Value outside binary
    [True, False, 2],  # Mixed binary and integer
    ['yes', 'no', 1],  # Non-numeric
])
def test_df_to_orange_table_raises_on_nonbinary_features(bad_values):
    df = pd.DataFrame({
        'feature1': bad_values,
        'feature2': [0, 1, 1],
        'target': ['A', 'B', 'A']
    })

    with pytest.raises(ValueError, match="Non-binary feature columns detected"):
        df_to_orange_table(df, target_col='target')


def test_df_to_orange_table_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'feature2', 'target'])
    with pytest.raises(KeyError):
        df_to_orange_table(df, target_col='target')

# Test for perform_cn2()
def test_perform_cn2_typical_binary_classification():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0],
        'feature2': [0, 1, 1, 0],
        'target': ['A', 'B', 'A', 'B']
    })

    rules, log_df = perform_cn2(df, 'target')

    assert isinstance(rules, list)
    assert all(isinstance(rule, list) for rule in rules)
    assert all(isinstance(cond, tuple) for rule in rules for cond in rule)
    assert isinstance(log_df, pd.DataFrame)
    assert {'n_rules', 'avg_depth'}.issubset(log_df.columns)

    # Basic value checks
    assert log_df.iloc[0]['n_rules'] >= 0
    assert log_df.iloc[0]['avg_depth'] >= 0


def test_perform_cn2_single_class():
    df = pd.DataFrame({
        'feature1': [1, 1, 1, 1],
        'feature2': [0, 0, 0, 0],
        'target': ['X', 'X', 'X', 'X']
    })

    rules, log_df = perform_cn2(df, 'target')

    # Should handle this gracefully—possibly few or no rules
    assert isinstance(rules, list)
    assert isinstance(log_df, pd.DataFrame)


def test_perform_cn2_raises_on_missing_target():
    df = pd.DataFrame({
        'feature1': [1, 0, 1],
        'feature2': [0, 1, 0]
    })

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        perform_cn2(df, 'target')


@pytest.mark.parametrize("n_samples, n_features", [
    (5, 2),
    (50, 10),
    (1, 1)
])
def test_perform_cn2_varied_dataset_shapes(n_samples, n_features):
    rng = np.random.default_rng(123)
    data = rng.integers(0, 2, size=(n_samples, n_features))
    df = pd.DataFrame(data, columns=[f'f{i}' for i in range(n_features)])
    df['target'] = rng.choice(['A', 'B'], size=n_samples)

    rules, log_df = perform_cn2(df, 'target')

    assert isinstance(rules, list)
    assert isinstance(log_df, pd.DataFrame)
    assert not log_df.empty


def test_perform_cn2_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'feature2', 'target'])
    with pytest.raises(ValueError):
        perform_cn2(df, 'target')

# Test for perform_cart()
def test_perform_cart_typical_multiclass():
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0],
        'feature2': [0, 1, 1, 0],
        'target': ['A', 'B', 'A', 'C']
    })

    rules, log_df = perform_cart(df, target_col='target')

    # Validate outputs
    assert isinstance(rules, list)
    assert all(isinstance(rule, list) for rule in rules)
    assert all(isinstance(cond, tuple) and isinstance(cond[0], str) and isinstance(cond[1], int)
               for rule in rules for cond in rule)
    assert isinstance(log_df, pd.DataFrame)
    assert {'n_rules', 'avg_depth', 'tree_depth'}.issubset(log_df.columns)
    assert log_df.iloc[0]['n_rules'] >= 1
    assert log_df.iloc[0]['tree_depth'] >= 1


def test_perform_cart_single_sample():
    df = pd.DataFrame({
        'feature1': [1],
        'feature2': [0],
        'target': ['X']
    })

    rules, log_df = perform_cart(df, target_col='target')

    assert isinstance(rules, list)
    assert isinstance(log_df, pd.DataFrame)
    assert log_df.iloc[0]['tree_depth'] == 0  # Single sample yields depth 0


def test_perform_cart_missing_target():
    df = pd.DataFrame({
        'feature1': [1, 0],
        'feature2': [0, 1]
    })

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        perform_cart(df, target_col='target')


@pytest.mark.parametrize("max_depth", [1, 3, 10])
def test_perform_cart_respects_max_depth(max_depth):
    df = pd.DataFrame({
        'feature1': [1, 0, 1, 0, 1],
        'feature2': [0, 1, 1, 0, 0],
        'target': ['A', 'B', 'A', 'B', 'A']
    })

    _, log_df = perform_cart(df, target_col='target', max_depth=max_depth)
    assert log_df.iloc[0]['tree_depth'] <= max_depth


def test_perform_cart_empty_dataframe():
    df = pd.DataFrame(columns=['feature1', 'feature2', 'target'])

    with pytest.raises(ValueError):
        perform_cart(df, target_col='target')

# Test for generate_synthetic_data_sdv()
@pytest.fixture
def minimal_dataframe():
    return pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C'],
        'numeric_col': [1.0, 2.5, 3.1, 4.2],
    })


@pytest.mark.parametrize("model", ["gaussian_copula", "ctgan", "tvae"])
def test_generate_synthetic_basic(minimal_dataframe, model):
    synthetic_data, metadata_df = generate_synthetic_data_sdv(
        df=minimal_dataframe,
        num_rows=10,
        model=model,
        verbose=False
    )

    # Synthetic output shape check
    assert isinstance(synthetic_data, pd.DataFrame)
    assert len(synthetic_data) == 10
    assert set(synthetic_data.columns) == set(minimal_dataframe.columns)

    # Metadata output structure
    assert isinstance(metadata_df, pd.DataFrame)
    assert 'column_name' in metadata_df.columns
    assert 'sdtype' in metadata_df.columns
    assert len(metadata_df) == minimal_dataframe.shape[1]


def test_generate_synthetic_invalid_model(minimal_dataframe):
    with pytest.raises(ValueError, match="Unsupported model"):
        generate_synthetic_data_sdv(
            df=minimal_dataframe,
            num_rows=5,
            model="not_a_model"
        )


def test_generate_synthetic_missing_values():
    df_with_nan = pd.DataFrame({
        'col1': [1.0, np.nan, 3.0],
        'col2': ['A', 'B', 'C']
    })

    with pytest.raises(ValueError, match="contains missing values"):
        generate_synthetic_data_sdv(
            df=df_with_nan,
            num_rows=5
        )


@pytest.mark.parametrize("row_count", [1, 100])
def test_generate_synthetic_row_boundaries(minimal_dataframe, row_count):
    synthetic_data, _ = generate_synthetic_data_sdv(
        df=minimal_dataframe,
        num_rows=row_count
    )
    assert len(synthetic_data) == row_count


def test_generate_synthetic_verbose_print(capfd, minimal_dataframe):
    generate_synthetic_data_sdv(
        df=minimal_dataframe,
        num_rows=5,
        model='gaussian_copula',
        verbose=True
    )
    out, _ = capfd.readouterr()
    assert "[SDV] Synthetic data quality score:" in out

# Test for generate_synthetic_data_synthcity()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_num": [1.0, 2.0, 3.0, 4.0],
        "feature_cat": ["A", "B", "A", "C"],
        "target": ["yes", "no", "yes", "no"]
    })


@pytest.mark.parametrize("model", ["ctgan", "tvae"])
def test_generate_synthetic_valid_model(sample_df, model):
    synthetic_df, metadata_df = generate_synthetic_data_synthcity(
        df=sample_df.drop(columns=["target"]),
        target_col=None,
        n_rows=5,
        model=model,
        n_iter=10,
        batch_size=2
    )

    assert isinstance(synthetic_df, pd.DataFrame)
    assert isinstance(metadata_df, pd.DataFrame)
    assert len(synthetic_df) == 5
    assert set(synthetic_df.columns) == set(sample_df.drop(columns=["target"]).columns)
    assert "column_name" in metadata_df.columns
    assert "sdtype" in metadata_df.columns


def test_generate_synthetic_with_target(sample_df):
    synthetic_df, metadata_df = generate_synthetic_data_synthcity(
        df=sample_df,
        target_col="target",
        n_rows=5,
        model="ctgan",
        n_iter=10,
        batch_size=2
    )

    assert "target" in synthetic_df.columns
    assert len(synthetic_df) == 5
    assert set(metadata_df.columns).issuperset({"column_name", "sdtype"})


def test_invalid_model_name_raises(sample_df):
    with pytest.raises(ValueError, match="is not a supported Synthcity plugin"):
        generate_synthetic_data_synthcity(
            df=sample_df,
            target_col="target",
            n_rows=5,
            model="invalid_model"
        )


def test_empty_dataframe_raises():
    with pytest.raises(Exception):
        generate_synthetic_data_synthcity(
            df=pd.DataFrame(),
            target_col=None,
            n_rows=5,
            model="ctgan"
        )


def test_single_row_input():
    df = pd.DataFrame({
        "feature_num": [1.0],
        "feature_cat": ["A"],
        "target": ["yes"]
    })

    synthetic_df, _ = generate_synthetic_data_synthcity(
        df=df,
        target_col="target",
        n_rows=2,
        model="ctgan",
        n_iter=5,
        batch_size=1
    )

    assert isinstance(synthetic_df, pd.DataFrame)
    assert len(synthetic_df) == 2

# Test for apply_class_imbalance() and generate_skewed_proportions()
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": ["A", "B", "C", "A", "B", "C"],
        "label": ["x", "y", "x", "y", "x", "z"]
    })


def test_apply_class_imbalance_typical(sample_df):
    result_df = apply_class_imbalance(sample_df, target_col="label", random_state=123)
    assert isinstance(result_df, pd.DataFrame)
    assert set(result_df.columns) == set(sample_df.columns)
    assert "label" in result_df.columns
    assert len(result_df) > 0


def test_apply_class_imbalance_with_custom_proportions(sample_df):
    proportions = {"x": 0.6, "y": 0.3, "z": 0.1}
    result_df = apply_class_imbalance(sample_df, target_col="label", proportions=proportions)
    counts = Counter(result_df["label"])
    total = sum(counts.values())
    dist = {cls: count / total for cls, count in counts.items()}

    # Check that the sampled distribution roughly matches the target proportions
    for cls in proportions:
        assert abs(dist[cls] - proportions[cls]) < 0.15  # Loose tolerance due to sampling noise


def test_apply_class_imbalance_missing_target_column(sample_df):
    with pytest.raises(ValueError, match="Target column 'missing_label' not found"):
        apply_class_imbalance(sample_df, target_col="missing_label")


def test_generate_skewed_proportions_distribution():
    y = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 1)
    skewed = generate_skewed_proportions(y, power=1.5)
    assert isinstance(skewed, dict)
    assert set(skewed.keys()) == {"A", "B", "C"}
    assert abs(sum(skewed.values()) - 1.0) < 1e-6
    # Check that smaller classes got higher weights
    assert skewed["C"] > skewed["B"] > skewed["A"]


def test_generate_skewed_proportions_uniform_input():
    y = pd.Series(["same"] * 10)
    result = generate_skewed_proportions(y)
    assert isinstance(result, dict)
    assert list(result.keys()) == ["same"]
    assert result["same"] == pytest.approx(1.0)

# Test for flip_boolean_values() and flip_labels()
@pytest.fixture
def bool_df():
    return pd.DataFrame({
        "a": [True, False, True, False, True],
        "b": [False, False, True, True, True],
        "c": [1, 2, 3, 4, 5],  # non-boolean column
    })


@pytest.fixture
def label_df():
    return pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "label": ["up", "down", "flat", "up", "down"]
    })


# ─────────────────────────────────────────────────────────────
# Tests for flip_boolean_values
# ─────────────────────────────────────────────────────────────

def test_flip_boolean_values_typical(bool_df):
    flipped = flip_boolean_values(bool_df, flip_fraction=0.5, seed=123)
    assert isinstance(flipped, pd.DataFrame)
    assert set(flipped.columns) == set(bool_df.columns)
    # Only boolean values should be flipped
    assert all(flipped["c"] == bool_df["c"])


def test_flip_boolean_values_subset_column(bool_df):
    flipped = flip_boolean_values(bool_df, columns=["a"], flip_fraction=1.0, seed=0)
    assert (flipped["a"] != bool_df["a"]).all()  # All should flip
    assert (flipped["b"] == bool_df["b"]).all()  # Unchanged


def test_flip_boolean_values_invalid_column_type(bool_df):
    with pytest.raises(TypeError):
        flip_boolean_values(bool_df, columns=["c"], flip_fraction=0.5)


def test_flip_boolean_values_invalid_fraction(bool_df):
    with pytest.raises(ValueError):
        flip_boolean_values(bool_df, flip_fraction=-0.1)


# ─────────────────────────────────────────────────────────────
# Tests for flip_labels
# ─────────────────────────────────────────────────────────────

def test_flip_labels_typical(label_df):
    flipped = flip_labels(label_df, target_col="label", flip_fraction=0.4, seed=999)
    assert isinstance(flipped, pd.DataFrame)
    assert set(flipped.columns) == set(label_df.columns)
    # Ensure some rows are changed but not all
    changed = (flipped["label"] != label_df["label"]).sum()
    assert 0 < changed < len(label_df)


def test_flip_labels_missing_target(label_df):
    with pytest.raises(ValueError, match="Target column 'missing' not found"):
        flip_labels(label_df, target_col="missing")


def test_flip_labels_fraction_bounds(label_df):
    with pytest.raises(ValueError):
        flip_labels(label_df, target_col="label", flip_fraction=1.1)


def test_flip_labels_single_class():
    df = pd.DataFrame({"label": ["only"] * 10})
    with pytest.raises(ValueError):
        flip_labels(df, target_col="label", flip_fraction=0.5)


@pytest.mark.parametrize("flip_fraction", [0.0, 1.0])
def test_flip_labels_fraction_extremes(label_df, flip_fraction):
    flipped = flip_labels(label_df, target_col="label", flip_fraction=flip_fraction, seed=42)
    if flip_fraction == 0.0:
        assert (flipped["label"] == label_df["label"]).all()
    elif flip_fraction == 1.0:
        assert (flipped["label"] != label_df["label"]).any()

# Test for generate_combined_synthetic_data()
# Mock synthetic generators (patch if needed during integration testing)
def mock_generate_synthetic_data_sdv(df, num_rows, model, verbose):
    data = df.sample(min(len(df), num_rows), replace=True).reset_index(drop=True)
    return data, pd.DataFrame({"source": ["sdv"] * len(data)})

def mock_generate_synthetic_data_synthcity(df, target_col, n_rows, model, n_iter, batch_size, lr, device):
    data = df.sample(min(len(df), n_rows), replace=True).reset_index(drop=True)
    return data, pd.DataFrame({"source": ["synthcity"] * len(data)})

# Patch the generators
@pytest.fixture(autouse=True)
def patch_generators(monkeypatch):
    monkeypatch.setattr("edge_research.mining.generate_synthetic_data_sdv", mock_generate_synthetic_data_sdv)
    monkeypatch.setattr("edge_research.mining.generate_synthetic_data_synthcity", mock_generate_synthetic_data_synthcity)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1, 0, 1, 0],
        "feature2": [0, 1, 0, 1],
        "forward_return": [1, 0, 1, 0]
    })


@pytest.mark.parametrize("to_sdv, to_synthcity, expected_keys", [
    (True, False, {"sdv"}),
    (False, True, {"synthcity"}),
    (True, True, {"sdv", "synthcity"})
])
def test_generate_combined_synthetic_data_variants(sample_df, to_sdv, to_synthcity, expected_keys):
    df_out, logs = generate_combined_synthetic_data(
        df=sample_df,
        target_col="forward_return",
        to_sdv=to_sdv,
        to_synthcity=to_synthcity,
        sdv_rows=3,
        sc_rows=3,
        silence=True
    )

    # Output dataframe must not be empty
    assert not df_out.empty
    # Logs must include only expected keys
    assert set(logs.keys()) == expected_keys
    # Each log entry must be a DataFrame
    assert all(isinstance(log, pd.DataFrame) for log in logs.values())


def test_generate_combined_synthetic_data_raises_on_no_generators(sample_df):
    with pytest.raises(RuntimeError, match="No synthetic data was generated"):
        generate_combined_synthetic_data(
            df=sample_df,
            target_col="forward_return",
            to_sdv=False,
            to_synthcity=False
        )


def test_generate_combined_synthetic_data_empty_input():
    empty_df = pd.DataFrame(columns=["feature1", "feature2", "forward_return"])

    with pytest.raises(ValueError):
        # Underlying generator will likely fail on empty input
        generate_combined_synthetic_data(
            df=empty_df,
            target_col="forward_return",
            to_sdv=True,
            to_synthcity=False
        )


def test_generate_combined_synthetic_data_output_shape(sample_df):
    n_rows = 5
    df_out, logs = generate_combined_synthetic_data(
        df=sample_df,
        target_col="forward_return",
        to_sdv=True,
        to_synthcity=False,
        sdv_rows=n_rows
    )
    # Since we sample with replacement, row count should match (unless input smaller)
    assert len(df_out) == min(len(sample_df), n_rows)

# Test for augment_dataset()
# Fixtures for consistent input
@pytest.fixture
def base_df():
    return pd.DataFrame({
        "feature_a": [True, False, True, False],
        "feature_b": [False, False, True, True],
        "forward_return": [1, 0, 1, 0],
    })


def test_augment_dataset_no_ops_returns_unchanged(base_df):
    result = augment_dataset(base_df, target_col="forward_return")
    pd.testing.assert_frame_equal(result, base_df)


@pytest.mark.parametrize("flip_feats_frac, flip_targs_frac", [
    (-0.1, 0.1),
    (1.1, 0.1),
    (0.1, -0.1),
    (0.1, 1.5),
])
def test_invalid_flip_fractions_raise(base_df, flip_feats_frac, flip_targs_frac):
    with pytest.raises(ValueError):
        augment_dataset(
            df=base_df,
            target_col="forward_return",
            to_aug_flip_feats=True,
            to_aug_flip_targets=True,
            flip_feats_frac=flip_feats_frac,
            flip_targs_frac=flip_targs_frac,
        )


def test_missing_target_column_raises(base_df):
    df = base_df.drop(columns=["forward_return"])
    with pytest.raises(ValueError, match="Target column 'forward_return' not found"):
        augment_dataset(df, target_col="forward_return")


def test_flip_boolean_features_only_applies_to_boolean_cols(base_df):
    result = augment_dataset(
        df=base_df,
        target_col="forward_return",
        to_aug_flip_feats=True,
        flip_feats_frac=0.5,
        random_state=123
    )
    assert set(result.columns) == set(base_df.columns)
    assert result.dtypes["feature_a"] == "bool"
    assert result.dtypes["feature_b"] == "bool"


def test_label_flipping_changes_some_labels(base_df):
    result = augment_dataset(
        df=base_df,
        target_col="forward_return",
        to_aug_flip_targets=True,
        flip_targs_frac=0.5,
        random_state=999
    )
    assert not (result["forward_return"] == base_df["forward_return"]).all()


def test_combined_augmentations_output_shape(base_df):
    result = augment_dataset(
        df=base_df,
        target_col="forward_return",
        to_aug_imbalance=True,
        to_aug_flip_feats=True,
        to_aug_flip_targets=True,
        flip_feats_frac=0.2,
        flip_targs_frac=0.2,
        random_state=123
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(base_df.columns)
    assert len(result) > 0

# Test for mine_stats()
# Dummy mining functions to patch
@pytest.fixture(autouse=True)
def patch_miner_dependencies(monkeypatch):
    dummy_stats = pd.DataFrame([{"antecedents": "('feat1', 1)", "rule_depth": 1}])
    dummy_log = pd.DataFrame([{"info": "log"}])
    dummy_rules = [[("feat1", 1), ("feat2", 0)]]

    monkeypatch.setattr("edge_research.mining.mine_univar", lambda df, cfg: (dummy_stats, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_apriori", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_rulefit", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_subgroup", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_elcs", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_cn2", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr("edge_research.mining.mine_cart", lambda *args, **kwargs: (dummy_rules, dummy_log))
    monkeypatch.setattr(
        "edge_research.mining.mine_multivar",
        lambda df, sources, target_col: (
            pd.DataFrame([{"antecedents": "('featA', 1) AND ('featB', 0)", "rule_depth": 2}]),
            dummy_log,
            pd.DataFrame([{"algorithm": "apriori", "unique_rule_count": 1}])
        )
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feat1": [1, 0, 1, 0],
        "feat2": [0, 1, 1, 0],
        "forward_return": [1, 0, 1, 0]
    })


@pytest.mark.parametrize("miners", [
    ["univar"],
    ["rulefit"],
    ["apriori"],
    ["univar", "apriori", "rulefit"],
])
def test_mine_stats_valid_outputs(sample_df, miners):
    cfg = {"dummy_config": True}
    stats_df, logs, rules_df = mine_stats(df=sample_df, target_col="forward_return", miners=miners, cfg=cfg)

    # Output must be dataframes
    assert isinstance(stats_df, pd.DataFrame)
    assert isinstance(rules_df, pd.DataFrame)
    assert isinstance(logs, dict)

    # Logs must contain keys for each miner used
    for m in miners:
        assert m in logs or m == "univar"
    if "univar" in miners:
        assert "multivar" in logs  # triggered by combined rule_sources


def test_mine_stats_unknown_miner_raises(sample_df):
    cfg = {}
    with pytest.raises(ValueError, match="Unrecognized miners"):
        mine_stats(df=sample_df, target_col="forward_return", miners=["badminer"], cfg=cfg)


def test_mine_stats_empty_miners_returns_empty_stats(sample_df):
    cfg = {}
    stats_df, logs, rules_df = mine_stats(df=sample_df, target_col="forward_return", miners=[], cfg=cfg)
    assert stats_df.empty
    assert rules_df.empty
    assert logs == {}


def test_mine_stats_only_univar_produces_rules_row(sample_df):
    cfg = {"dummy": True}
    stats_df, logs, rules_df = mine_stats(df=sample_df, target_col="forward_return", miners=["univar"], cfg=cfg)
    assert "univar" in logs
    assert "algorithm" in rules_df.columns
    assert (rules_df["algorithm"] == "univar").any()

# Test for coalesce_data()
@pytest.fixture
def real_df():
    return pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})


@pytest.fixture
def synth_df():
    return pd.DataFrame({"id": [3, 4], "val": ["c", "d"]})


@pytest.fixture
def augmented_real_df():
    return pd.DataFrame({"id": [1, 2], "val": ["a*", "b*"]})


@pytest.fixture
def augmented_synth_df():
    return pd.DataFrame({"id": [3, 4], "val": ["c*", "d*"]})


def test_real_only(real_df):
    result = coalesce_data(real_df, None, None, None)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), real_df.reset_index(drop=True))


def test_real_plus_synth(real_df, synth_df):
    result = coalesce_data(real_df, synth_df, None, None)
    expected = pd.concat([real_df, synth_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected)


def test_augmented_real_only(real_df, augmented_real_df):
    result = coalesce_data(real_df, None, augmented_real_df, None)
    pd.testing.assert_frame_equal(result, augmented_real_df)


def test_augmented_synth_only(real_df, synth_df, augmented_synth_df):
    result = coalesce_data(real_df, synth_df, None, augmented_synth_df)
    expected = pd.concat([real_df, augmented_synth_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected)


def test_augmented_real_and_augmented_synth(real_df, augmented_real_df, synth_df, augmented_synth_df):
    result = coalesce_data(real_df, synth_df, augmented_real_df, augmented_synth_df)
    expected = pd.concat([augmented_real_df, augmented_synth_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected)


def test_augmented_real_and_unaugmented_synth(real_df, augmented_real_df, synth_df):
    result = coalesce_data(real_df, synth_df, augmented_real_df, None)
    expected = pd.concat([augmented_real_df, synth_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected)


def test_edge_case_empty_real_but_augment_still_works():
    df = pd.DataFrame()
    result = coalesce_data(df, None, None, None)
    pd.testing.assert_frame_equal(result, df)


def test_inconsistent_columns_raises(real_df, synth_df):
    synth_df = synth_df.rename(columns={"val": "different_col"})
    with pytest.raises(ValueError):
        coalesce_data(real_df, synth_df, None, None)

# Test for data_prep_pipeline()
@pytest.fixture
def base_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "date": pd.date_range("2022-01-01", periods=4),
        "feature": [1, 2, 3, 4],
        "target": [0, 1, 0, 1]
    })


@pytest.fixture
def minimal_cfg():
    return SimpleNamespace(
        id_cols=["id"],
        date_col="date",
        target_col="target",
        drop_cols=[],
        to_sample=False,
        sample_size=100,
        drop_duplicates=True,
        to_sdv=False,
        to_synthcity=False,
        corrupt_data=False,
        corrupt_target=None,
        to_aug_imbalance=False,
        to_aug_flip_feats=False,
        to_aug_flip_targets=False,
        flip_feats_frac=0.0,
        flip_targs_frac=0.0,
        sdv_model="gaussian_copula",
        sdv_rows=10,
        sdv_verbose=False,
        sc_model="ctgan",
        sc_rows=10,
        sc_n_iter=10,
        sc_batch_size=2,
        sc_lr=0.01,
        sc_device="cpu",
        synth_silence=True,
        log_max_rows=5
    )


def test_pipeline_runs_basic_real_only(base_df, minimal_cfg):
    df_out, logs = data_prep_pipeline(df=base_df, cfg=minimal_cfg)
    assert isinstance(df_out, pd.DataFrame)
    assert "prep_log" in logs
    assert isinstance(logs["prep_log"], pd.DataFrame)
    assert "synth_logs" in logs
    assert logs["synth_logs"] == {}


def test_pipeline_override_config_flag(base_df, minimal_cfg):
    # Overrides to enable synthetic data but skip actual synth logic
    df_out, logs = data_prep_pipeline(
        df=base_df,
        cfg=minimal_cfg,
        to_sdv=True,
        to_synthcity=False,
    )
    assert isinstance(df_out, pd.DataFrame)
    assert "synth_logs" in logs


def test_pipeline_with_empty_dataframe(minimal_cfg):
    df = pd.DataFrame(columns=["id", "date", "feature", "target"])
    result_df, logs = data_prep_pipeline(df=df, cfg=minimal_cfg)
    assert result_df.empty
    assert isinstance(logs["prep_log"], pd.DataFrame)


def test_pipeline_invalid_config_raises(monkeypatch, base_df, minimal_cfg):
    # Patch config to remove required attr
    delattr(minimal_cfg, "target_col")
    with pytest.raises(AttributeError):
        data_prep_pipeline(df=base_df, cfg=minimal_cfg)


def test_pipeline_with_logger_mock(base_df, minimal_cfg):
    class MockLogger:
        def __init__(self):
            self.steps = []
        def log_step(self, step_name: str, info: Any, df: Any, max_rows: int):
            self.steps.append((step_name, info, df))

    logger = MockLogger()
    df_out, logs = data_prep_pipeline(df=base_df, cfg=minimal_cfg, logger=logger)

    assert isinstance(df_out, pd.DataFrame)
    assert any("Prepping" in step for step, *_ in logger.steps)

# Test for mining_pipeline()
# Mocked dependencies
@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "feature1": [1, 0, 1, 0],
        "feature2": [0, 1, 1, 0],
        "target": [1, 0, 1, 0],
    })

@pytest.fixture
def dummy_cfg():
    return SimpleNamespace(
        target_col="target",
        log_max_rows=5,
        miners=["univar"],  # Minimal supported miner
        apriori_min_support=0.01,
        apriori_metric="lift",
        apriori_min_metric=0.0,
        rulefit_tree_size=3,
        rulefit_min_depth=2,
        subgroup_top_n=10,
        subgroup_depth=3,
        subgroup_beam_width=20,
        cart_max_depth=5,
        cart_criterion="gini",
        cart_random_state=42,
        cart_min_samples_split=2,
        cart_min_samples_leaf=1,
    )

class DummyLogger:
    def __init__(self):
        self.logged = []

    def log_step(self, step_name, info, df, max_rows):
        self.logged.append((step_name, info, df, max_rows))

def test_mining_pipeline_basic(dummy_df, dummy_cfg):
    result_df, logs = mining_pipeline(df=dummy_df, cfg=dummy_cfg)
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(logs, dict)
    assert "mining_logs" in logs
    assert isinstance(logs["mining_logs"], list)

def test_mining_pipeline_with_logger(dummy_df, dummy_cfg):
    logger = DummyLogger()
    result_df, logs = mining_pipeline(df=dummy_df, cfg=dummy_cfg, logger=logger)
    assert len(logger.logged) == 1
    assert logger.logged[0][0] == "Mining Rules"
    assert isinstance(result_df, pd.DataFrame)

def test_mining_pipeline_override_param(dummy_df, dummy_cfg):
    # Override miner to a different valid type
    result_df, logs = mining_pipeline(
        df=dummy_df,
        cfg=dummy_cfg,
        miners=["univar", "rulefit"]  # Assuming these are supported in test setup
    )
    assert isinstance(result_df, pd.DataFrame)
    assert "mining_logs" in logs
    assert len(logs["mining_logs"]) >= 1

def test_mining_pipeline_empty_df(dummy_cfg):
    empty_df = pd.DataFrame(columns=["feature1", "feature2", "target"])
    result_df, logs = mining_pipeline(df=empty_df, cfg=dummy_cfg)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty or "mining_logs" in logs  # Should not raise

def test_mining_pipeline_invalid_miner(dummy_df, dummy_cfg):
    with pytest.raises(ValueError):
        mining_pipeline(df=dummy_df, cfg=dummy_cfg, miners=["invalid_miner"])

@pytest.mark.parametrize("miner_list", [
    ["univar"],
    ["rulefit"],
    ["apriori"],
    ["cn2", "cart"],
    ["univar", "subgroup", "elcs"]
])
def test_mining_pipeline_supported_variants(dummy_df, dummy_cfg, miner_list):
    dummy_cfg.miners = miner_list
    result_df, logs = mining_pipeline(df=dummy_df, cfg=dummy_cfg)
    assert isinstance(result_df, pd.DataFrame)
    assert "mining_logs" in logs
    assert isinstance(logs["mining_logs"], list)
