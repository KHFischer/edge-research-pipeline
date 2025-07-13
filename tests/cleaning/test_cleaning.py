import pytest
import pandas as pd
import numpy as np
from pandas import Timestamp
from scipy.stats import norm

from edge_research.cleaning import (
    clean_raw_strings, 
    parse_datetime_column, 
    coerce_numeric_columns, 
    coerce_boolean_columns, 
    coerce_categorical_columns,
    drop_high_missingness,
    impute_numeric_per_group,
    fill_categorical_per_group,
    mask_high_imputation,
    zscore_flexible,
    robust_scale_flexible,
    quantile_rank_transform_flexible,
    winsorize_flexible,
    unit_vector_scale_flexible,
    custom_apply_flexible, 
    winsorize,
    drop_low_variance_columns,
    drop_highly_correlated_columns
)

# Test for clean_raw_strings()
def test_clean_raw_strings_basic_behavior():
    df = pd.DataFrame({
        "a": ["  foo ", "Bar\n", "\tn/a", None, " "],
        "b": [1, 2, 3, 4, 5],  # numeric column (should remain unchanged)
    })
    result = clean_raw_strings(df, cols=["a"])

    expected = pd.Series(["foo", "bar", np.nan, "none", np.nan], name="a", dtype=object)
    assert result["a"].isna().sum() == 2
    assert result["b"].equals(df["b"])
    assert result["a"].iloc[0] == "foo"
    assert result["a"].iloc[1] == "bar"
    assert pd.isna(result["a"].iloc[2])  # n/a
    assert pd.isna(result["a"].iloc[4])  # single space

def test_clean_raw_strings_all_mode_on_object_columns():
    df = pd.DataFrame({
        "x": ["\tYES", " no ", "  True  ", "\nfalse", None],
        "y": ["null", " valid ", "N/A", ".", "-"]
    })
    result = clean_raw_strings(df, cols="all")

    assert pd.isna(result["x"].iloc[3])  # 'false' is in null_like
    assert result["y"].iloc[1] == "valid"
    assert pd.isna(result["y"].iloc[2])  # 'N/A' becomes np.nan

def test_clean_raw_strings_column_skipped_if_not_convertible():
    df = pd.DataFrame({
        "a": ["keep", " clean\n", "NULL"],
        "b": [123, 456, 789]  # numeric column, should be untouched
    })
    result = clean_raw_strings(df, cols="all")
    assert result["b"].equals(df["b"])
    assert result["a"].iloc[1] == "clean"
    assert pd.isna(result["a"].iloc[2])

def test_clean_raw_strings_empty_dataframe_returns_same_shape():
    df = pd.DataFrame(columns=["x", "y"])
    out = clean_raw_strings(df, cols="all")
    assert out.shape == df.shape
    assert set(out.columns) == set(df.columns)

def test_clean_raw_strings_invalid_cols_raises():
    df = pd.DataFrame({"a": ["foo", "bar"]})
    with pytest.raises(ValueError):
        clean_raw_strings(df, cols=["nonexistent_col"])

def test_clean_raw_strings_one_element_column():
    df = pd.DataFrame({"a": ["  \tNone\n"]})
    result = clean_raw_strings(df, cols="all")
    assert pd.isna(result["a"].iloc[0])

# Test for parse_datetime_column()
@pytest.mark.parametrize(
    "values, floor, expected",
    [
        # Typical mixed inputs: ISO, US, space, bad value
        (
            ['2022-01-01 08:15', ' 2022/01/02 ', 'bad', '', None],
            False,
            [
                Timestamp('2022-01-01 08:15', tz='UTC'),
                Timestamp('2022-01-02 00:00', tz='UTC'),
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
        ),
        # Floor to day enabled: times should go to midnight
        (
            ['2022-01-01 17:33', '2022-01-02 00:00:00', '2022-01-03'],
            True,
            [
                Timestamp('2022-01-01 00:00', tz='UTC'),
                Timestamp('2022-01-02 00:00', tz='UTC'),
                Timestamp('2022-01-03 00:00', tz='UTC'),
            ],
        ),
        # All unparseable
        (
            ['nonsense', 'not a date', '---'],
            False,
            [pd.NaT, pd.NaT, pd.NaT],
        ),
        # Already parsed datetimes (should be converted to UTC, floored if specified)
        (
            [Timestamp('2022-01-01 12:00'), Timestamp('2022-01-02', tz='UTC')],
            True,
            [
                Timestamp('2022-01-01 00:00', tz='UTC'),
                Timestamp('2022-01-02 00:00', tz='UTC'),
            ],
        ),
        # One element, whitespace
        (
            [' 2022-12-31 '],
            False,
            [Timestamp('2022-12-31 00:00', tz='UTC')],
        ),
    ]
)
def test_parse_datetime_column_various_cases(values, floor, expected):
    df = pd.DataFrame({'d': values})
    result = parse_datetime_column(df, 'd', floor_to_day=floor)
    expected_series = pd.Series(expected, name='d')
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected_series, check_freq=False)

def test_parse_datetime_column_empty_input():
    df = pd.DataFrame({'dt': []})
    out = parse_datetime_column(df, 'dt')
    assert out.empty
    assert isinstance(out, pd.Series)
    assert out.name == 'dt'

@pytest.mark.parametrize("bad_df", [None, [], "not_a_df", 123])
def test_parse_datetime_column_invalid_df_raises(bad_df):
    with pytest.raises(ValueError):
        parse_datetime_column(bad_df, 'col')

def test_parse_datetime_column_missing_column_raises():
    df = pd.DataFrame({'other': [1, 2, 3]})
    with pytest.raises(ValueError):
        parse_datetime_column(df, 'notfound')

def test_parse_datetime_column_does_not_mutate_input():
    # Input with whitespace, should not change original df
    df = pd.DataFrame({'d': [' 2022-01-01 09:00 ']})
    df_copy = df.copy(deep=True)
    _ = parse_datetime_column(df, 'd')
    pd.testing.assert_frame_equal(df, df_copy)

# Test for coerce_numeric_columns()
import pytest
import pandas as pd
import numpy as np
from your_module import coerce_numeric_columns  # Update this import to match your module

def test_coerce_numeric_columns_basic_behavior():
    df = pd.DataFrame({
        "a": ["1", "2", "3"],
        "b": ["4.5", "5.0", "not_a_number"],
        "c": ["foo", "bar", "baz"],  # fully non-numeric
        "d": [True, False, True],    # should be untouched if not object
    })
    result = coerce_numeric_columns(df, cols=["a", "b", "c"])

    assert pd.api.types.is_integer_dtype(result["a"])
    assert result["a"].tolist() == [1, 2, 3]

    assert pd.api.types.is_float_dtype(result["b"])
    assert result["b"].iloc[2] is np.nan or pd.isna(result["b"].iloc[2])

    assert result["c"].equals(df["c"])  # should be unchanged

    assert result["d"].equals(df["d"])  # untouched

def test_coerce_numeric_columns_all_mode_respects_string_cols():
    df = pd.DataFrame({
        "a": ["100", "200", "300"],
        "b": ["x", "y", "z"],  # uncoercible
        "c": [1, 2, 3],        # already numeric
    })
    result = coerce_numeric_columns(df, cols="all")

    assert result["a"].tolist() == [100, 200, 300]
    assert result["b"].equals(df["b"])  # unchanged
    assert result["c"].equals(df["c"])  # untouched

def test_coerce_numeric_columns_handles_mixed_strings_and_nans():
    df = pd.DataFrame({
        "vals": ["1.0", "NaN", "3", None, "bad"]
    })
    result = coerce_numeric_columns(df, cols=["vals"])

    assert pd.api.types.is_float_dtype(result["vals"])
    assert result["vals"].notna().sum() == 2  # Only '1.0' and '3' valid
    assert pd.isna(result["vals"].iloc[1])
    assert pd.isna(result["vals"].iloc[3])
    assert pd.isna(result["vals"].iloc[4])

def test_coerce_numeric_columns_invalid_column_name_raises():
    df = pd.DataFrame({"a": ["1", "2", "3"]})
    with pytest.raises(ValueError):
        coerce_numeric_columns(df, cols=["a", "nonexistent"])

def test_coerce_numeric_columns_skips_fully_invalid_columns():
    df = pd.DataFrame({
        "x": ["a", "b", "c"],
        "y": ["1", "2", "3"]
    })
    result = coerce_numeric_columns(df, cols=["x", "y"])

    assert result["x"].equals(df["x"])  # unchanged
    assert result["y"].tolist() == [1, 2, 3]

def test_coerce_numeric_columns_empty_df_returns_empty():
    df = pd.DataFrame(columns=["a", "b"])
    out = coerce_numeric_columns(df, cols="all")
    assert out.shape == (0, 2)
    assert set(out.columns) == {"a", "b"}

def test_coerce_numeric_columns_preserves_float_dtype_if_any_decimals():
    df = pd.DataFrame({
        "a": ["1", "2.5", "3"]
    })
    result = coerce_numeric_columns(df, cols=["a"])
    assert pd.api.types.is_float_dtype(result["a"])
    assert result["a"].tolist() == [1.0, 2.5, 3.0]

# Test for coerce_boolean_columns()
def test_coerce_boolean_columns_basic_truthy_falsy():
    df = pd.DataFrame({
        "a": ["Yes", " no ", " TRUE", "0", "foo", None]
    })
    result = coerce_boolean_columns(df, cols=["a"])

    expected = pd.Series([True, False, True, False, pd.NA, pd.NA], dtype="boolean", name="a")
    pd.testing.assert_series_equal(result["a"], expected)

def test_coerce_boolean_columns_mixed_cases_all_mode():
    df = pd.DataFrame({
        "col1": ["Y", "n", "T", "F", "1", "0", "N/A", ""],
        "col2": ["x", "yes", "False", None, "true", "no", "maybe", "n"]
    })
    result = coerce_boolean_columns(df, cols="all")

    assert result["col1"].tolist() == [True, False, True, False, True, False, pd.NA, pd.NA]
    assert result["col2"].tolist() == [pd.NA, True, False, pd.NA, True, False, pd.NA, False]

def test_coerce_boolean_columns_empty_input():
    df = pd.DataFrame(columns=["flag"])
    out = coerce_boolean_columns(df, cols="all")
    assert out.shape == (0, 1)

def test_coerce_boolean_columns_one_valid_one_invalid():
    df = pd.DataFrame({
        "truthy": ["yes", "no", "bad", "  true  "],
        "junk": ["x", "y", "z", "?"]
    })
    result = coerce_boolean_columns(df, cols=["truthy", "junk"])

    assert result["truthy"].tolist() == [True, False, pd.NA, True]
    # "junk" column should remain unchanged since it's fully unrecognizable
    assert result["junk"].equals(df["junk"])

def test_coerce_boolean_columns_invalid_column_raises():
    df = pd.DataFrame({"flag": ["yes", "no", "true"]})
    with pytest.raises(ValueError):
        coerce_boolean_columns(df, cols=["flag", "missing_col"])

@pytest.mark.parametrize("val,expected", [
    ("TRUE", True),
    ("false", False),
    (" 1 ", True),
    (" 0", False),
    ("n/a", pd.NA),
    ("yes", True),
    ("No", False),
    ("", pd.NA),
    ("unknown", pd.NA),
])
def test_coerce_boolean_parametrized(val, expected):
    df = pd.DataFrame({"col": [val]})
    result = coerce_boolean_columns(df, cols=["col"])
    actual = result["col"].iloc[0]

    if expected is pd.NA:
        assert pd.isna(actual)
    else:
        assert actual == expected

# Test for coerce_categorical_columns()
def test_basic_conversion_to_categorical():
    df = pd.DataFrame({
        "col": ["apple", " banana ", "APPLE", "banana", "  ", None]
    })
    out = coerce_categorical_columns(df, cols=["col"])
    assert is_categorical_dtype(out["col"])
    expected_categories = ["apple", "banana", "none"]
    # Check values are normalized and whitespace-stripped
    assert set(out["col"].dropna().unique()) == {"apple", "banana"}
    assert out["col"].isna().sum() == 2


def test_all_mode_selects_string_columns_only():
    df = pd.DataFrame({
        "a": ["X", "Y", "Z"],
        "b": [1, 2, 3],
        "c": [True, False, True],
    })
    out = coerce_categorical_columns(df, cols="all")
    assert is_categorical_dtype(out["a"])
    assert not is_categorical_dtype(out["b"])
    assert not is_categorical_dtype(out["c"])


def test_preserve_existing_categoricals():
    df = pd.DataFrame({
        "cat": pd.Series(["a", "b", "c"], dtype="category"),
        "raw": [" x ", "y", " z"]
    })
    out = coerce_categorical_columns(df, cols=["cat", "raw"])
    assert is_categorical_dtype(out["cat"])
    assert out["cat"].equals(df["cat"])  # untouched
    assert is_categorical_dtype(out["raw"])
    assert "x" in out["raw"].cat.categories


def test_handles_blank_or_null_only_column():
    df = pd.DataFrame({
        "blank": ["", " ", None]
    })
    out = coerce_categorical_columns(df, cols=["blank"])
    # Should not be converted
    assert not is_categorical_dtype(out["blank"])
    assert out["blank"].equals(df["blank"])


@pytest.mark.parametrize("lowercase,strip,expected", [
    (True, True, ["apple", "banana"]),
    (False, True, ["apple", "banana"]),
    (True, False, [" apple", "banana"]),
    (False, False, [" apple", "banana"]),
])
def test_parametrized_normalization_behavior(lowercase, strip, expected):
    df = pd.DataFrame({"f": [" apple ", "Banana"]})
    out = coerce_categorical_columns(df, cols=["f"], lowercase=lowercase, strip=strip)
    actual_cats = list(out["f"].cat.categories)
    assert actual_cats == expected


def test_drop_unused_categories_false_retains_levels():
    df = pd.DataFrame({"c": ["a", "b", "b", "c", "d"]})
    # Set values to only "a" and "b"
    df.loc[2:] = None
    out = coerce_categorical_columns(df, cols=["c"], drop_unused_categories=False)
    assert set(out["c"].cat.categories) >= {"a", "b", "c", "d"}
    assert pd.isna(out["c"]).sum() == 3


def test_invalid_column_name_raises():
    df = pd.DataFrame({"valid": ["x", "y", "z"]})
    with pytest.raises(ValueError):
        coerce_categorical_columns(df, cols=["valid", "missing"])

# Test for drop_high_missingness()
@pytest.mark.parametrize(
    "df_in, row_thresh, col_thresh, expected_shape, expected_log_types",
    [
        # Nothing dropped (no row/col exceeds threshold)
        (
            pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
            0.99, 0.99,
            (3, 2),
            []  # log should be empty
        ),
        # Drop one row (row 1, 100% missing)
        (
            pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, np.nan, 6]}),
            0.5, 0.99,
            (2, 2),
            ['row']
        ),
        # Drop one column (col "b", 2/3 missing > 0.5)
        (
            pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, 3]}),
            0.99, 0.5,
            (3, 1),
            ['column']
        ),
        # Drop both (row 2 and col "b")
        (
            pd.DataFrame({'a': [1, 2, np.nan], 'b': [np.nan, np.nan, np.nan]}),
            0.5, 0.5,
            (2, 1),
            ['row', 'column']
        ),
        # Single-row/single-col input (should drop if all missing)
        (
            pd.DataFrame({'a': [np.nan]}),
            0.5, 0.5,
            (0, 0),  # everything dropped
            ['row', 'column']
        ),
    ]
)
def test_drop_high_missingness_various_cases(df_in, row_thresh, col_thresh, expected_shape, expected_log_types):
    cleaned, log = drop_high_missingness(df_in, row_thresh=row_thresh, col_thresh=col_thresh)
    assert cleaned.shape == expected_shape
    # Check that log has the right types and number of entries
    log_types = log["type"].tolist()
    for expected_type in expected_log_types:
        assert expected_type in log_types
    assert len(log_types) == len(expected_log_types)


def test_drop_high_missingness_empty_df():
    df = pd.DataFrame()
    cleaned, log = drop_high_missingness(df)
    assert cleaned.empty
    assert log.empty


@pytest.mark.parametrize("bad_input", [None, [], "not_a_df", 123])
def test_drop_high_missingness_invalid_df_raises(bad_input):
    with pytest.raises(ValueError):
        drop_high_missingness(bad_input)

@pytest.mark.parametrize("bad_thresh", [-0.1, 0, 1, 1.1])
def test_drop_high_missingness_invalid_thresholds_raise(bad_thresh):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(ValueError):
        drop_high_missingness(df, row_thresh=bad_thresh)
    with pytest.raises(ValueError):
        drop_high_missingness(df, col_thresh=bad_thresh)

def test_drop_high_missingness_does_not_modify_input():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, np.nan, 6]})
    df_copy = df.copy(deep=True)
    _ = drop_high_missingness(df, row_thresh=0.5, col_thresh=0.99)
    pd.testing.assert_frame_equal(df, df_copy)

def test_drop_high_missingness_log_structure():
    df = pd.DataFrame({'a': [1, np.nan, np.nan], 'b': [np.nan, np.nan, 3]})
    _, log = drop_high_missingness(df, row_thresh=0.5, col_thresh=0.5)
    assert set(log.columns) == {'type', 'row_index', 'column', 'missing_fraction'}
    assert log['type'].isin(['row', 'column']).all()

# Test for impute_numeric_per_group()
@pytest.mark.parametrize(
    "df,id_cols,impute_cols,impute_strategy,expected_df,expected_log_success",
    [
        # Single group, mean strategy
        (
            pd.DataFrame({'id': ['A', 'A'], 'x': [1.0, np.nan]}),
            ['id'], ['x'], 'mean',
            pd.DataFrame({'id': ['A', 'A'], 'x': [1.0, 1.0]}),
            [True]
        ),
        # Multiple groups, median
        (
            pd.DataFrame({'id': ['A', 'A', 'B', 'B'], 'x': [1, np.nan, np.nan, 2]}),
            ['id'], ['x'], 'median',
            pd.DataFrame({'id': ['A', 'A', 'B', 'B'], 'x': [1, 1, 2, 2]}),
            [True, True]
        ),
        # Multi-col imputation, different group stats
        (
            pd.DataFrame({
                'id': ['A', 'A', 'B', 'B'],
                'x': [1, np.nan, np.nan, 2],
                'y': [3, 4, np.nan, np.nan]
            }),
            ['id'], ['x', 'y'], 'mean',
            pd.DataFrame({
                'id': ['A', 'A', 'B', 'B'],
                'x': [1, 1, 2, 2],
                'y': [3, 4, np.nan, np.nan]  # mean for B's y is nan
            }),
            [True, True, True, False]  # x:A, x:B, y:A, y:B
        ),
        # All missing in group for a column (should not impute)
        (
            pd.DataFrame({
                'id': ['C', 'C'],
                'x': [np.nan, np.nan]
            }),
            ['id'], ['x'], 'median',
            pd.DataFrame({'id': ['C', 'C'], 'x': [np.nan, np.nan]}),
            [False]
        ),
    ]
)
def test_impute_numeric_per_group_cases(
    df, id_cols, impute_cols, impute_strategy, expected_df, expected_log_success
):
    imputed, log = impute_numeric_per_group(
        df, id_cols, impute_cols, impute_strategy=impute_strategy
    )
    # Compare imputed values
    pd.testing.assert_frame_equal(imputed.reset_index(drop=True), expected_df.reset_index(drop=True))
    # Compare fill_successful status
    # There will be one log row per group*col
    assert log['fill_successful'].tolist() == expected_log_success

def test_impute_numeric_per_group_empty_df():
    df = pd.DataFrame(columns=['id', 'x'])
    imputed, log = impute_numeric_per_group(df, id_cols=['id'], impute_cols=['x'])
    assert imputed.empty
    assert log.empty

@pytest.mark.parametrize(
    "bad_strategy",
    ['min', 'max', 'average', 123, None]
)
def test_impute_numeric_per_group_invalid_strategy_raises(bad_strategy):
    df = pd.DataFrame({'id': ['A'], 'x': [1]})
    with pytest.raises(ValueError):
        impute_numeric_per_group(df, id_cols=['id'], impute_cols=['x'], impute_strategy=bad_strategy)

@pytest.mark.parametrize(
    "bad_id_cols,bad_impute_cols",
    [
        (['not_a_col'], ['x']),
        (['id'], ['not_a_col']),
        (['not_a_col'], ['not_a_col'])
    ]
)
def test_impute_numeric_per_group_missing_cols_raise(bad_id_cols, bad_impute_cols):
    df = pd.DataFrame({'id': ['A'], 'x': [1]})
    with pytest.raises(ValueError):
        impute_numeric_per_group(df, id_cols=bad_id_cols, impute_cols=bad_impute_cols)

def test_impute_numeric_per_group_does_not_modify_input():
    df = pd.DataFrame({'id': ['A', 'A'], 'x': [np.nan, 2]})
    df_copy = df.copy(deep=True)
    _ = impute_numeric_per_group(df, id_cols=['id'], impute_cols=['x'])
    pd.testing.assert_frame_equal(df, df_copy)

def test_impute_numeric_per_group_log_structure():
    df = pd.DataFrame({'id': ['A', 'A'], 'x': [1, np.nan]})
    _, log = impute_numeric_per_group(df, id_cols=['id'], impute_cols=['x'])
    assert set(log.columns) == {'id', 'column', 'count_filled', 'percent_filled', 'fill_value', 'fill_successful'}
    # Check column types
    assert all(isinstance(col, str) for col in log['column'])

# Test for fill_categorical_per_group()
@pytest.mark.parametrize(
    "df,id_cols,categorical_cols,expected_filled,expected_success",
    [
        # Single group, mode exists
        (
            pd.DataFrame({'id': ['A', 'A'], 'sector': ['Tech', None]}),
            ['id'],
            ['sector'],
            pd.DataFrame({'id': ['A', 'A'], 'sector': ['Tech', 'Tech']}),
            [True],
        ),
        # Multiple groups, each has a different mode
        (
            pd.DataFrame({
                'id': ['A', 'A', 'B', 'B'],
                'industry': ['Bank', None, None, 'Bank']
            }),
            ['id'],
            ['industry'],
            pd.DataFrame({
                'id': ['A', 'A', 'B', 'B'],
                'industry': ['Bank', 'Bank', 'Bank', 'Bank']
            }),
            [True, True]
        ),
        # Group with all missing (should remain NaN and log unsuccessful)
        (
            pd.DataFrame({'id': ['A', 'A'], 'region': [None, None]}),
            ['id'],
            ['region'],
            pd.DataFrame({'id': ['A', 'A'], 'region': [None, None]}),
            [False],
        ),
        # Multi-column: only one needs fill, the other is already complete
        (
            pd.DataFrame({
                'id': ['A', 'A', 'B'],
                'sector': ['Tech', None, 'Finance'],
                'region': ['US', 'US', None]
            }),
            ['id'],
            ['sector', 'region'],
            pd.DataFrame({
                'id': ['A', 'A', 'B'],
                'sector': ['Tech', 'Tech', 'Finance'],
                'region': ['US', 'US', None]
            }),
            [True, True, False]  # sector:A, sector:B, region:B
        ),
    ]
)
def test_fill_categorical_per_group_typical_cases(
    df, id_cols, categorical_cols, expected_filled, expected_success
):
    filled, log = fill_categorical_per_group(df, id_cols, categorical_cols)
    pd.testing.assert_frame_equal(filled.reset_index(drop=True), expected_filled.reset_index(drop=True))
    # Check log success flags match expectations (one per group*col)
    assert log['fill_successful'].tolist() == expected_success

def test_fill_categorical_per_group_empty_df():
    df = pd.DataFrame(columns=['id', 'cat'])
    categorical_cols = ['cat']
    filled, log = fill_categorical_per_group(df, id_cols=['id'], categorical_cols=categorical_cols)
    assert filled.empty
    assert log.empty

@pytest.mark.parametrize(
    "df,id_cols,categorical_cols,expected_exception",
    [
        # categorical_cols is not a list
        (pd.DataFrame({'id': ['A']}), ['id'], 'cat', ValueError),
        # id_col not in DataFrame
        (pd.DataFrame({'cat': [None]}), ['id'], ['cat'], ValueError),
        # categorical col not in DataFrame
        (pd.DataFrame({'id': ['A']}), ['id'], ['cat'], ValueError),
        # df is not a DataFrame
        ('not_a_df', ['id'], ['cat'], ValueError),
    ]
)
def test_fill_categorical_per_group_invalid_inputs(
    df, id_cols, categorical_cols, expected_exception
):
    with pytest.raises(expected_exception):
        fill_categorical_per_group(df, id_cols, categorical_cols)

def test_fill_categorical_per_group_no_mutation():
    df = pd.DataFrame({'id': ['A', 'A'], 'sector': ['Tech', None]})
    df_copy = df.copy(deep=True)
    categorical_cols = ['sector']
    _ = fill_categorical_per_group(df, id_cols=['id'], categorical_cols=categorical_cols)
    pd.testing.assert_frame_equal(df, df_copy)

def test_fill_categorical_per_group_log_structure():
    df = pd.DataFrame({'id': ['A', 'A'], 'sector': ['Tech', None]})
    categorical_cols = ['sector']
    _, log = fill_categorical_per_group(df, id_cols=['id'], categorical_cols=categorical_cols)
    assert set(log.columns) == {'id', 'column', 'count_filled', 'percent_filled', 'fill_value', 'fill_successful'}
    # Types
    assert all(isinstance(x, str) for x in log['column'])

# Test for mask_high_imputation()
@pytest.mark.parametrize(
    "df,log_dfs,id_cols,max_imputed,expected",
    [
        # Typical: Single group, over-threshold imputation (mask)
        (
            pd.DataFrame({"ticker": ["A", "A", "B"], "x": [1, 2, 3]}),
            [
                pd.DataFrame({
                    "ticker": ["A", "B"],
                    "column": ["x", "x"],
                    "imputed_successful": [True, True],
                    "percent_imputed": [0.7, 0.1]
                })
            ],
            ["ticker"],
            0.5,
            pd.DataFrame({"ticker": ["A", "A", "B"], "x": [pd.NA, pd.NA, 3]})
        ),
        # Multi-column, one over threshold, one not
        (
            pd.DataFrame({"id": ["A", "A", "B"], "foo": [0, 1, 2], "bar": [3, 4, 5]}),
            [
                pd.DataFrame({
                    "id": ["A", "B"],
                    "column": ["foo", "bar"],
                    "fill_successful": [True, True],
                    "percent_filled": [0.6, 0.2]
                })
            ],
            ["id"],
            0.5,
            pd.DataFrame({"id": ["A", "A", "B"], "foo": [pd.NA, pd.NA, 2], "bar": [3, 4, 5]})
        ),
        # Group below threshold (nothing masked)
        (
            pd.DataFrame({"group": ["X", "X", "Y"], "val": [5, 6, 7]}),
            [
                pd.DataFrame({
                    "group": ["X", "Y"],
                    "column": ["val", "val"],
                    "fill_successful": [True, True],
                    "percent_filled": [0.2, 0.2]
                })
            ],
            ["group"],
            0.5,
            pd.DataFrame({"group": ["X", "X", "Y"], "val": [5, 6, 7]})
        ),
        # Multi-id_col grouping, only one group masked
        (
            pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "col": [10, 20, 30]}),
            [
                pd.DataFrame({
                    "a": [1, 2],
                    "b": [3, 4],
                    "column": ["col", "col"],
                    "imputed_successful": [True, True],
                    "percent_imputed": [0.8, 0.3]
                })
            ],
            ["a", "b"],
            0.5,
            pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "col": [pd.NA, pd.NA, 30]})
        ),
        # Masking should skip columns not in df
        (
            pd.DataFrame({"id": [1, 2], "present": [10, 20]}),
            [
                pd.DataFrame({
                    "id": [1],
                    "column": ["missing"],
                    "imputed_successful": [True],
                    "percent_imputed": [0.99]
                })
            ],
            ["id"],
            0.5,
            pd.DataFrame({"id": [1, 2], "present": [10, 20]})
        ),
    ]
)
def test_mask_high_imputation_cases(df, log_dfs, id_cols, max_imputed, expected):
    masked = mask_high_imputation(df, log_dfs, id_cols, max_imputed)
    pd.testing.assert_frame_equal(masked.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)

def test_mask_high_imputation_empty_inputs():
    df = pd.DataFrame(columns=["a", "b"])
    log = pd.DataFrame(columns=["column", "imputed_successful", "percent_imputed"])
    masked = mask_high_imputation(df, [log], id_cols=["a"], max_imputed=0.5)
    assert masked.empty

@pytest.mark.parametrize(
    "df,log_dfs,id_cols,expected_exception",
    [
        # Not a DataFrame
        ("not_a_df", [], ["id"], ValueError),
        # log_dfs not a list
        (pd.DataFrame({"id": [1]}), "not_a_list", ["id"], ValueError),
        # log_dfs not all DataFrames
        (pd.DataFrame({"id": [1]}), [pd.DataFrame(), "bad"], ["id"], ValueError),
        # id_col not in df
        (pd.DataFrame({"not_id": [1]}), [pd.DataFrame()], ["id"], ValueError),
        # Missing success/percent columns
        (
            pd.DataFrame({"id": [1]}),
            [pd.DataFrame({"id": [1], "column": ["foo"]})],
            ["id"],
            ValueError
        ),
    ]
)
def test_mask_high_imputation_invalid_inputs(df, log_dfs, id_cols, expected_exception):
    with pytest.raises(expected_exception):
        mask_high_imputation(df, log_dfs, id_cols)

def test_mask_high_imputation_does_not_mutate_input():
    df = pd.DataFrame({"ticker": ["A", "A"], "val": [1, 2]})
    log = pd.DataFrame({"ticker": ["A"], "column": ["val"], "imputed_successful": [True], "percent_imputed": [0.9]})
    df_copy = df.copy(deep=True)
    _ = mask_high_imputation(df, [log], ["ticker"], max_imputed=0.5)
    pd.testing.assert_frame_equal(df, df_copy)

def test_mask_high_imputation_output_shape():
    df = pd.DataFrame({"id": [1, 2, 3], "col": [0, 1, 2]})
    log = pd.DataFrame({"id": [1, 2, 3], "column": ["col"]*3, "imputed_successful": [True]*3, "percent_imputed": [0.6, 0.7, 0.1]})
    masked = mask_high_imputation(df, [log], ["id"], max_imputed=0.5)
    # Only id 1, 2 should be masked
    assert masked.shape == df.shape
    assert masked.loc[masked['id'] == 3, 'col'].iloc[0] == 2
    assert masked.loc[masked['id'] == 1, 'col'].isna().iloc[0]
    assert masked.loc[masked['id'] == 2, 'col'].isna().iloc[0]

# Test for winsorize_flexible()
@pytest.mark.parametrize(
    "df,cols,expected,expected_log_len",
    [
        # No winsorization needed (already within quantiles)
        (
            pd.DataFrame({'a': [2, 3, 4], 'b': [1, 1, 1]}),
            ['a'],
            pd.DataFrame({'a': [2, 3, 4], 'b': [1, 1, 1]}),
            0
        ),
        # Some values are winsorized (outliers clipped)
        (
            pd.DataFrame({'x': [1, 2, 100, 4, 5]}),
            ['x'],
            pd.DataFrame({'x': [1.08, 2.0, 5.0, 4.0, 5.0]}),  # 1.08 = 1st percentile, 5 = 99th percentile
            2  # Two values clipped
        ),
        # Multi-column, different winsorization per column
        (
            pd.DataFrame({'a': [1, 1, 100], 'b': [10, 20, 30]}),
            ['a', 'b'],
            pd.DataFrame({'a': [1.0, 1.0, 67.0], 'b': [10.2, 20.0, 29.8]}),
            2  # One in a, one in b
        ),
    ]
)
def test_winsorize_flexible_none_grouping(df, cols, expected, expected_log_len):
    # Use quantiles to match expected (slightly more than min/max)
    result, log = winsorize_flexible(
        df, cols=cols, grouping="none", lower_quantile=0.01, upper_quantile=0.99
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), rtol=1e-2)
    assert isinstance(log, pd.DataFrame)
    assert len(log) == expected_log_len

def test_winsorize_flexible_ids_grouping():
    df = pd.DataFrame({
        'id': ['A', 'A', 'B', 'B', 'B'],
        'val': [1, 100, 2, 200, 3]
    })
    result, log = winsorize_flexible(
        df, cols=['val'], grouping='ids', id_cols=['id'], lower_quantile=0.1, upper_quantile=0.9
    )
    # For 'A', 1 and 100 are clipped to their own values, for 'B', 2 and 3 are not, 200 is winsorized down
    assert np.allclose(result.loc[df['id'] == 'A', 'val'].tolist(), [1.0, 100.0])
    assert np.allclose(result.loc[df['id'] == 'B', 'val'].tolist(), [2.0, 134.0, 3.0], atol=1e-2)
    assert 'group' in log.columns
    assert 'column' in log.columns

def test_winsorize_flexible_datetime_grouping():
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=6, freq='D'),
        'val': [1, 2, 3, 100, 4, 5]
    })
    result, log = winsorize_flexible(
        df, cols=['val'], grouping='datetime',
        date_col='date', n_datetime_units=3, lower_quantile=0.1, upper_quantile=0.9
    )
    # Group 1: [1,2,3], Group 2: [100,4,5]
    # Check clipped results in each window
    assert np.allclose(result.loc[:2, 'val'], [1.2, 2.0, 2.8], atol=1e-2)
    assert np.allclose(result.loc[3:, 'val'], [40.8, 4.0, 5.0], atol=1e-2)
    assert all(c in log.columns for c in ["group", "column", "row_index", "original_value", "winsorized_value", "was_winsorized"])

def test_winsorize_flexible_datetime_ids_grouping():
    df = pd.DataFrame({
        'id': ['A', 'A', 'B', 'B', 'B', 'A'],
        'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-03']),
        'val': [1, 100, 2, 200, 3, 4]
    })
    # Group by id, then rolling window within id
    result, log = winsorize_flexible(
        df, cols=['val'], grouping='datetime+ids', id_cols=['id'], date_col='date',
        n_datetime_units=2, lower_quantile=0.1, upper_quantile=0.9
    )
    # Just check that log contains correct structure and the right rows are clipped
    assert set(log.columns) == {"group", "column", "row_index", "original_value", "winsorized_value", "was_winsorized"}
    assert log['was_winsorized'].all()  # All rows in the log were clipped

def test_winsorize_flexible_all_cols_argument():
    df = pd.DataFrame({'x': [1, 99, 3], 'y': [100, 200, -100]})
    result, log = winsorize_flexible(df, cols='all', grouping='none', lower_quantile=0.1, upper_quantile=0.9)
    # Should winsorize both columns automatically
    assert list(result.columns) == ['x', 'y']
    assert 'x' in log['column'].values or 'y' in log['column'].values

def test_winsorize_flexible_invalid_inputs():
    df = pd.DataFrame({'a': [1, 2, 3]})
    # Not a DataFrame
    with pytest.raises(ValueError):
        winsorize_flexible("not_a_df", cols=['a'])
    # cols not a list or 'all'
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=123)
    # Some cols don't exist
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['not_a_col'])
    # Missing id_cols for ids grouping
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['a'], grouping='ids')
    # Missing date_col or n_datetime_units for datetime grouping
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['a'], grouping='datetime')
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['a'], grouping='datetime', date_col='a')
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['a'], grouping='datetime', n_datetime_units=2)
    # Invalid grouping
    with pytest.raises(ValueError):
        winsorize_flexible(df, cols=['a'], grouping='not_a_grouping')

def test_winsorize_flexible_output_no_mutation():
    df = pd.DataFrame({'x': [1, 100, 2]})
    df_copy = df.copy(deep=True)
    _ = winsorize_flexible(df, cols=['x'])
    pd.testing.assert_frame_equal(df, df_copy)

# Test for unit_vector_scale_flexible()
@pytest.fixture
def example_df():
    return pd.DataFrame({
        "id": ["A", "A", "A", "B", "B", "B", "B", "B"],
        "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"
        ]),
        "x": [1, 2, 3, 1, 2, 3, 4, np.nan],
        "y": [10, 11, 12, 20, 21, 22, 23, 24]
    })

@pytest.mark.parametrize("norm_type", ["l2", "l1", "max"])
def test_unit_vector_global_scaling(example_df, norm_type):
    df_out, log = unit_vector_scale_flexible(
        example_df, cols=['x'], grouping='none', norm_type=norm_type
    )
    x = example_df['x'].dropna().values.astype(float)
    if norm_type == 'l2':
        norm_val = np.linalg.norm(x, 2)
    elif norm_type == 'l1':
        norm_val = np.linalg.norm(x, 1)
    elif norm_type == 'max':
        norm_val = np.max(np.abs(x))
    expected = x / norm_val if norm_val != 0 else np.nan
    # Compare output for non-nan entries
    out_x = df_out.loc[~example_df['x'].isna(), 'x'].values
    assert np.allclose(out_x, expected, equal_nan=True)
    # Check log content
    assert log.iloc[0]['norm_type'] == norm_type
    assert np.isclose(log.iloc[0]['norm_value'], norm_val)
    assert log.iloc[0]['n_obs'] == len(x)

def test_unit_vector_ids_grouping(example_df):
    df_out, log = unit_vector_scale_flexible(
        example_df, cols=['y'], grouping='ids', id_cols=['id'], norm_type='l2'
    )
    # Group 'A' y values: [10, 11, 12]
    y_a = example_df.loc[example_df['id'] == 'A', 'y'].values.astype(float)
    expected_a = y_a / np.linalg.norm(y_a, 2)
    out_a = df_out.loc[example_df['id'] == 'A', 'y'].values
    assert np.allclose(out_a, expected_a)
    # Group 'B'
    y_b = example_df.loc[example_df['id'] == 'B', 'y'].values.astype(float)
    expected_b = y_b / np.linalg.norm(y_b, 2)
    out_b = df_out.loc[example_df['id'] == 'B', 'y'].values
    assert np.allclose(out_b, expected_b)
    # Log has both groups
    assert set(log['group']) == {'A', 'B'}

def test_unit_vector_datetime_grouping(example_df):
    df = example_df.copy().sort_values("date").reset_index(drop=True)
    # 3 rows per block = 3 groups: first 3, next 3, last 2
    df_out, log = unit_vector_scale_flexible(
        df, cols=['x'], grouping='datetime', date_col='date', n_datetime_units=3, norm_type='l2'
    )
    # Check group count and nontrivial scaling
    assert set(log['group']) == {'0', '1', '2'}
    for group in ['0', '1', '2']:
        idx = log['group'] == group
        n_obs = log.loc[idx, 'n_obs'].iloc[0]
        assert n_obs >= 1 or np.isnan(df_out.loc[log['group'] == group, 'x']).all()

def test_unit_vector_all_cols(example_df):
    df_out, log = unit_vector_scale_flexible(
        example_df, cols='all', grouping='none', norm_type='l2'
    )
    # Only numeric columns transformed
    for col in ['x', 'y']:
        col_vals = df_out[col].dropna()
        norm = np.linalg.norm(col_vals, 2)
        assert np.isclose(norm, 1.0)
    # Non-numeric columns untouched
    assert (df_out['id'] == example_df['id']).all()

def test_unit_vector_empty_df():
    df = pd.DataFrame(columns=['x', 'id'])
    df_out, log = unit_vector_scale_flexible(df, cols=['x'], grouping='none')
    assert df_out.empty
    assert log.empty

def test_unit_vector_singleton_group():
    df = pd.DataFrame({'x': [42], 'id': ['A']})
    df_out, log = unit_vector_scale_flexible(df, cols=['x'], grouping='ids', id_cols=['id'])
    assert np.isclose(df_out.loc[0, 'x'], 1.0)
    assert log.iloc[0]['n_obs'] == 1

def test_unit_vector_zero_norm():
    # All-zeros: should yield NaN
    df = pd.DataFrame({'x': [0, 0, 0]})
    df_out, log = unit_vector_scale_flexible(df, cols=['x'], grouping='none')
    assert np.isnan(df_out['x']).all()
    assert np.isclose(log.iloc[0]['norm_value'], 0)

def test_unit_vector_invalid_inputs(example_df):
    # Invalid grouping
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['x'], grouping='foo')
    # Invalid norm_type
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['x'], norm_type='bad_norm')
    # Invalid columns
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['not_in_df'])
    # ids without id_cols
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['x'], grouping='ids')
    # datetime without params
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['x'], grouping='datetime')
    # datetime with missing date_col
    with pytest.raises(ValueError):
        unit_vector_scale_flexible(example_df, cols=['x'], grouping='datetime', n_datetime_units=3)

# Test for zscore_flexible()
@pytest.mark.parametrize(
    "df,cols,grouping,id_cols,date_col,n_datetime_units,expected_values,expected_means,expected_stds",
    [
        # Global (none) grouping, 2 columns
        (
            pd.DataFrame({'x': [1, 2, 3], 'y': [2, 2, 2]}),
            ['x', 'y'], "none", None, None, None,
            pd.DataFrame({'x': [-1.224745, 0.0, 1.224745], 'y': [np.nan, np.nan, np.nan]}),
            [2.0, 2.0],    # means for x, y
            [0.8164966, 0.0]   # stds for x, y
        ),
        # ids grouping: by ticker, both groups length 2
        (
            pd.DataFrame({'ticker': ['A', 'A', 'B', 'B'], 'val': [1, 3, 2, 6]}),
            ['val'], "ids", ['ticker'], None, None,
            pd.DataFrame({'ticker': ['A', 'A', 'B', 'B'], 'val': [-1.0, 1.0, -1.0, 1.0]}),
            [2.0, 4.0],    # mean for A, B
            [1.0, 2.0],    # std for A, B
        ),
        # datetime grouping: rolling window size 2
        (
            pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
                          'x': [10, 12, 16, 16]}),
            ['x'], "datetime", None, 'date', 2,
            pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
                          'x': [-1.0, 1.0, 0.0, 0.0]}),
            [11.0, 16.0],  # means for each group
            [1.0, 0.0],    # stds for each group
        ),
        # cols='all' mode, should use all numeric columns
        (
            pd.DataFrame({'x': [1, 3, 5], 'y': [5, 7, 9]}),
            'all', "none", None, None, None,
            pd.DataFrame({'x': [-1.224745, 0.0, 1.224745], 'y': [-1.224745, 0.0, 1.224745]}),
            [3.0, 7.0],
            [1.632993, 1.632993]
        ),
        # ids grouping, std=0 edge case (constant group)
        (
            pd.DataFrame({'id': ['A', 'A', 'B'], 'val': [5, 5, 2]}),
            ['val'], "ids", ['id'], None, None,
            pd.DataFrame({'id': ['A', 'A', 'B'], 'val': [np.nan, np.nan, np.nan]}),
            [5.0, 2.0],
            [0.0, 0.0]
        ),
        # ids grouping, singleton group
        (
            pd.DataFrame({'id': ['A', 'B'], 'val': [10, 20]}),
            ['val'], "ids", ['id'], None, None,
            pd.DataFrame({'id': ['A', 'B'], 'val': [np.nan, np.nan]}),
            [10.0, 20.0],
            [0.0, 0.0]
        ),
    ]
)
def test_zscore_flexible_groupings_and_edge_cases(
    df, cols, grouping, id_cols, date_col, n_datetime_units,
    expected_values, expected_means, expected_stds
):
    z, log = zscore_flexible(
        df, cols=cols, grouping=grouping,
        id_cols=id_cols, date_col=date_col, n_datetime_units=n_datetime_units
    )
    # Check values (approx)
    pd.testing.assert_frame_equal(
        z.reset_index(drop=True)[expected_values.columns],
        expected_values.reset_index(drop=True)[expected_values.columns],
        check_exact=False, rtol=1e-5
    )
    # Check means and stds in log (all non-null groups)
    if len(expected_means) > 0:
        means = log['mean'].drop_duplicates().tolist()
        stds = log['std'].drop_duplicates().tolist()
        np.testing.assert_allclose(means, expected_means, rtol=1e-5)
        np.testing.assert_allclose(stds, expected_stds, rtol=1e-5)

def test_zscore_flexible_empty_input():
    df = pd.DataFrame({'x': [], 'y': []})
    z, log = zscore_flexible(df, cols=['x', 'y'], grouping='none')
    assert z.empty
    assert log.empty

@pytest.mark.parametrize(
    "grouping,id_cols,date_col,n_datetime_units,err_type",
    [
        ('ids', None, None, None, ValueError),
        ('ids', [], None, None, ValueError),
        ('ids', ['nonexistent'], None, None, ValueError),
        ('datetime', None, None, None, ValueError),
        ('datetime', None, None, 2, ValueError),
        ('datetime', None, 'dt', None, ValueError),
        ('datetime', None, 'nonexistent', 2, ValueError),
        ('bad', None, None, None, ValueError)
    ]
)
def test_zscore_flexible_invalid_args(grouping, id_cols, date_col, n_datetime_units, err_type):
    df = pd.DataFrame({'x': [1, 2], 'dt': pd.date_range('2020-01-01', periods=2)})
    with pytest.raises(err_type):
        zscore_flexible(
            df, cols=['x'],
            grouping=grouping, id_cols=id_cols,
            date_col=date_col, n_datetime_units=n_datetime_units
        )

def test_zscore_flexible_no_mutation():
    df = pd.DataFrame({'x': [1, 2, 3]})
    df_orig = df.copy(deep=True)
    _ = zscore_flexible(df, cols=['x'])
    pd.testing.assert_frame_equal(df, df_orig)

def test_zscore_flexible_output_shape_and_log_structure():
    df = pd.DataFrame({'x': [1, 2, 3]})
    z, log = zscore_flexible(df, cols=['x'])
    assert z.shape == df.shape
    assert set(log.columns) == {"group", "column", "mean", "std"}

# Test for robust_scale_flexible()
def example_df():
    return pd.DataFrame({
        "id": ["A", "A", "A", "B", "B", "B", "B", "B"],
        "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"
        ]),
        "x": [1, 2, 3, 1, 2, 3, 4, np.nan],
        "y": [10, 11, 12, 20, 21, 22, 23, 24]
    })

def test_robust_scale_global(example_df):
    df_out, log = robust_scale_flexible(
        example_df, cols=['x'], grouping='none'
    )
    x = example_df['x'].dropna()
    median = x.median()
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    expected = (x - median) / iqr if iqr != 0 else np.nan
    # Compare output for non-nan entries
    out_x = df_out.loc[x.index, 'x']
    assert np.allclose(out_x, expected, equal_nan=True)
    # Log checks
    assert log.iloc[0]['median'] == median
    assert log.iloc[0]['iqr'] == iqr
    assert log.iloc[0]['n_obs'] == len(x)

@pytest.mark.parametrize("quantile_range", [ (25, 75), [10, 90], (0, 100) ])
def test_robust_scale_quantile_range(example_df, quantile_range):
    df_out, log = robust_scale_flexible(
        example_df, cols=['y'], grouping='none', quantile_range=quantile_range
    )
    y = example_df['y']
    q_low, q_high = quantile_range
    expected_iqr = np.percentile(y, q_high) - np.percentile(y, q_low)
    assert np.isclose(log.iloc[0]['iqr'], expected_iqr)

def test_robust_scale_groupby_ids(example_df):
    df_out, log = robust_scale_flexible(
        example_df, cols=['x'], grouping='ids', id_cols=['id']
    )
    # Group 'A' values: [1,2,3]
    mask_a = example_df['id'] == 'A'
    x_a = example_df.loc[mask_a, 'x']
    median_a = x_a.median()
    q1_a = np.percentile(x_a, 25)
    q3_a = np.percentile(x_a, 75)
    iqr_a = q3_a - q1_a
    expected_a = (x_a - median_a) / iqr_a if iqr_a != 0 else np.nan
    out_a = df_out.loc[mask_a, 'x']
    assert np.allclose(out_a, expected_a, equal_nan=True)
    # Check log for both groups
    assert set(log['group']) == {'A', 'B'}

def test_robust_scale_groupby_datetime(example_df):
    # 3 per block, so 3 groups: first 3 rows, next 3, last 2
    df = example_df.copy().sort_values("date").reset_index(drop=True)
    df_out, log = robust_scale_flexible(
        df, cols=['y'], grouping='datetime', date_col='date', n_datetime_units=3
    )
    assert set(log['group']) == {'0', '1', '2'}
    # Each block: y scaled within block
    for group, group_log in log.groupby('group'):
        n_obs = group_log['n_obs'].iloc[0]
        assert n_obs >= 2 or np.isnan(df_out.loc[log['group'] == group, 'y']).all()

def test_robust_scale_all_cols(example_df):
    df_out, log = robust_scale_flexible(
        example_df, cols='all', grouping='none'
    )
    # Should robust scale both x and y; non-numeric columns unchanged
    assert all(col in df_out.columns for col in ['x', 'y', 'id', 'date'])
    assert (df_out['id'] == example_df['id']).all()

def test_robust_scale_empty_df():
    df = pd.DataFrame(columns=['x', 'id'])
    df_out, log = robust_scale_flexible(df, cols=['x'], grouping='none')
    assert df_out.empty
    assert log.empty

def test_robust_scale_singleton_group():
    df = pd.DataFrame({'x': [42], 'id': ['A']})
    df_out, log = robust_scale_flexible(df, cols=['x'], grouping='ids', id_cols=['id'])
    assert np.isnan(df_out.loc[0, 'x'])
    assert log.iloc[0]['n_obs'] == 1

def test_robust_scale_invalid_inputs(example_df):
    # Invalid grouping
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], grouping='foo')
    # Invalid column
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['not_in_df'])
    # ids without id_cols
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], grouping='ids')
    # datetime without params
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], grouping='datetime')
    # bad quantile_range (not two values)
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], quantile_range=[25])
    # bad quantile_range (not numeric)
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], quantile_range=["foo", "bar"])
    # quantile_range not ordered
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], quantile_range=[75, 25])
    # quantile_range out of bounds
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], quantile_range=[-5, 75])
    with pytest.raises(ValueError):
        robust_scale_flexible(example_df, cols=['x'], quantile_range=[25, 150])

# Test for quantile_rank_transform_flexible()
@pytest.fixture
def basic_df():
    # Simple test DataFrame, 2 groups of 5 values, with one NaN
    return pd.DataFrame({
        "id": ["A", "A", "A", "B", "B", "B", "B", "B"],
        "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"
        ]),
        "x": [1, 2, 3, 1, 2, 3, 4, np.nan],
        "y": [10, 11, 12, 20, 21, 22, 23, 24]
    })

def test_global_rank(basic_df):
    df_out, log = quantile_rank_transform_flexible(
        basic_df, cols=['x'], mode='rank', grouping='none'
    )
    # All not-null x should be evenly spaced in [0,1] (0, 0.166..., ... 1.0)
    result = df_out['x'].dropna().sort_values().values
    expected = np.linspace(0, 1, 7)  # 7 non-nan values
    assert np.allclose(result, expected)
    # Log output
    assert (log['mode'] == 'rank').all()
    assert set(log['column']) == {'x'}

@pytest.mark.parametrize("mode", ['rank', 'quantile_uniform', 'quantile_normal'])
def test_ids_grouping_modes(basic_df, mode):
    df_out, log = quantile_rank_transform_flexible(
        basic_df, cols=['x'], mode=mode, grouping='ids', id_cols=['id']
    )
    # Each id group x is [1,2,3] for 'A' and [1,2,3,4,np.nan] for 'B'
    # For group A: [1,2,3] should map to [0,0.5,1] in rank
    a_vals = df_out[basic_df['id'] == 'A']['x'].values
    if mode in ('rank', 'quantile_uniform'):
        assert np.allclose(a_vals, [0.0, 0.5, 1.0])
    elif mode == 'quantile_normal':
        # Normalized rank for [0, 0.5, 1] (using norm.ppf, with eps)
        from my_module import _NORM_EPS
        expected = norm.ppf(np.clip([0, 0.5, 1], _NORM_EPS, 1 - _NORM_EPS))
        assert np.allclose(a_vals, expected)
    # Log output contains two groups
    assert log['group'].nunique() == 2

def test_datetime_grouping(basic_df):
    # Rolling block of 3 by date (forces 3 groups: first 3, next 3, last 2)
    df = basic_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df_out, log = quantile_rank_transform_flexible(
        df, cols=['y'], mode='rank', grouping='datetime',
        date_col='date', n_datetime_units=3
    )
    # Block assignments
    expected_groups = ['0','0','0','1','1','1','2','2']
    # All groups with less than 2 valid values will be NaN
    assert set(log['group']) == {'0','1','2'}
    # Each block: y should be mapped within that block
    for group in ['0','1','2']:
        group_rows = log[log['group'] == group]['column'].values
        assert all(col == 'y' for col in group_rows)

def test_all_cols(basic_df):
    # Test 'all' on a DataFrame with numeric and non-numeric cols
    df_out, log = quantile_rank_transform_flexible(
        basic_df, cols='all', mode='rank', grouping='none'
    )
    # Only x and y should be transformed (both numeric)
    for col in ['x','y']:
        assert set(df_out[col].dropna().unique()).issubset([0.0, 0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333, 1.0])
    # Non-numeric columns untouched
    for col in ['id']:
        assert (df_out[col] == basic_df[col]).all()

def test_invalid_inputs(basic_df):
    # Invalid grouping
    with pytest.raises(ValueError):
        quantile_rank_transform_flexible(basic_df, cols=['x'], grouping='foo')
    # Invalid mode
    with pytest.raises(ValueError):
        quantile_rank_transform_flexible(basic_df, cols=['x'], mode='foo')
    # ids without id_cols
    with pytest.raises(ValueError):
        quantile_rank_transform_flexible(basic_df, cols=['x'], grouping='ids')
    # datetime without params
    with pytest.raises(ValueError):
        quantile_rank_transform_flexible(basic_df, cols=['x'], grouping='datetime')
    # missing column
    with pytest.raises(ValueError):
        quantile_rank_transform_flexible(basic_df, cols=['not_in_df'])

def test_one_element_group():
    # Should return NaN for single-element group
    df = pd.DataFrame({'x': [42], 'id': ['A']})
    df_out, log = quantile_rank_transform_flexible(df, cols=['x'], grouping='ids', id_cols=['id'])
    assert np.isnan(df_out.loc[0, 'x'])
    assert log.loc[0, 'n_obs'] == 1

def test_empty_df():
    # Should not fail on empty DataFrame
    df = pd.DataFrame(columns=['x', 'id'])
    df_out, log = quantile_rank_transform_flexible(df, cols=['x'], grouping='none')
    assert df_out.empty
    assert log.empty

# Test for custom_apply_flexible() with winsorize() as a reference
@pytest.fixture
def example_df():
    return pd.DataFrame({
        "id": ["A", "A", "B", "B", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02", "2021-01-03"]),
        "x": [1, 100, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "cat": ["u", "u", "v", "v", "v"]
    })

def test_custom_apply_none_grouping_identity(example_df):
    # Identity function should return input unchanged
    out = custom_apply_flexible(example_df, cols=['x'], func=lambda s: s, grouping='none')
    assert np.allclose(out['x'], example_df['x'])
    # All other columns unchanged
    assert (out['y'] == example_df['y']).all()
    assert (out['id'] == example_df['id']).all()

def test_custom_apply_all_numeric_cols(example_df):
    # Multiply everything by 2
    out = custom_apply_flexible(example_df, cols='all', func=lambda s: s * 2)
    assert np.allclose(out['x'], example_df['x'] * 2)
    assert np.allclose(out['y'], example_df['y'] * 2)
    assert (out['cat'] == example_df['cat']).all()  # non-numeric unchanged

def test_custom_apply_ids_grouping(example_df):
    # Center each group (subtract group mean)
    def center(s):
        return s - s.mean()
    out = custom_apply_flexible(example_df, cols=['x'], func=center, grouping='ids', id_cols=['id'])
    # Group "A"
    mask_a = example_df['id'] == 'A'
    expected_a = example_df.loc[mask_a, 'x'] - example_df.loc[mask_a, 'x'].mean()
    assert np.allclose(out.loc[mask_a, 'x'], expected_a)
    # Group "B"
    mask_b = example_df['id'] == 'B'
    expected_b = example_df.loc[mask_b, 'x'] - example_df.loc[mask_b, 'x'].mean()
    assert np.allclose(out.loc[mask_b, 'x'], expected_b)

def test_custom_apply_datetime_grouping(example_df):
    # Add 10 to each value, but process in rolling windows of 2
    def add_ten(s):
        return s + 10
    out = custom_apply_flexible(
        example_df, cols=['y'],
        func=add_ten, grouping='datetime',
        date_col='date', n_datetime_units=2
    )
    assert np.allclose(out['y'], example_df['y'] + 10)

def test_custom_apply_func_kwargs(example_df):
    # Use winsorize with custom kwargs
    out = custom_apply_flexible(
        example_df, cols=['x'],
        func=winsorize,
        func_kwargs={'lower': 0.2, 'upper': 0.8}
    )
    # The result should be clipped at the 20th and 80th percentiles
    low, high = example_df['x'].quantile(0.2), example_df['x'].quantile(0.8)
    expected = example_df['x'].clip(lower=low, upper=high)
    assert np.allclose(out['x'], expected)

def test_custom_apply_empty_df():
    df = pd.DataFrame(columns=['x', 'id'])
    out = custom_apply_flexible(df, cols=['x'], func=lambda s: s)
    assert out.empty

def test_custom_apply_singleton_group():
    df = pd.DataFrame({'x': [10], 'id': ['A']})
    out = custom_apply_flexible(df, cols=['x'], func=lambda s: s + 5, grouping='ids', id_cols=['id'])
    assert out.loc[0, 'x'] == 15

def test_custom_apply_invalid_inputs(example_df):
    # Invalid grouping
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=lambda s: s, grouping='foo')
    # Invalid func (not callable)
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=42)
    # ids without id_cols
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=lambda s: s, grouping='ids')
    # datetime without params
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=lambda s: s, grouping='datetime')
    # missing columns
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['not_in_df'], func=lambda s: s)
    # Function returns wrong type
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=lambda s: 123)
    # Function returns wrong length
    with pytest.raises(ValueError):
        custom_apply_flexible(example_df, cols=['x'], func=lambda s: s.head(1))

def test_custom_apply_func_raises(example_df):
    # Test that function error is wrapped in RuntimeError
    def bad_func(s):
        raise RuntimeError("my test error")
    with pytest.raises(RuntimeError):
        custom_apply_flexible(example_df, cols=['x'], func=bad_func)

# Test for drop_low_variance_columns()
@pytest.mark.parametrize(
    "df,cols,threshold,expected_cols,expected_dropped",
    [
        # Drop constant col, keep variable
        (
            pd.DataFrame({'x': [1, 1, 1], 'y': [1, 2, 3]}),
            None, 1e-8,
            ['y'],
            [True, False]  # x dropped, y kept
        ),
        # No columns dropped if above threshold
        (
            pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
            None, 1e-8,
            ['a', 'b'],
            [False, False]
        ),
        # Drop col with variance just below threshold
        (
            pd.DataFrame({'a': [1, 1, 1+1e-9], 'b': [0, 0, 0]}),
            None, 1e-8,
            ['a'],
            [True, False]
        ),
        # Restrict checked columns (explicit cols list)
        (
            pd.DataFrame({'x': [1, 1, 1], 'y': [9, 9, 9], 'z': [1, 2, 3]}),
            ['y', 'z'], 1e-8,
            ['x', 'z'],  # Only z kept (x not checked, y dropped)
            [True, False]
        ),
    ]
)
def test_drop_low_variance_columns_typical(
    df, cols, threshold, expected_cols, expected_dropped
):
    cleaned, log = drop_low_variance_columns(df, cols=cols, variance_threshold=threshold)
    assert list(cleaned.columns) == expected_cols
    # Log contains dropped status for all checked columns in input order
    assert log['dropped'].tolist() == expected_dropped

def test_drop_low_variance_columns_no_numeric_cols():
    df = pd.DataFrame({'a': ['foo', 'bar'], 'b': ['baz', 'qux']})
    with pytest.raises(ValueError):
        drop_low_variance_columns(df)

def test_drop_low_variance_columns_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        drop_low_variance_columns(df)

def test_drop_low_variance_columns_invalid_input_type():
    with pytest.raises(ValueError):
        drop_low_variance_columns("not_a_dataframe")

def test_drop_low_variance_columns_invalid_cols_arg():
    df = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(ValueError):
        drop_low_variance_columns(df, cols="not_a_list")
    with pytest.raises(ValueError):
        drop_low_variance_columns(df, cols=[123, 456])  # not strings

def test_drop_low_variance_columns_output_log_structure():
    df = pd.DataFrame({'x': [1, 1, 1], 'y': [1, 2, 3]})
    cleaned, log = drop_low_variance_columns(df)
    assert set(log.columns) == {"column", "variance", "dropped"}
    assert all(isinstance(c, str) for c in log["column"])
    assert log.shape[0] == 2  # one entry per checked column

def test_drop_low_variance_columns_does_not_mutate_input():
    df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 3, 4]})
    df_copy = df.copy(deep=True)
    _ = drop_low_variance_columns(df)
    pd.testing.assert_frame_equal(df, df_copy)

# Test for drop_highly_correlated_columns()
@pytest.mark.parametrize(
    "df,cols,threshold,expected_cols,expected_log_len",
    [
        # x and y are perfectly correlated: y dropped
        (
            pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [2, 4, 8]}),
            None, 0.9,
            ['x', 'z'],  # y is dropped
            1            # One correlated pair: (x,y)
        ),
        # No columns correlated above threshold, keep all
        (
            pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 2], 'c': [7, 8, 9]}),
            None, 0.99,
            ['a', 'b', 'c'],
            0
        ),
        # Restrict checked columns
        (
            pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'other': [1, 1, 2]}),
            ['x', 'y'], 0.8,
            ['other'],  # x and y are both dropped (since they correlate to each other)
            1
        ),
        # Multiple correlated pairs; only one of each pair dropped
        (
            pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 3, 6]}),
            None, 0.8,
            ['a', 'c'],  # b is dropped (correlates highly with a)
            1
        ),
    ]
)
def test_drop_highly_correlated_columns_various_cases(df, cols, threshold, expected_cols, expected_log_len):
    cleaned, log = drop_highly_correlated_columns(df, cols=cols, correlation_threshold=threshold)
    assert list(cleaned.columns) == expected_cols
    assert isinstance(log, pd.DataFrame)
    assert len(log) == expected_log_len

def test_drop_highly_correlated_columns_empty_input():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        drop_highly_correlated_columns(df)

def test_drop_highly_correlated_columns_no_numeric():
    df = pd.DataFrame({'a': ['foo', 'bar'], 'b': ['baz', 'qux']})
    with pytest.raises(ValueError):
        drop_highly_correlated_columns(df)

def test_drop_highly_correlated_columns_invalid_input_type():
    with pytest.raises(ValueError):
        drop_highly_correlated_columns("not_a_dataframe")

def test_drop_highly_correlated_columns_invalid_cols_arg():
    df = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(ValueError):
        drop_highly_correlated_columns(df, cols="not_a_list")
    with pytest.raises(ValueError):
        drop_highly_correlated_columns(df, cols=[123, 456])  # not strings

def test_drop_highly_correlated_columns_log_structure():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    cleaned, log = drop_highly_correlated_columns(df, correlation_threshold=0.9)
    assert set(log.columns) == {"column_1", "column_2", "correlation", "dropped_column"}
    # Log should correctly indicate which col is dropped
    if not log.empty:
        assert all(log['dropped_column'] == log['column_2'])

def test_drop_highly_correlated_columns_does_not_mutate_input():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    df_copy = df.copy(deep=True)
    _ = drop_highly_correlated_columns(df)
    pd.testing.assert_frame_equal(df, df_copy)
