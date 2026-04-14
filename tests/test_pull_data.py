import pytest
import pandas as pd
import macrotools as mt


# ============================================================
# get_series_list() tests
# ============================================================

VALID_SOURCES = ['ln', 'ce', 'jt', 'ci', 'cu', 'pc', 'wp', 'ei', 'cx', 'tu']


def test_get_series_list_returns_dataframe():
    result = mt.get_series_list('ln')
    assert isinstance(result, pd.DataFrame)
    assert 'series_id' in result.columns
    assert 'description' in result.columns


@pytest.mark.parametrize("source", VALID_SOURCES)
def test_get_series_list_valid_sources(source):
    result = mt.get_series_list(source)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_get_series_list_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        mt.get_series_list('fake_source')


def test_get_series_list_nonempty():
    result = mt.get_series_list('ln')
    assert len(result) > 100  # household survey has thousands of series


# ============================================================
# search_bls_series() tests
# ============================================================

def test_search_returns_dataframe():
    result = mt.search_bls_series('ln', 'unemployment rate')
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'series_id', 'description', 'score'}


def test_search_respects_n():
    result = mt.search_bls_series('ln', 'unemployment rate', n=3)
    assert len(result) <= 3


def test_search_exact_match_ranks_high():
    result = mt.search_bls_series('ln', 'labor force level 25 54')
    assert len(result) > 0
    # Top result should contain relevant terms
    top_desc = result.iloc[0]['description'].lower()
    assert 'labor force' in top_desc


def test_search_sa_filter():
    result = mt.search_bls_series('ln', 'unemployment', sa=True)
    if len(result) > 0:
        # All series should have 'S' as 3rd character (seasonally adjusted)
        assert all(sid[2] == 'S' for sid in result['series_id'])


def test_search_sa_filter_unadjusted():
    result = mt.search_bls_series('ln', 'unemployment', sa=False)
    if len(result) > 0:
        assert all(sid[2] == 'U' for sid in result['series_id'])


def test_search_no_results():
    result = mt.search_bls_series('ln', 'zzxxqqjjkk')
    assert len(result) == 0


def test_search_field_id():
    result = mt.search_bls_series('cu', 'SA0', field='id')
    assert len(result) > 0
    # Results should contain SA0 in the series_id
    assert all('SA0' in sid for sid in result['series_id'])


# ============================================================
# pull_data() tests
# ============================================================

@pytest.fixture(scope="module")
def ln_data():
    """Pull household survey data once for all pull_data tests."""
    return mt.pull_data('ln')


def test_pull_data_returns_dataframe(ln_data):
    assert isinstance(ln_data, pd.DataFrame)


def test_pull_data_has_datetime_index(ln_data):
    assert isinstance(ln_data.index, pd.DatetimeIndex)
    assert ln_data.index.name == 'date'


def test_pull_data_has_series_columns(ln_data):
    # LN source should have thousands of series
    assert ln_data.shape[1] > 1000
    # Columns should be LN series IDs
    assert all(col.startswith('LN') for col in ln_data.columns[:10])


def test_pull_data_values_are_numeric(ln_data):
    assert ln_data.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d)).all()


def test_pull_data_date_range_is_reasonable(ln_data):
    # Household survey data starts in 1948
    assert ln_data.index.min().year <= 1950
    assert ln_data.index.max().year >= 2024


def test_pull_data_columns_filter():
    cols = ['LNS12000000', 'LNS14000000']
    data = mt.pull_data('ln', columns=cols)
    assert list(data.columns) == cols
    assert isinstance(data.index, pd.DatetimeIndex)


def test_pull_data_freq_all():
    data = mt.pull_data('ln', freq='all')
    assert isinstance(data, pd.DataFrame)
    assert 'series_id' in data.columns
    assert 'value' in data.columns
    assert 'period' in data.columns
    assert 'frequency' in data.columns
    # period should be int (e.g. M01 -> 1, M13 -> 13)
    assert data['period'].dtype in ('int64', 'int32', int)
    assert 13 in data['period'].values
    # month and quarter columns should not be present
    assert 'month' not in data.columns
    assert 'quarter' not in data.columns
    # Should contain multiple frequencies
    freqs = set(data['frequency'].unique())
    assert 'M' in freqs


def test_pull_data_freq_quarterly():
    data = mt.pull_data('ln', freq='Q')
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.name == 'date'
    # All columns should be quarterly LN series
    assert data.shape[1] > 0
    assert all(col.startswith('LN') for col in data.columns[:10])


def test_pull_data_freq_annual():
    data = mt.pull_data('ln', freq='A')
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.name == 'date'
    assert data.shape[1] > 0


def test_pull_data_freq_all_columns_filter():
    cols = ['LNS12000000', 'LNS14000000']
    data = mt.pull_data('ln', freq='all', columns=cols)
    assert set(data['series_id'].unique()) == set(cols)


def test_pull_data_invalid_freq():
    with pytest.raises(ValueError, match="Invalid freq"):
        mt.pull_data('ln', freq='X')


def test_pull_data_empty_freq():
    with pytest.raises(ValueError, match="No 'S' frequency data"):
        mt.pull_data('ln', freq='S')


# ============================================================
# pull_bls_series() tests
# ============================================================

def test_pull_bls_series_quarterly():
    data = mt.pull_bls_series('LNS11000000Q')
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.name == 'date'
    assert 'LNS11000000Q' in data.columns


def test_pull_bls_series_mixed_freq_error():
    with pytest.raises(ValueError, match="Mixed frequencies"):
        mt.pull_bls_series(['LNS12000000', 'LNS11000000Q'])


# ============================================================
# Input validation tests (no network calls)
# ============================================================

def test_pull_data_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        mt.pull_data('not_a_real_source')
