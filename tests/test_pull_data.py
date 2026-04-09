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
# Input validation tests (no network calls)
# ============================================================

def test_pull_data_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        mt.pull_data('not_a_real_source')
