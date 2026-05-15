import os
import warnings
import pytest
import pandas as pd
import numpy as np

import macrotools as mt
from macrotools.storage import (
    _save_cache_file,
    _load_cache_file,
    _resolve_cache_file,
    _get_cache_file_path,
    _save_cached_data,
    _load_cached_data,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a fresh temp dir for the duration of the test."""
    monkeypatch.setenv('MACRODATA_CACHE_DIR', str(tmp_path))
    return tmp_path


# ============================================================
# Single-level Index columns -> feather (backwards compat)
# ============================================================

def test_single_level_columns_roundtrip_as_feather(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=12, freq='MS', name='date')
    df = pd.DataFrame({'a': np.arange(12, dtype=float), 'b': np.arange(12, 24, dtype=float)}, index=idx)
    df.index.name = 'date'

    base = _get_cache_file_path('unit_test_single', 'M')
    _save_cache_file(df, base)

    assert base.with_suffix('.feather').exists()
    assert not base.with_suffix('.parquet').exists()

    loaded = _load_cache_file(base)
    pd.testing.assert_frame_equal(loaded, df)
    assert loaded.index.freq is not None
    assert loaded.index.freq.freqstr == 'MS'


# ============================================================
# MultiIndex columns -> parquet, dtypes preserved, no warning
# ============================================================

def test_multiindex_columns_roundtrip_as_parquet(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=6, freq='MS', name='date')
    cols = pd.MultiIndex.from_tuples(
        [(1, 'price'), (1, 'quantity'), (35, 'price'), (35, 'quantity')],
        names=['line', 'datatype'],
    )
    df = pd.DataFrame(np.random.rand(6, 4), index=idx, columns=cols)

    base = _get_cache_file_path('unit_test_multi', 'M')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _save_cache_file(df, base)
        mixed_type_warnings = [
            w for w in caught
            if 'mixed type' in str(w.message).lower() or 'roundtrip' in str(w.message).lower()
        ]
        assert not mixed_type_warnings, f"Unexpected mixed-type warning: {[str(w.message) for w in mixed_type_warnings]}"

    assert base.with_suffix('.parquet').exists()
    assert not base.with_suffix('.feather').exists()

    loaded = _load_cache_file(base)
    assert isinstance(loaded.columns, pd.MultiIndex)
    assert loaded.columns.equals(df.columns)
    # Level dtypes preserved — line is int, datatype is object/str
    assert pd.api.types.is_integer_dtype(loaded.columns.get_level_values('line'))
    # And data is reachable by tuple
    assert loaded[(35, 'price')].notna().all()
    pd.testing.assert_frame_equal(loaded, df)


# ============================================================
# Migration: old .feather gets replaced by .parquet on resave
# ============================================================

def test_resave_swaps_feather_for_parquet(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=4, freq='MS', name='date')
    base = _get_cache_file_path('unit_test_migrate', 'M')

    # Step 1: write single-level-column version as feather
    single = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    _save_cache_file(single, base)
    assert base.with_suffix('.feather').exists()
    assert not base.with_suffix('.parquet').exists()

    # Step 2: resave as MultiIndex — feather should be removed, parquet created
    cols = pd.MultiIndex.from_tuples([(1, 'x'), (2, 'y')], names=['line', 'kind'])
    multi = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], index=idx, columns=cols)
    _save_cache_file(multi, base)

    assert base.with_suffix('.parquet').exists()
    assert not base.with_suffix('.feather').exists(), "Stale .feather should be removed after parquet resave"

    # And _resolve_cache_file finds the parquet version
    resolved = _resolve_cache_file('unit_test_migrate', 'M')
    assert resolved == base.with_suffix('.parquet')

    loaded = _load_cache_file(base)
    assert isinstance(loaded.columns, pd.MultiIndex)
    pd.testing.assert_frame_equal(loaded, multi)


def test_resave_swaps_parquet_for_feather(isolated_cache):
    """The reverse migration — MultiIndex collapses back to single level."""
    idx = pd.date_range('2020-01-01', periods=4, freq='MS', name='date')
    base = _get_cache_file_path('unit_test_migrate_rev', 'M')

    cols = pd.MultiIndex.from_tuples([(1, 'x'), (2, 'y')], names=['line', 'kind'])
    multi = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], index=idx, columns=cols)
    _save_cache_file(multi, base)
    assert base.with_suffix('.parquet').exists()

    single = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    _save_cache_file(single, base)

    assert base.with_suffix('.feather').exists()
    assert not base.with_suffix('.parquet').exists()


# ============================================================
# End-to-end: nipa-pce round-trips MultiIndex through the cache
# (BEA flat file, no API key required — real network call)
# ============================================================

def test_nipa_pce_multiindex_survives_cache(isolated_cache):
    fresh = mt.pull_data('nipa-pce', force_refresh=True)
    cached = mt.pull_data('nipa-pce')

    assert isinstance(fresh.columns, pd.MultiIndex)
    assert isinstance(cached.columns, pd.MultiIndex)
    assert cached.columns.equals(fresh.columns)
    # The line level must remain numeric, not be stringified by feather
    assert pd.api.types.is_integer_dtype(cached.columns.get_level_values('line'))

    # And the cache file on disk is parquet, not feather
    assert _resolve_cache_file('nipa-pce', 'default').suffix == '.parquet'
