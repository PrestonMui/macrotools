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
    _preferred_cache_extension,
    _WIDE_BLS_SOURCES,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a fresh temp dir for the duration of the test."""
    monkeypatch.setenv('MACRODATA_CACHE_DIR', str(tmp_path))
    return tmp_path


# ============================================================
# Routing policy: wide BLS sources -> feather, others -> parquet
# ============================================================

def test_routing_wide_bls_pivoted_uses_feather():
    # Every wide BLS source's pivoted frequencies route to feather
    for source in _WIDE_BLS_SOURCES:
        for freq in ('M', 'Q', 'A', 'S'):
            assert _preferred_cache_extension(source, freq) == '.feather', (source, freq)
        # And freq='all' is parquet even for BLS sources (long format, narrow)
        assert _preferred_cache_extension(source, 'all') == '.parquet', source


def test_routing_non_bls_uses_parquet():
    # Non-BLS sources route to parquet regardless of freq key
    for source in ['nipa-pce', 'ny-mfg', 'philly-mfg', 'stclaims', 'kc-svc']:
        for freq in ('default', 'M', 'Q', 'all'):
            assert _preferred_cache_extension(source, freq) == '.parquet', (source, freq)


# ============================================================
# Wide BLS pivoted source -> feather (with index.freq preserved)
# ============================================================

def test_wide_bls_pivoted_roundtrip_as_feather(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=12, freq='MS', name='date')
    df = pd.DataFrame({'a': np.arange(12, dtype=float), 'b': np.arange(12, 24, dtype=float)}, index=idx)

    base = _get_cache_file_path('ln', 'M')
    _save_cache_file(df, base)

    assert base.with_suffix('.feather').exists()
    assert not base.with_suffix('.parquet').exists()

    loaded = _load_cache_file(base)
    pd.testing.assert_frame_equal(loaded, df)
    assert loaded.index.freq is not None
    assert loaded.index.freq.freqstr == 'MS'


# ============================================================
# Non-BLS source -> parquet, even with single-level columns
# ============================================================

def test_non_bls_single_level_roundtrip_as_parquet(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=12, freq='MS', name='date')
    df = pd.DataFrame({'a': np.arange(12, dtype=float), 'b': np.arange(12, 24, dtype=float)}, index=idx)

    base = _get_cache_file_path('ny-mfg', 'default')
    _save_cache_file(df, base)

    assert base.with_suffix('.parquet').exists()
    assert not base.with_suffix('.feather').exists()

    loaded = _load_cache_file(base)
    pd.testing.assert_frame_equal(loaded, df)
    assert loaded.index.freq is not None
    assert loaded.index.freq.freqstr == 'MS'


# ============================================================
# MultiIndex columns override to parquet even on a BLS source path
# ============================================================

def test_multiindex_columns_force_parquet(isolated_cache):
    idx = pd.date_range('2020-01-01', periods=6, freq='MS', name='date')
    cols = pd.MultiIndex.from_tuples(
        [(1, 'price'), (1, 'quantity'), (35, 'price'), (35, 'quantity')],
        names=['line', 'datatype'],
    )
    df = pd.DataFrame(np.random.rand(6, 4), index=idx, columns=cols)

    # Even if the caller hands us a wide-BLS path, MultiIndex overrides to parquet
    base = _get_cache_file_path('ln', 'M')

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
    assert pd.api.types.is_integer_dtype(loaded.columns.get_level_values('line'))
    assert loaded[(35, 'price')].notna().all()
    pd.testing.assert_frame_equal(loaded, df)


# ============================================================
# Migration: old .feather (e.g. from 0.3.2) gets replaced
# ============================================================

def test_resave_swaps_feather_for_parquet_on_multiindex(isolated_cache):
    """An existing feather cache is replaced by parquet when the data turns MultiIndex."""
    idx = pd.date_range('2020-01-01', periods=4, freq='MS', name='date')
    base = _get_cache_file_path('ln', 'M')

    single = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    _save_cache_file(single, base)
    assert base.with_suffix('.feather').exists()

    cols = pd.MultiIndex.from_tuples([(1, 'x'), (2, 'y')], names=['line', 'kind'])
    multi = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], index=idx, columns=cols)
    _save_cache_file(multi, base)

    assert base.with_suffix('.parquet').exists()
    assert not base.with_suffix('.feather').exists(), "Stale .feather should be removed after parquet resave"

    resolved = _resolve_cache_file('ln', 'M')
    assert resolved == base.with_suffix('.parquet')


def test_resave_swaps_feather_for_parquet_on_routing_change(isolated_cache):
    """A pre-routing-swap feather cache for a non-BLS source migrates to parquet."""
    idx = pd.date_range('2020-01-01', periods=4, freq='MS', name='date')
    cache_dir = isolated_cache
    # Simulate an old .feather sitting in the cache dir from before 0.3.3 routing
    legacy_path = cache_dir / 'ny-mfg_default.feather'
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    df.reset_index().to_feather(legacy_path)
    (cache_dir / 'ny-mfg_default.meta.json').write_text('{"__index_name__": "date"}')

    # _resolve_cache_file finds it
    assert _resolve_cache_file('ny-mfg', 'default') == legacy_path

    # Resaving via the cache layer writes parquet and removes the feather
    base = _get_cache_file_path('ny-mfg', 'default')
    _save_cache_file(df, base)
    assert base.with_suffix('.parquet').exists()
    assert not legacy_path.exists()


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
    assert pd.api.types.is_integer_dtype(cached.columns.get_level_values('line'))

    assert _resolve_cache_file('nipa-pce', 'default').suffix == '.parquet'
