import pytest
import pandas as pd
import numpy as np
import macrotools as mt


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def monthly_series():
    """Simple monthly series: 10% annual growth over 2 years."""
    dates = pd.date_range('2020-01', periods=24, freq='MS')
    values = 100 * (1.1 ** (np.arange(24) / 12))
    return pd.Series(values, index=dates)


@pytest.fixture
def quarterly_series():
    dates = pd.date_range('2020-01', periods=8, freq='QS')
    values = 100 * (1.1 ** (np.arange(8) / 4))
    return pd.Series(values, index=dates)


# ============================================================
# cagr() tests
# ============================================================

def test_cagr_monthly(monthly_series):
    result = mt.cagr(monthly_series, lag=1)
    assert isinstance(result, pd.Series)
    assert len(result) == len(monthly_series)
    # First value should be NaN (no prior period)
    assert pd.isna(result.iloc[0])
    # Remaining values should be finite
    assert result.iloc[1:].notna().all()


def test_cagr_monthly_lag12(monthly_series):
    result = mt.cagr(monthly_series, lag=12)
    # Only last value should be non-NaN (need 12 periods of lag)
    assert pd.isna(result.iloc[0])
    assert result.iloc[-1] == pytest.approx(0.1, abs=0.01)


def test_cagr_with_ma(monthly_series):
    result = mt.cagr(monthly_series, lag=1, ma=3)
    assert isinstance(result, pd.Series)
    # First few values should be NaN due to rolling window
    assert pd.isna(result.iloc[0])


def test_cagr_quarterly(quarterly_series):
    result = mt.cagr(quarterly_series, lag=1)
    assert isinstance(result, pd.Series)
    assert result.iloc[1:].notna().all()


def test_cagr_dataframe(monthly_series):
    df = pd.DataFrame({'a': monthly_series, 'b': monthly_series * 2})
    result = mt.cagr(df, lag=1)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['a', 'b']


def test_cagr_rejects_non_dataframe():
    with pytest.raises(TypeError):
        mt.cagr([1, 2, 3])


def test_cagr_rejects_non_datetime_index():
    s = pd.Series([100, 110, 121], index=[0, 1, 2])
    with pytest.raises(TypeError):
        mt.cagr(s)


# ============================================================
# rebase() tests
# ============================================================

def test_rebase_single_period(monthly_series):
    result = mt.rebase(monthly_series, '2020-01')
    assert result.loc['2020-01-01'] == pytest.approx(100.0)


def test_rebase_range(monthly_series):
    result = mt.rebase(monthly_series, ('2020-01', '2020-03'))
    # Average of first 3 months should be 100
    base_avg = result.loc['2020-01':'2020-03'].mean()
    assert base_avg == pytest.approx(100.0, abs=0.1)


def test_rebase_custom_value(monthly_series):
    result = mt.rebase(monthly_series, '2020-01', basevalue=1)
    assert result.loc['2020-01-01'] == pytest.approx(1.0)


def test_rebase_preserves_shape(monthly_series):
    result = mt.rebase(monthly_series, '2020-01')
    assert len(result) == len(monthly_series)


def test_rebase_dataframe(monthly_series):
    df = pd.DataFrame({'a': monthly_series, 'b': monthly_series * 2})
    result = mt.rebase(df, '2020-01')
    assert isinstance(result, pd.DataFrame)
    assert result.loc['2020-01-01', 'a'] == pytest.approx(100.0)
    assert result.loc['2020-01-01', 'b'] == pytest.approx(100.0)


def test_rebase_rejects_non_dataframe():
    with pytest.raises(TypeError):
        mt.rebase([1, 2, 3], '2020-01')


def test_rebase_rejects_non_datetime_index():
    s = pd.Series([100, 110], index=[0, 1])
    with pytest.raises(TypeError):
        mt.rebase(s, 0)
