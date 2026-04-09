import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@pytest.fixture(scope="session")
def sample_data():
    """Load household survey data once for all tests."""
    return pd.read_pickle('examples/lndata.pkl')


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')
