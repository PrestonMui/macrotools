"""
MacroTools
==========

A collection of custom plotting utilities for pulling, organizing, and displaying macroeconomic time series analysis.

Available functions:
- ea_tsgraph: Create time series graphs in EA house style. Options for formatting.
"""

# Register custom font once when package is imported
from pathlib import Path
from matplotlib import font_manager
fontfile = Path(__file__).parent / 'styles' / 'fonts' / 'Montserrat-Regular.ttf'
font_manager.fontManager.addfont(str(fontfile))

from .time_series_graph import (
	tsgraph,
	eacolors
)

from .time_series import (
	cagr,
    rebase,
)

from .pull_data import (
	pull_data,
	pull_bls_series,
	search_bls_series
)

from .storage import (
	clear_macrodata_cache,
	store_email,
	get_stored_email,
)

# Define what gets imported with "from macrotools import *"
__all__ = ['tsgraph', 'eacolors', 'pull_data_full', 'pull_bls_series', 'clear_macrodata_cache', 'store_email', 'get_stored_email']

# Package metadata
__version__ = '0.1.0'
__author__ = 'Preston Mui'
__email__ = 'preston@employamerica.org'
