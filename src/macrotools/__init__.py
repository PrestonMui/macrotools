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
fontfile_bold = Path(__file__).parent / 'styles' / 'fonts' / 'Lato-Bold.ttf'
font_manager.fontManager.addfont(str(fontfile))
font_manager.fontManager.addfont(str(fontfile_bold))

from .time_series_graph import (
	tsgraph,
	eacolors,
	ea_alert_colors
)

from .time_series import (
	cagr,
    rebase,
)

from .pull_data import (
	pull_data,
	pull_bls_series,
	search_bls_series,
	alfred_as_reported
)

from .storage import (
	clear_macrodata_cache,
	store_email,
	get_stored_email,
	store_fred_api_key,
	get_stored_fred_api_key,
)

# Define what gets imported with "from macrotools import *"
__all__ = ['tsgraph', 'eacolors', 'ea_alert_colors', 'pull_data', 'pull_bls_series', 'search_bls_series', 'alfred_as_reported', 'clear_macrodata_cache', 'store_email', 'get_stored_email', 'store_fred_api_key', 'get_stored_fred_api_key']

# Package metadata
__version__ = '0.1.7'
__author__ = 'Preston Mui'
__email__ = 'preston@employamerica.org'
