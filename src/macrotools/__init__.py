"""
MacroTools
==========

A collection of custom plotting utilities for pulling, organizing, and displaying macroeconomic time series analysis.

Available functions:
- tsgraph: Create time series graphs with customizable styles.
"""

from .time_series_graph import (
	tsgraph,
	default_colors,
	default_alert_colors,
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
__all__ = ['tsgraph', 'default_colors', 'default_alert_colors', 'eacolors', 'ea_alert_colors', 'pull_data', 'pull_bls_series', 'search_bls_series', 'alfred_as_reported', 'clear_macrodata_cache', 'store_email', 'get_stored_email', 'store_fred_api_key', 'get_stored_fred_api_key']

# Package metadata
__version__ = '0.1.7'
__author__ = 'Preston Mui'
__email__ = 'preston@employamerica.org'
