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

from .timeseries import (
	tsgraph,
	cagr,
    rebase,
	eacolors
)

from .pulldata import (
	pull_data_full,
	pull_bls_series
)

# Define what gets imported with "from macrotools import *"
__all__ = ['tsgraph', 'eacolors', 'pull_flat_file']

# Package metadata
__version__ = '0.1.0'
__author__ = 'Preston Mui'
__email__ = 'preston@employamerica.org'
