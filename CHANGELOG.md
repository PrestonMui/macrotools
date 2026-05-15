# Changelog

## Version 0

### Version 0.3.3
- Cache layer now writes MultiIndex-column DataFrames to parquet instead of feather, preserving level dtypes (e.g. integer `line` codes on `nipa-pce`). Single-level column DataFrames continue to use feather. Existing caches will be transparently upgraded on next refresh.

### Version 0.3.2
- Added `'la'` source: BLS Local Area Unemployment Statistics (LAUS) — 33,985 series covering states, metros, counties, and cities. Pulls all 9 BLS flat files (8 NSA buckets + 1 SA file) and concatenates into one DataFrame
- LAUS works with `pull_data('la')`, `pull_bls_series('LAUST...')` (both `flatfiles` and `api` source modes), `get_series_list('la')`, and `search_bls_series('la', ...)`
- `pull_bls_series()` now sets `index.freq` on the returned DataFrame so downstream functions like `cagr()` work without a manual `.asfreq()` call
- Refactored frequency-to-offset mapping into a single shared helper used across `pull_data` flatfiles, `pull_bls_series` flatfiles, and `pull_bls_series` api branches
- Bug fix: `pull_data('philly-mfg')`, `pull_data('dallas-mfg')`, and `pull_data('dallas-retail')` no longer raise `ValueError: Cannot specify both 'axis' and 'index'/'columns'` on newer pandas

### Version 0.3.1
- Added `freq` parameter to `pull_data()` — supports `'M'`, `'Q'`, `'A'`, `'S'`, and `'all'` (unpivoted long-format)
- Added `columns` parameter to `pull_data()` to filter specific series
- `pull_bls_series()` now supports quarterly, annual, and semiannual series with auto-frequency detection; raises `ValueError` on mixed frequencies
- Switched default cache format from pickle to feather (parquet for `freq='all'`); existing caches will be re-pulled automatically
- Streamlined credential storage: replaced per-credential functions (`store_email`, `store_fred_api_key`, etc.) with generic `store_credential(name, value)` and `get_stored_credential(name)` **(breaking change)**
- Added environment variable support for email (`MACROTOOLS_EMAIL`)
- Credential storage prompt now tells users where credentials are saved and that they are stored as plain text
- Set file permissions (0o600) on credentials file on Unix systems
- Updated README with full documentation of all features

### Version 0.3.0
- Major refactor of `tsgraph()` for readability; series input now uses a unified parameter format (see docstring)
- Added recession shading via `xaxis={'shading': 'nber'}` or custom date ranges
- Added data callouts: annotate specific data points with markers and text labels
- Added horizontal line option `hline` to `tsgraph()`
- Added `source='api'` option to `pull_bls_series()` for faster small pulls
- Improved `search_bls_series()` to use fuzzy matching instead of exact substring matching
- Added test infrastructure and visual gallery

### Version 0.2.1
- Bug fix: `tsgraph()` now works with `xdata` when `xlim` is not set
- Added `get_series_list()` function to look up BLS series IDs and descriptions
- Shipped static series catalog CSVs for BLS sources (`ln`, `ce`, `ci`, `jt`, `cu`, `pc`, `wp`, `ei`, `cx`, `tu`)
- Series information is no longer attached as `.attrs` on DataFrames returned by `pull_data()`
- Changed default logo size from 0.06 to 0.07

### Version 0.2.0
- Added new default graph style with Lato font; EA house style now available via `style='ea'`
- Added `footnote` option to `tsgraph()` for adding footnotes to graphs
- Added `logo` option to `tsgraph()` for including a logo in the bottom-right corner
- Added two new colors to the default color palette
- Various style and layout improvements

### Version 0.1.7
- Added 'bgcolor' as format_info entry to change background color of tsgraph()

### Version 0.1.5
- Fixed url on Richmond Nonmfg survey

### Version 0.1.5
- Added function to generate as-reported series. See help(macrotools.alfred_as_reported) for more.

### Version 0.1.4
- Added time use survey ('tu') to pull_data
- Various bug fixes and improvements

### Version 0.1.3
- Added Consumer Expenditures ('cx') to pull_data and pull_bls_series.
- Removed `freq` as option for pull_data.
- Added function search_bls_series.

### Version 0.1.2
- Re-added ability to pull state-level claims data.
- Changed tsgraph function to no longer require an xdata entry, and to look at the index of ydata for x data points.

### Version 0.1.1

- Removed state-level claims pulling for now
- Changed pull_bls_series to use pull_data instead of the API. Note the new syntax.
- Some reorganization of the code
- `pull_data_full` is now `pull_data

### Version 0.1.0

Initial commit.