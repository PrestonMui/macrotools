# Changelog

## Version 0

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