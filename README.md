# MacroTools

A Python package for pulling, caching, and graphing U.S. macroeconomic data. Built at [Employ America](https://employamerica.org).

MacroTools makes it easy to download flat files from the BLS, BEA, and regional Federal Reserve surveys, and to produce publication-ready time series charts with a consistent visual style.

## Installation

```bash
pip install macrotools
```

For [FRED/ALFRED](https://fred.stlouisfed.org/) support:

```bash
pip install macrotools[fred]
```

## Quick start

```python
import macrotools as mt
import matplotlib.pyplot as plt

# Pull household survey data (cached automatically for 7 days)
data = mt.pull_data('ln')

# Plot prime-age employment rate
fig = mt.tsgraph(
    series=data['LNS12300060'] / 100,
    xaxis={'lim': ('2019-01', '2025-06')},
    yaxis={'lim': (0.76, 0.82), 'ticksize': 0.01, 'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Prime-Age Employment Rate'},
)
plt.show()
```

## Features

### Data access

**`pull_data(source, freq, columns)`** downloads and caches full flat files from BLS and other sources. BLS sources support a `freq` parameter (`'M'`, `'Q'`, `'A'`, `'S'`, or `'all'` for unpivoted long-format data) and a `columns` parameter to select specific series.

```python
# Pull monthly household survey data
data = mt.pull_data('ln')

# Pull only specific series
data = mt.pull_data('ln', columns=['LNS12000000', 'LNS14000000'])

# Pull quarterly data
data = mt.pull_data('ln', freq='Q')

# Pull raw long-format data with all frequencies
data = mt.pull_data('ln', freq='all')
```

**`pull_bls_series(series_list)`** extracts individual series by code, auto-detecting the frequency from the series IDs. Supports monthly, quarterly, annual, and semiannual series. All series must share the same frequency.

```python
data = mt.pull_bls_series(['LNS12000000', 'CES0000000001'])
```

**`search_bls_series(source, query)`** fuzzy-searches BLS series catalogs to find series IDs.

```python
mt.search_bls_series('ln', 'prime age employment')
mt.search_bls_series('cu', 'shelter', sa=True)  # seasonally adjusted only
```

**`get_series_list(source)`** returns the full series catalog for a BLS source as a DataFrame.

Supported sources:

| Source | Description |
|--------|-------------|
| `ln` | Household survey (CPS) labor force statistics |
| `ce` | Establishment survey (CES) statistics |
| `ci` | Employment Cost Index (ECI) |
| `jt` | Job Openings and Labor Turnover Survey (JOLTS) |
| `cu` | CPI — All Urban Consumers |
| `pc` | PPI — Industry Data |
| `wp` | PPI — Commodity Data |
| `ei` | Import/Export Price Indices |
| `cx` | Consumer Expenditures |
| `tu` | Time Use Survey |
| `nipa-pce` | NIPA Personal Consumption Expenditures |
| `stclaims` | State-level unemployment claims |
| `ny-mfg`, `ny-svc` | NY Fed Empire Manufacturing & Services |
| `philly-mfg`, `philly-nonmfg` | Philadelphia Fed surveys |
| `richmond-mfg`, `richmond-nonmfg` | Richmond Fed surveys |
| `dallas-mfg`, `dallas-svc`, `dallas-retail` | Dallas Fed surveys |
| `kc-mfg`, `kc-svc` | Kansas City Fed surveys |

BLS sources require an email address. MacroTools will prompt you the first time and store it locally in `~/.macrodata_credentials/`.

With the optional `fredapi` dependency, `alfred_as_reported()` pulls historical vintage data from ALFRED.

### Graphing

`tsgraph()` wraps matplotlib with EA house styling (colors, fonts, layout). Pass data as a Series, DataFrame, or list of dicts. Formatting is controlled via separate `xaxis`, `yaxis`, `title`, `legend`, and `footnote` dicts — see `help(mt.tsgraph)` for the full list of options.

Features:

- Single and multi-series plots
- Dual y-axes (`axis='right'` on individual series)
- Customizable axis limits, tick formatting, labels, and legends
- Percentage and decimal tick formats
- Titles, subtitles, and footnotes
- NBER recession shading or custom shading regions
- Horizontal reference lines with callouts (`hline`)
- Data callouts on individual series (annotate specific points with values)
- Saving directly to file (`save_path`)

### Time series utilities

- **`cagr(data, lag, ma)`** — Compounded annual growth rates with optional moving-average smoothing.
- **`rebase(data, baseperiod, basevalue)`** — Reindex series to a base period (single date or date range).

### Cache management

Data is cached locally in `~/.macrodata_cache/` with a 7-day TTL. Use `mt.clear_macrodata_cache()` to clear all cached data, or `mt.clear_macrodata_cache(source)` to clear a specific source.

## Credential setup

**BLS email:** Required for BLS flat-file data pulls. Set once with `mt.store_credential('email', 'you@example.com')`, pass `email=` to `pull_data()`, or set the `MACROTOOLS_EMAIL` environment variable.

**BLS API key:** Required for `pull_bls_series(source='api')`. Register at [BLS](https://data.bls.gov/registrationEngine/), then `mt.store_credential('bls_api_key', 'your-key')` or set the `BLS_API_KEY` environment variable.

**FRED API key:** Required for `alfred_as_reported()`. Register at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html), then `mt.store_credential('fred_api_key', 'your-key')` or set the `FRED_API_KEY` environment variable.

Credentials are resolved in order: function argument > stored file > environment variable > interactive prompt. Credentials are stored as plain text in `~/.macrodata_credentials/credentials.json`. For sensitive keys, prefer environment variables instead of storing to disk.

## Examples

See the [example notebook](https://github.com/PrestonMui/macrotools/blob/main/examples/macrotools_guide.ipynb) for detailed usage with output.

## License

[MIT](LICENSE)
