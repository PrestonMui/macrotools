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
    data['LNS12300060'] / 100,
    format_info={
        'title': 'Prime-Age Employment Rate',
        'xlim': ('2019-01', '2025-06'),
        'ylim': (0.76, 0.82),
        'yticksize': 0.01,
        'ydecimals': 0,
        'ytickformat': 'pctg',
    }
)
plt.show()
```

## Features

### Data access

`pull_data(source)` downloads and caches full flat files. `pull_bls_series()` extracts individual series by code.

Supported sources:

| Source | Description |
|--------|-------------|
| `ln` | Household survey (CPS) labor force statistics |
| `ce` | Establishment survey (CES) statistics |
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

`tsgraph()` wraps matplotlib with EA house styling (colors, fonts, layout). It supports:

- Single and multi-series plots
- Dual y-axes
- Customizable axis limits, tick formatting, labels, and legends
- Percentage and decimal tick formats
- Subtitles, shading, and reference lines
- Saving directly to PNG

Pass data as a Series, DataFrame, or dict of series. Formatting is controlled via the `format_info` dict — see `help(mt.tsgraph)` for the full list of options.

### Time series utilities

- **`cagr(data, lag, ma)`** — Compounded annual growth rates with optional moving-average smoothing.
- **`rebase(data, baseperiod, basevalue)`** — Reindex series to a base period (single date or date range).

## Credential setup

**BLS email:** Required for BLS data. Set once with `mt.store_email('you@example.com')` or pass `email=` to `pull_data()`.

**FRED API key:** Required for `alfred_as_reported()`. Register at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html), then `mt.store_fred_api_key('your-key')` or set the `FRED_API_KEY` environment variable.

Credentials are stored in `~/.macrodata_credentials/credentials.json`. Cached data lives in `~/.macrodata_cache/` with a 7-day TTL.

## Examples

See the [example notebook](https://github.com/PrestonMui/macrotools/blob/main/examples/macrotools_guide.ipynb) for detailed usage with output.

## License

[MIT](LICENSE)
