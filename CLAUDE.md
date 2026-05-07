# MacroTools — Claude Instructions

## Project overview
MacroTools is a public Python package (PyPI: `macrotools`) for pulling, caching, and graphing U.S. macroeconomic data. It is built and maintained by Employ America.

## Key files
- `src/macrotools/time_series_graph.py` — Main graphing function `tsgraph()`
- `src/macrotools/pull_data.py` — Data pulling functions (BLS, BEA, Fed surveys, FRED/ALFRED)
- `src/macrotools/time_series.py` — Time series utilities (`cagr`, `rebase`)
- `src/macrotools/storage.py` — Credential and cache storage
- `src/macrotools/styles/eastyle.mplstyle` — Employ America house style (colors, fonts, layout)
- `src/macrotools/styles/fonts/` — Montserrat font files
- `src/macrotools/__init__.py` — Package exports
- `pyproject.toml` — Build config and dependencies

## Workflow preferences
- Always read and understand the relevant source files before suggesting or making changes.
- Plan, explain and refine before suggesting changes.
- Prefer editing existing files over creating new ones.

## Known issues
- `pull_data('cu', freq='S')` breaks: CU semiannual data has an `S03` period (annual average) that the `freq='S'` pivot code doesn't handle — it computes month 13, which is invalid. Need to drop `S03` rows (like `M13` is dropped for monthly).

## To do
- Add vertical line annotation support to `tsgraph()` (similar to `hline` but for marking specific dates/events)
- Add Laspeyres, Paasche, and Fisher price index calculation functions to `time_series.py`
- **Census EITS support** — partially implemented on branch `feature/census-eits` (commit a57c5cb). Adds `pull_data('census-eits-XXX')` for the 21 Census Economic Indicators programs (mrts, qss, resconst, etc.). Monthly + quarterly basic pulls work; cache, attrs round-trip, and bad-program errors verified. Two known issues to finish before merge: (1) `columns=` filter only works on the cached-load path, not on the fresh-pull path (need to apply the filter post-pivot before returning); (2) `.index.freq` is not set, so `cagr()` raises `AttributeError` on EITS DataFrames — needs `.asfreq('MS')` for monthly programs, `.asfreq('QS')` for quarterly. Plan file: `~/.claude/plans/linear-foraging-aurora.md`. Note: `qtax` requires a `for=` geography arg the current code doesn't pass — will 400 if anyone calls it.

## Current Context
- Preparing for v1.0 release