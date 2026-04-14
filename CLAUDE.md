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

## Current Context
- Preparing for v1.0 release