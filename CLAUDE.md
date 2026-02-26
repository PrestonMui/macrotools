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

## Current Context
- Preparing for v1.0 release
- Want to move away from making EA house style the default style of tsgraph (colors, Montserrat font, off-white background `#F9F7F5`).
- Want to move towards creating a different default graph style, but allow for selecting ea style using `style = 'ea'`; default style should be default and generic.
- Package currently lacks tests and documentation