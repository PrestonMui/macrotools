# MacroTools

A Python package providing flexible tools to work with macroeconomic data and create Employ America-style time series graphs.

## Features

- Create professional time series graphs with matplotlib in EA style
- Support for dual y-axes for comparing different data series
- Flexible formatting options
- Easy integration with existing matplotlib workflows
- Download BLS Series

## Installation

`pip install git+https://github.com/PrestonMui/macrotools.git`

## Examples

See [this notebook](https://github.com/PrestonMui/macrotools/blob/main/examples/macrotools_guide.ipynb) for examples on how to use Macrotools

## Roadmap

Some features I am working on.

- Store email, API keys etc in a settings file; eliminate need to enter email with every BLS pull
- BLS series pull -- allow for > 10 years data pulling at once
- Wrapper for FRED API -- allow for pulling multiple series 
- Upload to PyPi