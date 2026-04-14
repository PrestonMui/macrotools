"""
Generate series catalog CSV files for each BLS source.

Pulls series files directly from the BLS website
(https://download.bls.gov/pub/time.series/{source}/{source}.series)
and writes them as CSV files to src/macrotools/data/.

Run this script before a release to update the static series catalogs
shipped with the macrotools package.

Usage:
    python scripts/generate_series_lists.py

Requires a valid BLS email address (passed as argument or stored via
macrotools.store_credential('email', ...)).
"""

import sys
import io
from pathlib import Path

# Add src to path so we can import macrotools directly from the repo
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import requests
from macrotools.storage import _resolve_credential

OUTPUT_DIR = Path(__file__).parent.parent / 'src' / 'macrotools' / 'data'

# BLS sources that have a {source}.series file with a series_title column.
# Note: 'jt' (JOLTS) has a series file but uses coded fields instead of
# series_title, so it requires separate handling and is excluded here.
BLS_SOURCES = ['ln', 'ce', 'ci', 'jt', 'cu', 'pc', 'wp', 'ei', 'cx', 'tu']

def generate_catalogs():
    email = _resolve_credential('email')
    headers = {'User-Agent': email}

    for source in BLS_SOURCES:
        print(f"\n--- Generating catalog for '{source}' ---")

        if source != 'jt':

            try:
                url = f'https://download.bls.gov/pub/time.series/{source}/{source}.series'
                r = requests.get(url, headers=headers)
                r.raise_for_status()

                series = pd.read_csv(io.StringIO(r.text), sep='\t', low_memory=False)
                series.columns = series.columns.str.strip()
                series['series_id'] = series['series_id'].str.strip()
                series['series_title'] = series['series_title'].str.strip()

                df = series[['series_id', 'series_title']].rename(
                    columns={'series_title': 'description'}
                )

                output_path = OUTPUT_DIR / f'series_{source}.csv'
                df.to_csv(output_path, index=False)
                print(f"  Wrote {len(df)} series to {output_path}")

            except Exception as e:
                print(f"  ERROR generating catalog for '{source}': {e}")

        elif source == 'jt':

            try:
                # Download all lookup files
                jt_lookups = {}
                for filename in ['dataelement', 'seasonal', 'ratelevel', 'state', 'sizeclass', 'industry']:
                    url = f'https://download.bls.gov/pub/time.series/jt/jt.{filename}'
                    r = requests.get(url, headers=headers)
                    r.raise_for_status()
                    df = pd.read_csv(io.StringIO(r.text), sep='\t', low_memory=False)
                    df.columns = df.columns.str.strip()
                    jt_lookups[filename] = df

                # Download series file
                url = f'https://download.bls.gov/pub/time.series/jt/jt.series'
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                series = pd.read_csv(io.StringIO(r.text), sep='\t', low_memory=False)
                series.columns = series.columns.str.strip()
                series['series_id'] = series['series_id'].str.strip()

                # Rename 'seasonal' to 'seasonal_code' to match lookup key convention
                series = series.rename(columns={'seasonal': 'seasonal_code'})

                # Merge each lookup to get text descriptions
                for filename in ['dataelement', 'seasonal', 'ratelevel', 'state', 'sizeclass', 'industry']:
                    code_col = f'{filename}_code'
                    text_col = f'{filename}_text'
                    lookup = jt_lookups[filename][[code_col, text_col]].copy()
                    lookup[code_col] = lookup[code_col].astype(str).str.strip()
                    series[code_col] = series[code_col].astype(str).str.strip()
                    series = series.merge(lookup, on=code_col, how='left')

                # Build description string from text columns
                series['description'] = (
                    series['dataelement_text']
                    + ', ' + series['ratelevel_text']
                    + ', ' + series['industry_text']
                    + ', ' + series['state_text']
                    + ', ' + series['sizeclass_text']
                    + ', ' + series['seasonal_text']
                )

                df = series[['series_id', 'description']]
                output_path = OUTPUT_DIR / f'series_{source}.csv'
                df.to_csv(output_path, index=False)
                print(f"  Wrote {len(df)} series to {output_path}")

            except Exception as e:
                print(f"  ERROR generating catalog for '{source}': {e}")

if __name__ == '__main__':
    generate_catalogs()
