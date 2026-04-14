from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np
import io, requests, webbrowser
from pathlib import Path
from .utils import timer
from .storage import (
    _get_cache_file_path,
    _should_refresh_cache,
    _load_cached_data,
    _save_cached_data,
    _save_dataframe,
    _get_email_for_bls,
    _get_fred_api_key,
    _get_bls_api_key,
)

def get_series_list(source):
    """
    Return the series catalog for a BLS source as a DataFrame.

    Series catalogs are pre-built CSV files shipped with the macrotools package.
    Each CSV has two columns: 'series_id' and 'description'. They can be accessed
    directly as files at:

        import macrotools
        from pathlib import Path
        catalog_dir = Path(macrotools.__file__).parent / 'data'
        # Files: series_ln.csv, series_ce.csv, series_jt.csv, series_ci.csv,
        #        series_cu.csv, series_pc.csv, series_wp.csv, series_ei.csv,
        #        series_cx.csv, series_tu.csv

    Or via this function:

        import macrotools as mt
        df = mt.get_series_list('ln')
        df[df['description'].str.contains('unemployment rate', case=False)]

    Parameters:
    -----------
    source : str
        BLS source code. Valid sources:
        'ln': Household Survey
        'ce': Establishment Survey
        'jt': JOLTS (Job Openings and Labor Turnover Survey)
        'ci': ECI (Employment Cost Index)
        'cu': CPI (Consumer Price Index)
        'pc': PPI (Producer Price Index) Industry
        'wp': PPI (Producer Price Index)Commodity
        'ei': Import/Export Price Indices
        'cx': CEX (Consumer Expenditures)
        'tu': Time Use Survey

    Returns:
    --------
    pd.DataFrame with columns 'series_id' and 'description'
    """
    valid_sources = ['ln', 'ce', 'jt', 'ci', 'cu', 'pc', 'wp', 'ei', 'cx', 'tu']
    if source not in valid_sources:
        raise ValueError(
            f"Invalid source: '{source}'. Valid sources: {', '.join(valid_sources)}"
        )

    catalog_path = Path(__file__).parent / 'data' / f'series_{source}.csv'
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Series catalog for '{source}' not found at {catalog_path}. "
            "The catalog CSV may not have been generated yet."
        )

    return pd.read_csv(catalog_path)

@timer
def pull_data(source, email=None, freq=None, save_file=None, force_refresh=False, cache=True, columns=None):

    """
    Pull full data files from BLS and BEA.

    Parameters:
    -----------
    source : str
        The data source to pull.
        Valid source arguments:
        'ln': Household survey labor force statistics.
        'ce': Establishment survey statistics.
        'jt': Job Openings and Labor Turnover Survey
        'cu': CPI (all Urban Consumers)
        'pc': PPI Industry Data
        'wp': PPI Commodity Data
        'ei': Import/Export Price Index
        'cx': Consumer Expenditures
        'tu': Time Use Survey
        'nipa-pce': NIPA PCE Data
        'stclaims': State-level unemployment claims
        'ny-mfg': NYFed Empire Manufacturing Survey
        'ny-svc': NYFed Services Survey
        'philly-mfg': Philadelphia Fed Manufacturing Survey
        'philly-nonmfg': Philadelphia Fed Nonmanufacturing Survey
        'richmond-mfg': Richmond Fed Manufacturing Survey
        'richmond-nonmfg': Richmond Fed Nonmanufacturing Survey
        'dallas-mfg': Dallas Fed Manufacturing Survey
        'dallas-svc': Dallas Fed Services Survey
        'dallas-retail': Dallas Fed Retail Survey
        'kc-mfg': Kansas City Fed Manufacturing Survey
        'kc-svc': Kansas City Fed Services Survey

    freq : str, default None
        Frequency of data to return. Only applies to BLS flat file sources.
        'M': Monthly data
        'Q': Quarterly data
        'A': Annual data
        'S': Semiannual data
        'all': Return unpivoted long-format data with all frequencies.
        Note: M13 (annual average) observations are always dropped.
        Non-BLS sources ignore this parameter.
        If None, defaults to the source's natural frequency:
        'Q' for 'ci', 'A' for 'cx' and 'tu', 'M' for all others.

    email : str
        Provide an email address to pull files from the BLS.

    save_file : str
        Provide a filepath to save the pulled data.
        Format is inferred from the extension:
        .pkl/.pickle -> pickle; .parquet -> parquet; anything else -> feather (default).

    cache : bool, default=True
        If True, use cached data if available (up to 7 days old).

    force_refresh : bool, default=False
        If True, ignore cache and pull fresh data.
    """

    if isinstance(columns, str):
        columns = [columns]

    valid_sources = [
        'ce',
        'ln',
        'ci',
        'jt',
        'cu',
        'pc',
        'wp',
        'ei',
        'cx', 
        'tu',
        'stclaims',
        'nipa-pce',
        'ny-mfg',
        'ny-svc',
        'philly-mfg', 
        'philly-nonmfg', 
        'richmond-mfg', 
        'richmond-nonmfg', 
        'dallas-mfg', 
        'dallas-svc', 
        'dallas-retail', 
        'kc-mfg', 
        'kc-svc'
    ]
    
    # Check if source is valid, Pull with pulling data.
    if source not in valid_sources:
        raise ValueError(
            f"Invalid source: '{source}'. "
            """
            Please use one of the following sources:
            'ce': Establishment Survey
            'ln': Household Survey
            'ci': ECI
            'jt': JOLTS
            'cu': CPI
            'pc': PPI Industry
            'wp': PPI Commodity
            'cx': Consumer Expenditures
            'ei': Import and Export Price Indices
            'tu': Time Use Survey
            'nipa-pce': Monthly NIPA PCE Data
            'stclaims': State-level claims
            'ny-mfg': NYFed Empire Survey
            'ny-svc': NYFed Services Survey
            'philly-mfg': Philly Fed Mfg Survey
            'philly-nonmfg': Philly Fed Nonmfg Survey
            'richmond-mfg': Richmond Fed Mfg Survey
            'richmond-nonmfg': Richmond Fed Nonmfg Survey
            'dallas-mfg': Dallas Fed Mfg Survey
            'dallas-svc': Dallas Fed Services Survey
            'dallas-retail': Dallas Fed Retail Survey
            'kc-mfg': Kansas City Fed Mfg Survey
            'kc-svc': Kansas City Fed Services Survey
            """
        )

    # Apply source-specific default frequencies
    if freq is None:
        source_default_freqs = {'ci': 'Q', 'cx': 'A', 'tu': 'A'}
        freq = source_default_freqs.get(source, 'M')

    # Determine cache key: freq for BLS flat file sources, 'default' for others
    bls_sources = ['ce', 'ln', 'ci', 'jt', 'cu', 'pc', 'wp', 'ei', 'cx', 'tu']
    valid_freqs = {'M', 'Q', 'A', 'S', 'all'}
    if source in bls_sources:
        if freq not in valid_freqs:
            raise ValueError(f"Invalid freq: '{freq}'. Must be one of {valid_freqs}.")
        cache_freq = freq
    else:
        cache_freq = 'default'

    # Check cache first
    cache_file = _get_cache_file_path(source, cache_freq)
    if cache and not force_refresh and not _should_refresh_cache(cache_file):
        cached_data = _load_cached_data(source, cache_freq)
        if cached_data is not None:
            if columns is not None:
                if freq == 'all':
                    available = set(cached_data['series_id'].unique())
                    missing = [c for c in columns if c not in available]
                else:
                    missing = [c for c in columns if c not in cached_data.columns]
                if missing:
                    raise ValueError(f"Series not found in '{source}': {missing}")
                if freq == 'all':
                    cached_data = cached_data[cached_data['series_id'].isin(columns)]
                else:
                    cached_data = cached_data[columns]
            if save_file:
                _save_dataframe(cached_data, save_file)
            return cached_data

    # If data not cached, pull from source
    print(f"Pulling {source} data from source")

    if source in ['ce', 'ln', 'ci', 'jt', 'cu', 'pc', 'wp','ei', 'cx', 'tu']:

        email = _get_email_for_bls(email)

        flat_file_name = {
            'ce': 'ce.data.0.AllCESSeries',
            'ln': 'ln.data.1.AllData',
            'ci': 'ci.data.1.AllData',
            'jt': 'jt.data.1.AllItems',
            'cu': 'cu.data.0.Current',
            'pc': 'pc.data.0.Current',
            'wp': 'wp.data.0.Current',
            'ei': 'ei.data.0.Current',
            'cx': 'cx.data.1.AllData',
            'tu': 'tu.data.1.AllData'
        }

        base_url = 'https://download.bls.gov/pub/time.series/' + source + '/'
        flat_file_url = base_url + flat_file_name[source]

        # Pull flat file data
        headers = {'User-Agent': email}
        r = requests.get(flat_file_url, headers=headers)
        data = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
        
        # Rename columns
        data.columns = data.columns.str.strip()

        # Clean up data
        data['series_id'] = data['series_id'].str.strip()
        data['value'] = pd.to_numeric(data['value'], errors='coerce')

        # Clean periods and classify frequency
        data['period'] = data['period'].str.strip()

        data['frequency'] = data['period'].apply(
            lambda x: 'A' if (x == 'M13') or x[0] == 'A'
            else ('Q' if x[0] == 'Q'
            else ('S' if x[0] == 'S'
            else 'M'))
        )
        # Pivot by frequency
        if freq == 'all':
            data['period'] = data['period'].str[1:].astype(int)


        else:
            # Drop M13 annual averages for pivoted output
            data = data[data['period'] != 'M13']
            data = data.loc[data['frequency'] == freq].copy()

            if data.empty:
                raise ValueError(
                    f"No '{freq}' frequency data found in source '{source}'."
                )

            data['month'] = data['period'].apply(
                lambda x: pd.NA if x[0] in ('A', 'Q', 'S') else int(x[1:]))
            data['quarter'] = data['period'].apply(
                lambda x: int(x[2:]) if x[0] == 'Q' else pd.NA)

            if freq == 'M':
                data['date'] = pd.to_datetime(
                    data['year'].astype(str) + '-' + data['month'].astype(str))
                data = (data[['series_id', 'value', 'date']]
                        .pivot_table(values='value', index='date', columns='series_id')
                        .asfreq('MS'))

            elif freq == 'Q':
                if source == 'ci':
                    # ECI convention: Q01 -> March (quarter-end month)
                    data['date'] = pd.to_datetime(
                        data['year'].astype(str) + '-' + (data['quarter'] * 3).astype(str) + '-01')
                    asfreq_rule = 'QS-MAR'
                else:
                    # Standard: Q01 -> January (quarter-start month)
                    data['date'] = pd.to_datetime(
                        data['year'].astype(str) + '-' + (data['quarter'] * 3 - 2).astype(str) + '-01')
                    asfreq_rule = 'QS'
                data = (data[['series_id', 'value', 'date']]
                        .pivot_table(values='value', index='date', columns='series_id')
                        .asfreq(asfreq_rule))

            elif freq == 'A':
                data['date'] = pd.to_datetime(data['year'], format='%Y')
                data = (data[['series_id', 'value', 'date']]
                        .pivot_table(values='value', index='date', columns='series_id')
                        .asfreq('YS'))

            elif freq == 'S':
                data['half'] = data['period'].str[1:].astype(int)
                data['date'] = pd.to_datetime(
                    data['year'].astype(str) + '-' + (data['half'] * 6 - 5).astype(str) + '-01')
                data = (data[['series_id', 'value', 'date']]
                        .pivot_table(values='value', index='date', columns='series_id'))

            data.index.name = 'date'

        if cache: _save_cached_data(data, source, cache_freq)
        if columns is not None:
            if freq == 'all':
                available = set(data['series_id'].unique())
                missing = [c for c in columns if c not in available]
                if missing:
                    raise ValueError(f"Series not found in '{source}': {missing}")
                data = data[data['series_id'].isin(columns)]
            else:
                missing = [c for c in columns if c not in data.columns]
                if missing:
                    raise ValueError(f"Series not found in '{source}': {missing}")
                data = data[columns]
        if save_file: _save_dataframe(data, save_file)
        return data

    if source=='stclaims':

        # Pull flat file data
        webbrowser.open_new_tab('https://oui.doleta.gov/unemploy/csv/ar539.csv')
        stclaims_path = input('The DOL website does not permit bot access. I have opened the State-level unemployment data in a new browser. Please save the file and write the path, without quote marks, in this box, e.g.: C:/Users/prest/test/ar539.csv ; if you input nothing I will look for the file in data/ar539.csv ; Press Enter to Continue AFTER the file has finished downloading')

        if stclaims_path=='':
            filepath = 'data/ar539.csv'
        else:
            filepath = stclaims_path        
        
        stclaims = (pd.read_csv(filepath, parse_dates=['rptdate','c2', 'c23','curdate','priorwk_pub','priorwk'], low_memory=False)
                .rename(columns={
                'st': 'state',
                'c1': 'weeknumber',
                'c2': 'weekending',
                'c3': 'ic',
                'c4': 'fic',
                'c5': 'xic',
                'c6': 'wsic',
                'c7': 'wseic',
                'c8': 'cw',
                'c9': 'fcw',
                'c10': 'xcw',
                'c11': 'wscw',
                'c12': 'wsecw',
                'c13': 'ebt',
                'c14': 'ebui',
                'c15': 'abt',
                'c16': 'abui',
                'c17': 'at',
                'c18': 'ce',
                'c19': 'r',
                'c20': 'ar',
                'c21': 'p',
                'c22': 'status',
                'c23': 'changedate'}))

        column_descriptions = {
            'ic': 'State UI Initial Claims, less intrastate transitional.',
            'fic': 'UCFE-no UI Initial Claims.',
            'xic': 'UCX only Initial Claims',
            'wsic': 'STC or workshare total initial claims',
            'wseic': 'STC or workshare equivalent initial claims',
            'cw': 'State UI adjusted continued weeks claimed',
            'fcw': 'UCFE-no UI adjusted continued weeks claimed',
            'xcw': 'UCX only adjusted continued weeks claimed',
            'wscw': 'STC or workshare total continued weeks claimed',
            'wsecw': 'STC or workshare equivalent continued weeks claimed',
            'ebt': 'Total continued weeks claimed under the Federal/State Extended Benefit Program--includes all intrastate and interstate continued weeks claimed filed from an agent state under the state UI, UCFE and UCX programs.',
            'ebui': 'That part of EBT which represents only state UI weeks claimed under the Federal/State EB program.',
            'abt': 'Total continued weeks claimed under a state additional benefit program for those states which have such a program. (Includes UCFE and UCX.)',
            'abui': 'That part of ABT which represents only state UI additional continued weeks claimed for those states which have such a program.',
            'at': 'Average adjusted Total Continued Weeks Claimed.',
            'ce': 'Covered employment. 12-month average monthly covered employment for first 4 of last 6 completed quarters. Will only change once per quarter.',
            'r': 'Rate of insured unemployment.',
            'ar': 'Average Rate of Insured Employment in Prior Two Years',
            'p': 'Current Rate as Percent of Average Rate in Prior Two Years',
            'status': 'Beginning or Ending of State Extended Benefit Period',
            'changedate': 'If status has changed since prior week, date which change is effective.'
        }
        
        # Initial claims follow the report date; continuing claims follow the weekending date.
        stclaims_rptdate = (stclaims[['state','rptdate','weeknumber','ic','fic','xic','wsic','wseic']]
                            .rename(columns={'rptdate': 'weekending'})
                            .set_index(['state', 'weekending']))
        stclaims_weekending = (stclaims
                               .drop(['rptdate','weeknumber','ic','fic','xic','wsic','wseic'], axis=1)
                               .set_index(['state', 'weekending']))
        stclaims = pd.concat([stclaims_rptdate, stclaims_weekending], axis=1)

        stclaims['initial_claims'] = stclaims['ic'] + stclaims['wseic']
        stclaims['continuing_claims'] = stclaims['cw'] + stclaims['wsecw']

        stclaims = stclaims.sort_index()

        # Pivot Table (dates as rows, variables and states as columns)
        # stclaims = stclaims.asfreq('W-SAT')

        # Attributes
        stclaims.attrs['data_description'] = """Weekly state-level claims data. The structure is a pivot table where rows indexed by 'state' and 'weekending' (Timetamps reflecting the Saturday the week ends); columns are data types. Be careful that not all states appear in every week."""
        stclaims.attrs['series'] = column_descriptions
        stclaims.attrs['date_created'] = pd.Timestamp.now().date()

        if save_file: _save_dataframe(stclaims, save_file)
        if cache: _save_cached_data(stclaims, source, cache_freq)
        return stclaims
    
    if source=='nipa-pce':

        print('Pulling Monthly NIPA-PCE data.')

        freq = 'M'

        url_nipa = 'https://apps.bea.gov/national/Release/TXT/NipaData' + freq + '.txt'
        r = requests.get(url_nipa)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'%SeriesCode': 'series_code', 'Period': 'period', 'Value': 'value'}, inplace=True)

        # Keep only PCE Series
        pceseries = pd.read_csv(str(Path(__file__).parent / 'data' / 'pceseries.csv'))

        pceseries_melted = pd.melt(pceseries, id_vars
        =['line', 'name'], value_vars = ['quantitycode','pricecode','nominalcode','realcode'], var_name='datatype')
        pceseries_melted['datatype'] = pceseries_melted['datatype'].str.removesuffix('code')
        pceseries_melted.rename(columns={'value': 'series_code'}, inplace=True)
        data = pd.merge(data, pceseries_melted, how='right', right_on='series_code', left_on='series_code')

        # Format date column
        if freq=='M':
            data['date'] = pd.to_datetime(data['period'], format='%YM%m')
        data.drop(['period'], axis=1, inplace=True)

        # Format numeric
        data['value'] = pd.to_numeric(data['value'].str.replace(',',''))

        # Format as pivot table
        data = data.pivot_table(values='value', index = 'date', columns = ['line', 'datatype'])
        if freq=='M':
            data.asfreq('MS')

        data.attrs['series'] = pceseries.set_index('line')['name'].to_dict()
        data.attrs['parents'] = pceseries.set_index('line')['parent'].astype('Int64').to_dict()
        data.attrs['levels'] = pceseries.set_index('line')['level'].to_dict()
        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='ny-mfg':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/Empire/data/ESMS_SeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data['date'] = pd.to_datetime(data['date']).dt.to_period('M').dt.to_timestamp()
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='ny-svc':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/business_leaders/data/BLS_NotSeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data['date'] = pd.to_datetime(data['date']).dt.to_period('M').dt.to_timestamp()
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='philly-mfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/MBOS/Historical-Data/Diffusion-Indexes/bos_dif.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)

        data['date'] = pd.to_datetime(data['DATE'], format='%b-%y')
        data.loc[data['date'].dt.year==2068, 'date'] = data.loc[data['date'].dt.year==2068, 'date'].apply(lambda x: x.replace(year=1968))
        data.drop(columns='DATE', axis=1, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, cache_freq)
        return data

    if source=='philly-nonmfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/NBOS/nboshistory.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='richmond-mfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/manufacturing/data/mfg_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='richmond-nonmfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/non-manufacturing/data/nmf_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='dallas-mfg':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tmos/documents/index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data = (data
                .drop(columns='Date', axis=1)
                .dropna(subset=['date'])
                .set_index('date')
        )
        if cache: _save_cached_data(data, source, cache_freq)
        return data

    if source=='dallas-svc':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tssos_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['date'], format='%b-%y')
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, cache_freq)
        return data

    if source=='dallas-retail':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tros_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data = (data
            .dropna(subset=['date'])
            .drop(columns='Date', axis=1)
            .set_index('date')
        )
        if cache: _save_cached_data(data, source, cache_freq)
        return data
    
    if source=='kc-mfg':

        webbrowser.open_new_tab('https://www.kansascityfed.org/surveys/manufacturing-survey/')
        url = input('Enter the url for the services survey data at the tab opened.')
        data = pd.read_excel(url, skiprows=2)
        data.loc[2:16, 'Unnamed: 0'] = data.iloc[2:16]['Unnamed: 0'].astype(str) + ' vs month ago sa'
        data.loc[18:32, 'Unnamed: 0'] = data.iloc[18:32]['Unnamed: 0'].astype(str) + ' vs month ago nsa'
        data.loc[34:48, 'Unnamed: 0'] = data.iloc[34:48]['Unnamed: 0'].astype(str) + ' vs year ago nsa'
        data.loc[50:64, 'Unnamed: 0'] = data.iloc[50:64]['Unnamed: 0'].astype(str) + ' exp six months sa'
        data.loc[66:80, 'Unnamed: 0'] = data.iloc[66:80]['Unnamed: 0'].astype(str) + ' exp six months nsa'
        data = data.drop([0,1, 16, 17, 32, 33, 48, 49, 64, 65])

        data = data.set_index('Unnamed: 0').transpose()
        if cache: _save_cached_data(data, source, cache_freq)
        return data

    if source=='kc-svc':

        webbrowser.open_new_tab('https://www.kansascityfed.org/surveys/services-survey/')
        url = input('Enter the url for the services survey data at the tab opened.')
        data = pd.read_excel(url, skiprows=2)
        data.loc[2:13, 'Unnamed: 0'] = data.iloc[2:13]['Unnamed: 0'].astype(str) + ' vs month ago sa'
        data.loc[16:27, 'Unnamed: 0'] = data.iloc[16:27]['Unnamed: 0'].astype(str) + ' vs month ago nsa'
        data.loc[30:43, 'Unnamed: 0'] = data.iloc[30:43]['Unnamed: 0'].astype(str) + ' vs year ago nsa'
        data.loc[46:57, 'Unnamed: 0'] = data.iloc[46:57]['Unnamed: 0'].astype(str) + ' exp six months sa'
        data.loc[60:71, 'Unnamed: 0'] = data.iloc[60:71]['Unnamed: 0'].astype(str) + ' exp six months nsa'
        data = data.drop([0,1, 13, 14, 15, 27, 28, 29, 43, 44, 45, 57, 58, 59])

        data = data.set_index('Unnamed: 0').transpose()
        if cache: _save_cached_data(data, source, cache_freq)
        return data

def _detect_series_freq(source: str, series_ids: list) -> str:
    """
    Detect the frequency of BLS series from the series ID suffix.

    Returns one of: 'M', 'Q', 'A'
    Raises ValueError if series have mixed frequencies.
    """
    # Sources with fixed frequencies
    if source == 'ci':
        return 'Q'
    if source in ('cx', 'tu'):
        return 'A'

    # 'ln': trailing 'Q' = quarterly, trailing 'A' = annual, else monthly
    # 'cu': 4th char 'S' (CUUS) = semiannual, 'R' (CUUR/CUSR) = monthly
    # All other BLS sources (ce, jt, pc, wp, ei) are monthly.
    freqs = set()
    for sid in series_ids:
        if source == 'ln' and sid.endswith('Q'):
            freqs.add('Q')
        elif source == 'ln' and sid.endswith('A'):
            freqs.add('A')
        elif source == 'cu' and sid[3] == 'S':
            freqs.add('S')
        else:
            freqs.add('M')

    if len(freqs) > 1:
        raise ValueError(
            f"Mixed frequencies detected: {freqs}. "
            f"All series must have the same frequency. "
            f"Separate series by frequency into different calls."
        )

    return freqs.pop()


def pull_bls_series(series_list: Union[str, List],
    date_range = None,
    save_file: Optional[str] = None,
    force_refresh=False,
    source: str = 'flatfiles',
    api_key: Optional[str] = None):

    """
    Pull single or multiple data series from the BLS.

    Parameters:
    -----------
    series_list: either a string or list of strings. For example
        'CUUR0000SA0'
        or
        ['CUUR0000SA0','SUUR0000SA0']

    Optional: date_range: tuple
        e.g. ('2020', '2021') or ('2020-3', '2021-6')

    Optional: save_file - save pulled data to file.
        Format is inferred from extension: .pkl/.pickle -> pickle;
        .parquet -> parquet; anything else -> feather (default). e.g. 'data.feather'

    source : str, default 'flatfiles'
        Data source method.
        'flatfiles': Download full BLS flat files (default, best for large pulls)
        'api': Use BLS API v2 (faster for a small number of specific series)

    api_key : str, optional
        BLS API v2 registration key. Only used when source='api'.
        If not provided, will check stored credentials, environment variable
        BLS_API_KEY, or prompt user. Register free at:
        https://data.bls.gov/registrationEngine/

    Returns a pivot table with a DateTimeIndex and the series_list as columns.
    All series must share the same frequency; raises ValueError if mixed.
    """

    if isinstance(series_list, str):
        series_list = [series_list]

    if source not in ('flatfiles', 'api'):
        raise ValueError(f"Invalid source: '{source}'. Must be 'flatfiles' or 'api'.")

    if source == 'flatfiles':

        valid_sources = [
            'ce', 'ln', 'ci', 'jt', 'cu', 'pc', 'wp', 'ei', 'cx', 'tu'
        ]

        # Validate prefixes and group series by BLS source
        source_groups = {}
        for series in series_list:
            prefix = series[0:2].lower()
            if prefix not in valid_sources:
                raise ValueError(
                f"Invalid series prefix: '{prefix}'. "
                """
                Please choose a series from one of the following BLS sources:
                'ce': Establishment Survey
                'ln': Household Survey
                'ci': ECI
                'jt': JOLTS
                'cu': CPI
                'pc': PPI Industry
                'wp': PPI Commodity
                'ei': Import and Export Price Indices
                'cx': Consumer Expenditures Survey
                'tu': Time Use Survey
                """
            )
            source_groups.setdefault(prefix, []).append(series)

        # Detect frequency for each source group and ensure all match
        detected_freq = None
        for bls_source, group_series in source_groups.items():
            group_freq = _detect_series_freq(bls_source, group_series)
            if detected_freq is None:
                detected_freq = group_freq
            elif group_freq != detected_freq:
                raise ValueError(
                    f"Mixed frequencies across sources: '{detected_freq}' and '{group_freq}'. "
                    f"All series must have the same frequency."
                )

        data_list = []
        for bls_source, group_series in source_groups.items():
            df = pull_data(bls_source, freq=detected_freq, force_refresh=force_refresh, columns=group_series)
            if date_range:
                df = df.loc[date_range[0]:date_range[1]]
            data_list.append(df)

        data = pd.concat(data_list, axis=1)

    elif source == 'api':

        api_key = _get_bls_api_key(api_key)

        # Determine year range
        if date_range:
            start_year = int(date_range[0].split('-')[0])
            end_year = int(date_range[1].split('-')[0])
        else:
            end_year = pd.Timestamp.now().year
            start_year = end_year - 9

        # Build year chunks (max 10 years per API request)
        year_chunks = []
        y = start_year
        while y <= end_year:
            chunk_end = min(y + 9, end_year)
            year_chunks.append((str(y), str(chunk_end)))
            y = chunk_end + 1

        # Build series chunks (max 50 per API request)
        series_chunks = [series_list[i:i+50] for i in range(0, len(series_list), 50)]

        # Fetch data from BLS API v2
        all_records = []
        for s_chunk in series_chunks:
            for y_start, y_end in year_chunks:
                payload = {
                    'seriesid': s_chunk,
                    'startyear': y_start,
                    'endyear': y_end,
                    'registrationkey': api_key,
                }
                import json
                response = requests.post(
                    'https://api.bls.gov/publicAPI/v2/timeseries/data/',
                    data=json.dumps(payload),
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                result = response.json()

                if result.get('status') != 'REQUEST_SUCCEEDED':
                    raise RuntimeError(
                        f"BLS API error: {result.get('message', result.get('status', 'Unknown error'))}"
                    )

                for series_obj in result['Results']['series']:
                    sid = series_obj['seriesID']
                    for obs in series_obj['data']:
                        all_records.append({
                            'series_id': sid,
                            'year': obs['year'],
                            'period': obs['period'],
                            'value': obs['value'],
                        })

        if not all_records:
            raise ValueError("No data returned from BLS API for the requested series and date range.")

        raw = pd.DataFrame(all_records)
        raw['value'] = pd.to_numeric(raw['value'], errors='coerce')

        # Drop M13 annual averages
        raw = raw[raw['period'] != 'M13']

        # Detect frequency from API response
        raw['frequency'] = raw['period'].apply(
            lambda x: 'A' if x[0] == 'A'
            else ('Q' if x[0] == 'Q'
            else ('S' if x[0] == 'S'
            else 'M'))
        )
        api_freqs = raw['frequency'].unique()
        if len(api_freqs) > 1:
            raise ValueError(
                f"Mixed frequencies in API response: {set(api_freqs)}. "
                f"All series must have the same frequency."
            )
        api_freq = api_freqs[0]

        if api_freq == 'M':
            raw['date'] = pd.to_datetime(
                raw['year'].astype(str) + '-' + raw['period'].str[1:].astype(str),
                format='%Y-%m'
            )
        elif api_freq == 'Q':
            raw['quarter'] = raw['period'].str[2:].astype(int)
            raw['date'] = pd.to_datetime(
                raw['year'].astype(str) + '-' + (raw['quarter'] * 3 - 2).astype(str) + '-01')
        elif api_freq == 'A':
            raw['date'] = pd.to_datetime(raw['year'], format='%Y')
        elif api_freq == 'S':
            raw['half'] = raw['period'].str[1:].astype(int)
            raw['date'] = pd.to_datetime(
                raw['year'].astype(str) + '-' + (raw['half'] * 6 - 5).astype(str) + '-01')

        data = raw.pivot_table(values='value', index='date', columns='series_id').sort_index()
        data.index.name = 'date'
        data.columns.name = None

        if date_range:
            data = data.loc[date_range[0]:date_range[1]]

    if save_file: _save_dataframe(data, save_file)
    return data

def search_bls_series(source: str,
                      query: str,
                      sa: Optional[bool] = None,
                      field: str = 'description',
                      n: int = 10) -> pd.DataFrame:
    """
    Fuzzy-search BLS series catalogs using VS Code-style subsequence matching.

    Splits query on whitespace into tokens. Every token must independently match
    somewhere in the target string as a subsequence. Returns top n matches ranked
    by fuzzy match score.

    Parameters:
    -----------
    source : str
        BLS source code (e.g., 'ln', 'ce', 'cu', 'jt').

    query : str
        Search query. Split on whitespace into tokens; each token is matched
        independently as a subsequence against the target field.

    sa : bool, optional
        Filter by seasonal adjustment. Only applies to programs with SA codes
        (ce, ci, cu, jt, ln, wp). True = seasonally adjusted (3rd char 'S'),
        False = not seasonally adjusted (3rd char 'U'), None = no filter (default).

    field : str, default 'description'
        Column to search: 'description' or 'id' (searches series_id).

    n : int, default 10
        Number of top results to return.

    Returns:
    --------
    pd.DataFrame
        Top n matches with columns [series_id, description, score],
        sorted by score descending.

    Examples:
    ---------
        import macrotools as mt
        mt.search_bls_series('ln', 'labor force level 25 54')
        mt.search_bls_series('ce', 'total nonfarm', sa=True)
        mt.search_bls_series('cu', 'SA0', field='id')
    """
    catalog = get_series_list(source)

    # Apply seasonal adjustment filter
    if sa is not None and source in ('ce', 'ci', 'cu', 'jt', 'ln', 'wp'):
        sa_char = 'S' if sa else 'U'
        catalog = catalog[catalog['series_id'].str[2] == sa_char]

    tokens = query.lower().split()
    search_col = 'series_id' if field == 'id' else 'description'

    scores = []
    for _, row in catalog.iterrows():
        target = row[search_col].lower()
        total_score = 0
        matched = True

        for token in tokens:
            # Try starting the subsequence match at every position where the
            # first character matches; keep the best-scoring alignment.
            best_token_score = -1
            for start in range(len(target) - len(token) + 1):
                if target[start] != token[0]:
                    continue
                # Score this alignment: greedily match subsequence from start
                ti = 0  # index into token
                prev_pos = -2  # position of previous matched char
                alignment_score = 0
                for pos in range(start, len(target)):
                    if ti < len(token) and target[pos] == token[ti]:
                        char_score = 1  # base
                        if pos == prev_pos + 1:
                            char_score += 8  # consecutive
                        if pos == 0 or target[pos - 1] in ' _-.':
                            char_score += 5  # word boundary
                        if pos == 0 and ti == 0:
                            char_score += 10  # prefix
                        alignment_score += char_score
                        prev_pos = pos
                        ti += 1
                if ti == len(token):  # full token matched
                    # Exact substring bonuses
                    idx = target.find(token)
                    if idx == 0:
                        alignment_score += 35
                    elif idx > 0:
                        alignment_score += 20
                    if alignment_score > best_token_score:
                        best_token_score = alignment_score

            if best_token_score < 0:
                matched = False
                break
            total_score += best_token_score

        # Brevity normalization: concise matches rank above verbose ones
        if matched:
            score = round(total_score * 1000 / len(target))
            # Zero bonus: boost aggregate/national series (more 0s in ID)
            if source in ('ce', 'ci', 'cu', 'jt', 'ln'):
                score += row['series_id'].count('0') * 200
            scores.append(score)
        else:
            scores.append(None)

    catalog = catalog.copy()
    catalog['score'] = pd.array(scores, dtype=pd.Int64Dtype())
    result = catalog.dropna(subset=['score'])
    result = result.nlargest(n, 'score')[['series_id', 'description', 'score']]
    return result.reset_index(drop=True)

@timer
def alfred_as_reported(
    fred_series: str,
    function: Optional[callable] = None,
    release_start_date: Optional[str] = None,
    release_end_date: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Pull as-reported data from ALFRED (Archival FRED).

    Returns a DataFrame indexed by vintage dates, where each row contains
    the last observed data point available at that vintage date and the
    date of that observation. This captures how data evolved over time as
    revisions were released.

    Parameters:
    -----------
    fred_series : str
        FRED series ID (e.g., 'GDP', 'UNRATE', 'CPIAUCSL')

    function : callable, optional
        Transformation function to apply to the data before taking the last value.
        Examples: np.log, lambda x: x * 100, or custom functions
        The function is applied to the entire vintage series, then the last value is taken.

    release_start_date : str, optional
        Filter vintage dates >= this date. Format: 'YYYY-MM-DD' or any pandas-parseable date.

    release_end_date : str, optional
        Filter vintage dates <= this date. Format: 'YYYY-MM-DD' or any pandas-parseable date.

    api_key : str, optional
        FRED API key. If not provided, will check stored credentials, environment
        variable FRED_API_KEY, or prompt user.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by vintage dates (DateTimeIndex), with two columns:
        - 'value': Last observed data point at each vintage date
        - 'last_date': The observation date of that last value

    Attributes:
    -----------
    The returned DataFrame has the following attributes:
    - attrs['series_id']: FRED series ID
    - attrs['source']: 'ALFRED'
    - attrs['function']: String representation of transformation function applied
    - attrs['release_date_range']: Tuple of (start, end) release dates if filtered
    - attrs['date_created']: Timestamp when data was pulled

    Notes:
    ------
    - Requires fredapi package: `pip install fredapi` or `pip install macrotools[fred]`
    - Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    - Uses a single API call to `get_series_all_releases` for performance (not repeated calls per vintage)

    Examples:
    ---------
    # Basic usage - get GDP as reported
    gdp_as_reported = alfred_as_reported('GDP')

    # Get unemployment rate with date filtering
    unrate = alfred_as_reported('UNRATE',
                                release_start_date='2020-01-01',
                                release_end_date='2023-12-31')

    # Apply log transformation before taking last value
    import numpy as np
    gdp_log = alfred_as_reported('GDP', function=np.log)

    # Apply user-defined function before taking last value
    # quarterly_3ma_growth(data) takes the mean, the 3-month change, and annualizes it:
    def quarterly_3ma_growth(data):
        return (1 + data.rolling(window=3).mean().pct_change(periods=3))**4  - 1
    as_reported_payems = alfred_as_reported(fred_series='PAYEMS', function = quarterly_3ma_growth)
    """

    # Check for fredapi dependency
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi package is required for alfred_as_reported(). "
            "Install with: pip install fredapi or pip install macrotools[fred]"
        )

    # Get API key
    api_key = _get_fred_api_key(api_key)

    # Initialize Fred client
    fred = Fred(api_key=api_key)

    # Get all releases for the series
    try:
        all_releases = fred.get_series_all_releases(fred_series)
    except Exception as e:
        raise ValueError(
            f"Error retrieving releases for series '{fred_series}': {e}\n"
            "Check that the series ID is valid and your API key is correct."
        )

    # Rename columns to be clearer
    all_releases = all_releases.rename(columns={'realtime_start': 'vintage_date', 'date': 'obs_date'})

    # Convert to datetime for filtering
    all_releases['vintage_date'] = pd.to_datetime(all_releases['vintage_date'])
    all_releases['obs_date'] = pd.to_datetime(all_releases['obs_date'])

    # List of all Vintage Dates
    all_vintage_dates = all_releases['vintage_date'].sort_values().unique()

    # Filter vintage dates by release_start_date and release_end_date
    vintage_dates = all_vintage_dates
    if release_start_date:
        release_start = pd.to_datetime(release_start_date)
        vintage_dates = [d for d in vintage_dates if d >= release_start]
    if release_end_date:
        release_end = pd.to_datetime(release_end_date)
        vintage_dates = [d for d in vintage_dates if d <= release_end]

    if len(vintage_dates) == 0:
        raise ValueError(
            f"No vintage dates found for series '{fred_series}' "
            f"in the specified release date range."
        )

    print(f"Processing {len(vintage_dates)} vintage dates...")

    # Process each vintage date
    results = []
    for vintage_date in vintage_dates:
        try:
            # Filter to releases published by this vintage date
            available_releases = all_releases[all_releases['vintage_date'] <= vintage_date].copy()

            if len(available_releases) == 0:
                continue

            # For each observation date, keep only the most recent vintage
            # This reconstructs the full series as it appeared at this vintage date
            vintage_series = (available_releases
                             .sort_values('vintage_date')
                             .groupby('obs_date', as_index=False)
                             .last())

            if len(vintage_series) == 0:
                continue

            # Apply transformation function if provided
            if function:
                vintage_series['value'] = function(vintage_series['value'])

            # Get the most recent observation (maximum obs_date)
            last_row = vintage_series.loc[vintage_series['obs_date'].idxmax()]

            results.append({
                'vintage_date': vintage_date,
                'value': last_row['value'],
                'last_date': last_row['obs_date']
            })

        except Exception as e:
            # Log warning but continue processing other vintages
            print(f"Warning: Could not process vintage date {vintage_date}: {e}")
            continue

    # Convert to DataFrame
    if len(results) == 0:
        raise ValueError(f"No data retrieved for series '{fred_series}' with the specified parameters.")

    result_df = pd.DataFrame(results)
    output = result_df.set_index('vintage_date')
    output.index.name = 'vintage_date'

    # Add metadata
    output.attrs['series_id'] = fred_series
    output.attrs['source'] = 'ALFRED'
    output.attrs['function'] = function.__name__ if function else None
    output.attrs['release_date_range'] = (release_start_date, release_end_date)
    output.attrs['date_created'] = pd.Timestamp.now()

    return output