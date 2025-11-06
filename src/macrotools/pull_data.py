from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np
import json, io, requests, webbrowser
from pathlib import Path
from .utils import timer
from .storage import (
    _get_cache_file_path,
    _should_refresh_cache,
    _load_cached_data,
    _save_cached_data,
    _get_email_for_bls,
)

@timer
def pull_data(source, email = None, pivot = True, save_file = None, freq='M', force_refresh=False, cache=True):

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
        'stclaims': State-level unemployment insurance claims

        Not yet implemented:
        'ppi': Producer Price Index
        'ipi': Import Price Index.
        'nipa': National Income and Product Accounts

    email : str
        Provide an email address to pull files from the BLS.

    save_file : str
        Provide a filepath to save the file as a .pkl file.

    cache : bool, default=True
        If True, use cached data if available (up to 7 days old).

    force_refresh : bool, default=False
        If True, ignore cache and pull fresh data.
    """

    valid_sources = [
        'ce',
        'ln',
        'ci',
        'jt',
        'cu',
        'pc',
        'wp',
        'ei',
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
            'ei': Import and Export Price Indices
            'stclaims': State UI Claims
            'nipa-pce': Monthly NIPA PCE Data
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
            'kc-svc': Dallas Fed Services Survey
            """
        )

    # Check cache first
    cache_file = _get_cache_file_path(source, pivot, freq)
    if cache and not force_refresh and not _should_refresh_cache(cache_file):
        cached_data = _load_cached_data(source, pivot, freq)
        if cached_data is not None:
            if save_file:
                cached_data.to_pickle(save_file)
            return cached_data

    print(f"Pulling data from source: {source}")

    # Format flatfile -- BLS Sources
    if source in ['ce', 'ln', 'ci', 'jt', 'cu', 'pc', 'wp','ei']:

        email = _get_email_for_bls(email)

        flat_file_name = {
            'ce': 'ce.data.0.AllCESSeries',
            'ln': 'ln.data.1.AllData',
            'ci': 'ci.data.1.AllData',
            'jt': 'jt.data.1.AllItems',
            'cu': 'cu.data.0.Current',
            'pc': 'pc.data.0.Current',
            'wp': 'wp.data.0.Current',
            'ei': 'ei.data.0.Current'
        }

        base_url = 'https://download.bls.gov/pub/time.series/' + source + '/'
        flat_file_url = base_url + flat_file_name[source]
        series_url = base_url + source + '.series'

        # Pull flat file data
        headers = {'User-Agent': email}
        r = requests.get(flat_file_url, headers=headers)
        data = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
        
        # Rename columns
        data.columns = data.columns.str.strip()

        # Clean up data
        data['series_id'] = data['series_id'].str.strip()
        data['value'] = pd.to_numeric(data['value'], errors='coerce')

        # Dates
        data['frequency'] = data['period'].apply(
            lambda x: 'A' if (x=='M13') or x[0]=='A' else ('Q' if x[0]=='Q' else 'M')
        )
        data['month'] = data['period'].apply(lambda x: pd.NA if (x=='M13') or (x=='A01') or (x[0]=='Q') else int(x[1:]))
        data['quarter'] = data['period'].apply(lambda x: pd.NA if (x[0]=='M') or (x=='A01') else int(x[2:]))

        # Pivot Data
        if pivot:
            print(f"Converting File to Pivot Table. Be aware that this will drop footnotes and only keep monthly data. If you want long data, set pivot=False")

            if source in ['ce', 'ln', 'jt', 'cu', 'pc', 'wp', 'ei']:

                data = data[data['frequency']=='M'][['series_id','value','year','month']].copy()
                data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
                data = data.drop(['month','year'], axis=1)

                data = data.pivot_table(values = 'value', index = 'date', columns = 'series_id')
                data = data.asfreq('MS')

            if source in ['ci']:

                data = data[['series_id','value','year','quarter']].copy()
                data['period'] = pd.PeriodIndex(data['year'].astype(str) + 'Q' + data['quarter'].astype(str), freq='Q')
                data = data.drop(['year','quarter'], axis=1)

                data = data.pivot_table(values = 'value', index = 'period', columns = 'series_id')
                data = data.asfreq('Q')

        # Attributes
        data.attrs['data description'] = ''

        # Series
        if source in ['ln','ce', 'ci', 'cu','pc','wp']:
            r = requests.get(series_url, headers=headers)
            series = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
            series.columns = series.columns.str.strip()
            series['series_id'] = series['series_id'].str.strip()
            series['series_title'] = series['series_title'].str.strip()
            series_dict = series.set_index('series_id')['series_title'].to_dict()
            data.attrs['series'] = series_dict

        if save_file: data.to_pickle(save_file)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data

    if source=='stclaims':

        # Pull flat file data
        webbrowser.open_new_tab('https://oui.doleta.gov/unemploy/csv/ar539.csv')
        stclaims_path = input('The DOL website does not permit bot access. I have opened the State-level unemployment data in a new browser. Please save the file and write the path, without quote marks, in this box, e.g.: C:/Users/prest/test/stclaims.csv ; if you input nothing I will look for the file in data/stclaims.csv ; Press Enter to Continue AFTER the file has finished downloading')

        if stclaims_path=='':
            stclaims = pd.read_csv('data/stclaims.csv', parse_dates=['rptdate','c2', 'c23','curdate','priorwk_pub','priorwk'])
        else:
            stclaims = pd.read_csv(stclaims_path, parse_dates=['rptdate','c2', 'c23','curdate','priorwk_pub','priorwk'])

        stclaims.rename(columns={
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
           'c23': 'changedate'}, inplace=True)

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

        stclaims.set_index(['state', 'weekending'], inplace=True)

        # Attributes
        stclaims.attrs['data_description'] = """Weekly state-level claims data. The structure is a multiindex with state (string) and weekending (Timestamps reflecting the Saturday the week ends). You can access subsets of the dataset using syntax like: stclaims.xs('2025-9-13', level='weekending').index.tolist(). Be careful that not all states appear in every week."""
        stclaims.attrs['series'] = column_descriptions
        stclaims.attrs['date_created'] = pd.Timestamp.now().date()

        if save_file: stclaims.to_pickle(save_file)
        if cache: _save_cached_data(stclaims, source, pivot, freq)
        return stclaims
    
    if source=='nipa-pce':

        if freq!='M':
            raise ValueError('Currently only frequency \'M\' is supported.')

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
        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='ny-mfg':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/Empire/data/ESMS_SeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='ny-svc':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/business_leaders/data/BLS_NotSeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='philly-mfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/MBOS/Historical-Data/Diffusion-Indexes/bos_dif.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)

        data['date'] = pd.to_datetime(data['DATE'], format='%b-%y')
        data.loc[data['date'].dt.year==2068, 'date'] = data.loc[data['date'].dt.year==2068, 'date'].apply(lambda x: x.replace(year=1968))
        data.drop(columns='DATE', axis=1, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot, freq)
        return data

    if source=='philly-nonmfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/NBOS/nboshistory.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='richmond-mfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/manufacturing/data/mfg_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='richmond-nonmfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/services/data/nmf_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data
    
    if source=='dallas-mfg':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tmos/documents/index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data.drop(columns='Date', axis=1, inplace=True)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data

    if source=='dallas-svc':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tssos_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['date'], format='%b-%y')
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
        return data

    if source=='dallas-retail':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tros_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data.drop(columns='Date', axis=1, inplace=True)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot, freq)
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
        if cache: _save_cached_data(data, source, pivot, freq)
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
        if cache: _save_cached_data(data, source, pivot, freq)
        return data

def pull_bls_series(series_list: Union[str, List],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    save_file: Optional[str] = None):

    """
    Pull single or multiple data series from the BLS.
    
    Parameters:
    -----------
    series_list: either a string or list of strings. For example
        'CUUR0000SA0'
        or
        ['CUUR0000SA0','SUUR0000SA0']

    Optional: start_year : int

        for example, 2014.
    Optional: end_year : int
        for example, 2024.

    Returns a pivot table with a DateTimeIndex and the series_list as columns.

    WARNING: Only use this for monthly data series at this time.
    """

    if isinstance(series_list, str):
        series_list = [series_list]

    json_dict = {'seriesid': series_list}
    if start_year: json_dict['startyear'] = start_year
    if end_year: json_dict['endyear'] = end_year

    # BLS API
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=json.dumps(json_dict), headers={'Content-type': 'application/json'})

    # Extract the list of series
    json_data = json.loads(p.text)['Results']['series']

    # Load data into dataframe
    dfs = []
    for item in json_data:
        series_id = item['seriesID']
        df = pd.DataFrame(item['data'])
        df['series_id'] = series_id
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    data['value'] = pd.to_numeric(data['value'])

    data['month'] = data['period'].apply(lambda x: pd.NA if (x=='M13') or (x=='A01') or (x[0]=='Q') else int(x[1:]))
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
    data = data[['date', 'value', 'series_id']].pivot_table(values = 'value', index = 'date', columns = 'series_id')
    data = data.asfreq('MS')

    if save_file: data.to_pickle(save_file)

    return data
