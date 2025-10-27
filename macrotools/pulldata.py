from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np
import json, io, requests, webbrowser
from .utils import timer

@timer
def pull_flat_file(source, email = None, pivot = True, save_file = None):

    """
    Pull flat macro data files from BLS and BEA.
    
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
        Provide an email address to pull flat files from the BLS.

    save_file : str
        Provide a filepath to save the file as a .pkl file.
    """

    valid_sources = ['ce','ln','ci','jt','cu','pc','wp','stclaims']
    
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
            """
        )
    print(f"Pulling data from source: {source}")

    # Format flatfile -- BLS Sources
    if source in ['ce', 'ln', 'ci', 'jt', 'cu', 'pc', 'wp','ei']:

        if email is None:
            raise ValueError("An email address input is required in order to pull BLS series.")

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
        flatdata = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
        
        # Rename columns
        flatdata.columns = flatdata.columns.str.strip()

        # Clean up data
        flatdata['series_id'] = flatdata['series_id'].str.strip()
        flatdata['value'] = pd.to_numeric(flatdata['value'], errors='coerce')

        # Dates
        flatdata['frequency'] = flatdata['period'].apply(
            lambda x: 'A' if (x=='M13') or x[0]=='A' else ('Q' if x[0]=='Q' else 'M')
        )
        flatdata['month'] = flatdata['period'].apply(lambda x: pd.NA if (x=='M13') or (x=='A01') or (x[0]=='Q') else int(x[1:]))
        flatdata['quarter'] = flatdata['period'].apply(lambda x: pd.NA if (x[0]=='M') or (x=='A01') else int(x[2:]))

        # Pivot Data
        if pivot:
            print(f"Converting File to Pivot Table. Be aware that this will drop footnotes and only keep monthly data. If you want long data, set pivot=False")

            if source in ['ce', 'ln', 'jt', 'cu', 'pc', 'wp', 'ei']:

                flatdata = flatdata[flatdata['frequency']=='M'][['series_id','value','year','month']].copy()
                flatdata['date'] = pd.to_datetime(flatdata['year'].astype(str) + '-' + flatdata['month'].astype(str), format='%Y-%m')
                flatdata = flatdata.drop(['month','year'], axis=1)

                flatdata = flatdata.pivot_table(values = 'value', index = 'date', columns = 'series_id')
                flatdata = flatdata.asfreq('MS')

            if source in ['ci']:

                flatdata = flatdata[['series_id','value','year','quarter']].copy()
                flatdata['period'] = pd.PeriodIndex(flatdata['year'].astype(str) + 'Q' + flatdata['quarter'].astype(str), freq='Q')
                flatdata = flatdata.drop(['year','quarter'], axis=1)

                flatdata = flatdata.pivot_table(values = 'value', index = 'period', columns = 'series_id')
                flatdata = flatdata.asfreq('Q')

        # Attributes
        flatdata.attrs['data description'] = ''

        # Series
        if source in ['ln','ce', 'ci', 'cu','pc','wp']:
            r = requests.get(series_url, headers=headers)
            series = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
            series.columns = series.columns.str.strip()
            series['series_id'] = series['series_id'].str.strip()
            series['series_title'] = series['series_title'].str.strip()
            series_dict = series.set_index('series_id')['series_title'].to_dict()
            flatdata.attrs['series'] = series_dict

        if save_file: flatdata.to_pickle(save_file)
        return flatdata

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
        return stclaims

def pull_bls_data(series_list: Union[str, List],
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
    df = pd.concat(dfs, ignore_index=True)

    df['value'] = pd.to_numeric(df['value'])

    df['month'] = df['period'].apply(lambda x: pd.NA if (x=='M13') or (x=='A01') or (x[0]=='Q') else int(x[1:]))
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    df = df[['date', 'value', 'series_id']].pivot_table(values = 'value', index = 'date', columns = 'series_id')
    df = df.asfreq('MS')

    if save_file: df.to_pickle(save_file)

    return df
