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
    _get_email_for_bls,
    _get_fred_api_key,
)

@timer
def pull_data(source, email = None, pivot = True, save_file = None, force_refresh=False, cache=True):

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

    # Check cache first
    cache_file = _get_cache_file_path(source, pivot)
    if cache and not force_refresh and not _should_refresh_cache(cache_file):
        cached_data = _load_cached_data(source, pivot)
        if cached_data is not None:
            if save_file:
                cached_data.to_pickle(save_file)
            return cached_data

    print(f"Pulling data from source: {source}")

    # Format flatfile -- BLS Sources
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
                data = data.loc[data['frequency']=='M', ['series_id','value','year','month']]
                data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
                data = (data
                        .drop(['month','year'], axis=1)
                        .pivot_table(values = 'value', index = 'date', columns = 'series_id')
                        .asfreq('MS'))

            if source in ['ci']:
                data = data[['series_id','value','year','quarter']]
                data['period'] = pd.to_datetime(data['year'].astype(str) + '-' + (data['quarter'] * 3).astype(str) + '-01')
                data = (data
                        .drop(['year','quarter'], axis=1)
                        .pivot_table(values = 'value', index = 'period', columns = 'series_id')
                        .asfreq('QS-MAR'))
                
            if source in ['cx', 'tu']:
                data = data[['series_id', 'value', 'year']]
                data['date'] = pd.to_datetime(data['year'], format='%Y')
                data = (data
                        .set_index('date')
                        .drop('year', axis=1)
                        .pivot_table(values = 'value', index = 'date', columns = 'series_id')
                        .asfreq('YS'))

        # Attributes
        data.attrs['data description'] = ''

        # Series
        if source in ['ln','ce', 'ci', 'cu','pc','wp','cx', 'tu']:
            r = requests.get(series_url, headers=headers)
            series = pd.read_csv(io.StringIO(r.text), sep = '\t', low_memory=False)
            series.columns = series.columns.str.strip()
            series['series_id'] = series['series_id'].str.strip()
            series['series_title'] = series['series_title'].str.strip()
            series_dict = series.set_index('series_id')['series_title'].to_dict()
            data.attrs['series'] = series_dict

        if save_file: data.to_pickle(save_file)
        if cache: _save_cached_data(data, source, pivot)
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

        if save_file: stclaims.to_pickle(save_file)
        if cache: _save_cached_data(stclaims, source, pivot)
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
        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='ny-mfg':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/Empire/data/ESMS_SeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='ny-svc':

        url = 'https://www.newyorkfed.org/medialibrary/media/Survey/business_leaders/data/BLS_NotSeasonallyAdjusted_Diffusion.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)
        data.rename(columns={'surveyDate': 'date'}, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='philly-mfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/MBOS/Historical-Data/Diffusion-Indexes/bos_dif.csv'
        r = requests.get(url)
        data = pd.read_csv(io.StringIO(r.text), sep = ',', low_memory=False)

        data['date'] = pd.to_datetime(data['DATE'], format='%b-%y')
        data.loc[data['date'].dt.year==2068, 'date'] = data.loc[data['date'].dt.year==2068, 'date'].apply(lambda x: x.replace(year=1968))
        data.drop(columns='DATE', axis=1, inplace=True)
        data.set_index('date', inplace=True)

        if cache: _save_cached_data(data, source, pivot)
        return data

    if source=='philly-nonmfg':

        url = 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/NBOS/nboshistory.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='richmond-mfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/manufacturing/data/mfg_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='richmond-nonmfg':

        url = 'https://www.richmondfed.org/-/media/RichmondFedOrg/region_communities/regional_data_analysis/regional_economy/surveys_of_business_conditions/non-manufacturing/data/nmf_historicaldata.xlsx'
        data = pd.read_excel(url)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
        return data
    
    if source=='dallas-mfg':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tmos/documents/index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data.drop(columns='Date', axis=1, inplace=True)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
        return data

    if source=='dallas-svc':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tssos_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['date'], format='%b-%y')
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
        return data

    if source=='dallas-retail':
        url = 'https://www.dallasfed.org/~/media/Documents/research/surveys/tssos/documents/tros_index_sa.xls'
        data = pd.read_excel(url)
        data['date'] = pd.to_datetime(data['Date'], format='%b-%y')
        data.drop(columns='Date', axis=1, inplace=True)
        data.set_index('date', inplace=True)
        if cache: _save_cached_data(data, source, pivot)
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
        if cache: _save_cached_data(data, source, pivot)
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
        if cache: _save_cached_data(data, source, pivot)
        return data

def pull_bls_series(series_list: Union[str, List],
    date_range = None,
    save_file: Optional[str] = None,
    force_refresh=False):

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

    Optional: save_file - save as pickle
        e.g. 'data.pkl'

    Returns a pivot table with a DateTimeIndex and the series_list as columns.

    WARNING: Only use this for monthly data series at this time.
    """

    if isinstance(series_list, str):
        series_list = [series_list]

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
        'tu'
    ]

    data_list = []
    for series in series_list:
        
        if series[0:2].lower() not in valid_sources:
            raise ValueError(
            f"Invalid source: '{source}'. "
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

        if date_range:
            individual_series = pull_data(series[0:2].lower(), force_refresh=force_refresh).loc[date_range[0]:date_range[1]][series]
        else:
            individual_series = pull_data(series[0:2].lower(), force_refresh=force_refresh)[series]
        data_list.append(individual_series)
    
    data = pd.concat(data_list, axis=1)

    if save_file: data.to_pickle(save_file)
    return data

def search_bls_series(source, input):

    """
    Function to search the 'series' attribute on series dictionaries in pulled BLS data.
    
    Parameters:
    -----------
    source: a dataset pulled by pull_data, or one that has a series dictionary stored in source.attrs['series']

    string_list: a list of strings you want to find in the series names, or a single string

    How it works:
    -----------
    search_bls series finds every series name in source.attrs['series'] that contains each string in string_list (case-insensitive). A series name must contain every string in string_list to be returned. search_bls returns a dictionary of series ids and names.

    Example:
    -----------
    import macrotools as mt
    found_series = search_bls_series('ce', ['Average Hourly Earnings', 'nonsupervisory', 'mining', 'seAsOnaLly aDjusTed'])
    """

    if isinstance(input, str):
        string_list = [input]
    else:
        string_list = input

    series_list = pull_data(source, force_refresh=False).attrs['series']

    found_series = []
    for (key, value) in series_list.items():
        if all(string.casefold() in value.casefold() for string in string_list):
            found_series.append(key)
    
    print(f'Found {len(found_series)} series that match your search.')
    return {k: series_list[k] for k in found_series}

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