from typing import Union, List, Dict, Optional
import numpy as np

import pandas as pd
import datetime as dt
import warnings
from dateutil.relativedelta import relativedelta

# Annualized Growth Rate Function
def cagr(data: Union[pd.DataFrame, pd.Series], lag = 1, ma = 1):

    """
    Calculate Continuous Annualized Growth Rates on a Dataframe

    Parameters:
    -----------
    data : DataFrame with Quarterly, Monthly, or Annual pandas Datetime Index
    lag : how many periods to calculate growth rate over (default value = 1)
    ma : take moving average before calculating growth rate (default value = 1, no moving average)
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError('Expected DataFrame input.')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError('Expected DataFrame index as pandas DatetimeIndex.')

    if data.index.freq.freqstr[0]=='M':
        return (data.rolling(ma).mean() / data.rolling(ma).mean().shift(lag)).pow(12/lag) - 1
    elif data.index.freq.freqstr[0]=='Q':
        return (data.rolling(ma).mean() / data.rolling(ma).mean().shift(lag)).pow(4/lag) - 1
    elif data.index.freq.freqstr[0]=='A':
        return (data.rolling(ma).mean() / data.rolling(ma).mean().shift(lag)).pow(1/lag) - 1
    else:
        raise Exception('Currently cagr only supports indices with Monthly, Quarterly or Annual Frequencies.')

def rebase(data: Union[pd.DataFrame, pd.Series], baseperiod, basevalue = 100):

    """
    Rebases time series.

    Parameters:
    data : Data Series(es) to rebase
        DataFrame with Quarterly, Monthly, or Annual pandas Datetime Index
    baseperiod : What period (or over what period) to rebase. If entry is one period, will take the value at that period. If entry is a tuple, will take the average value over that period.
        a string (e.g. '2020-1') or datetime object (pd.to_datetime('2020-1')) or tuple of these ('2020-1','2020-12').
        If baseperiod is a year, e.g. '2020', will take the average over 2020.
    basevalue : Optional numeric; what value to rebase to; 100 by default.
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError('Expected DataFrame input.')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError('Expected DataFrame index as pandas DatetimeIndex.')
    
    if isinstance(baseperiod, tuple):
        output = basevalue * data / data.loc[baseperiod[0]:baseperiod[1]].mean()

    else:
        output = basevalue * data / data.loc[baseperiod:baseperiod].mean()

    return output
