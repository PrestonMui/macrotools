from typing import Union, List, Dict, Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm

import pandas as pd
import datetime as dt
import warnings
from dateutil.relativedelta import relativedelta
from PIL import Image
from pathlib import Path

# Things I'd like to implement
# Allow input of y series without dictionary
# Fix position of title when there is a legend and no subtitle

########################################
# COLORS
########################################

# Apply stylesheet
stylefile = Path(__file__).parent / 'styles' / 'eastyle.mpstyle'
plt.style.use(str(stylefile))

# Extract and store colors from stylesheet
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Also create named dictionary if helpful
eacolors = {
    'eablue': style_colors[0],
    'eaorange': style_colors[1],
    'eagreen': style_colors[2],
    'eapurple': style_colors[3],
    'eayellow': style_colors[4],
    'eamagenta': style_colors[5],
    'eaviolet': style_colors[6]
}

########################################
# GRAPHING FUNCTION
########################################

def tsgraph(xdata: Union[List, np.ndarray, Dict], 
                 ydata: Union[List, np.ndarray, Dict], 
                 y2data: Optional[Union[List, np.ndarray, Dict]] = None,
                 format_info: Optional[Dict] = None,
                 save_file: Optional[str] = None):
    """
    Create a customizable graph with flexible formatting options in EA House Style.
    
    Parameters:
    -----------
    xdata : list or array-like
        X-axis data points
        - list if your x-axis is the same for all y series
        - Dict with {label: data} if not. labels must match labels used in ydata and be unique.
    ydata : array-like or dict, optional
        Y-axis data points. Can be:
        - Single list/array for one series
        - Dict with {label: data} for one or multiple series
    y2data: dict, optional
        RHS Y-axis data points. Must be a Dictionary in the form of {label: data} even if there is only one series.
    format_info: dict, optional
    save_file: str, optional
            
        Dictionary containing formatting options:
        
        Text options:
        - 'title': str - Graph title
        - 'subtitle': str - Graph subtitle

        X-axis options
        - 'xlabel': str - X-axis label
        - 'xinterval': Number of periods between x ticks
        - 'xlim': tuple - X-axis limits (min, max)
            min, max can be either string (e.g. '2012-01') or a datetime object ('pd.to_datetime('2022-01-01')')

        Y-axis options
        - 'ylabel': str - Y-axis label
        - 'ytickformat': str - Ylabel format ('pctg' or 'dec')
        - 'xticksize': Xtick interval
        - 'yticksize': Ytick interval
        - 'ydecimals': Y decimal size
        - 'ylim': tuple - Y-axis limits (min, max)

        Second Y-axis options
        - 'y2label': str - Y-axis label
        - 'y2ticksize': Ytick interval
        - 'y2tickformat': str - Ylabel format ('pctg' or 'dec')
        - 'y2decimals': Number of decimal points
        - 'y2lim': tuple - Y-axis limits (min, max)

        Figure options:
        - 'figsize': tuple - Figure size (width, height)

        Plot Style options:
        - 'line_style': str or list - Line style(s) ('-', '--', '-.', ':')
        - 'line_width': float or list - Line width(s)
        - 'colors': str or list - Color(s) for lines
        - 'colors2': str or list - Color(s) for lines on second axis
        - 'line2_style': str or list - Line style(s) ('-', '--', '-.', ':')
        - 'line2_width': float or list - Line width(s)

        Legend options:
        - 'legend': 'on' or 'off'
        - 'legend_ncol': int - number of columns for legend

        Colors:
        macrotools comes with the default EA color palette. You can call them via the color cycler (e.g. 'C2') or reference them by name:
        macrotools.eacolors['colorname'], where 'colorname' is one of 'eablue', 'eaorange', 'eagreen', 'eapurple', 'eayellow', 'eamagenta', 'eaviolet'

    save_file: str - Optional
        If left out, will not save the graph
        If included, will save the graph in png format at the filepath provided
        e.g. 'data/graph.png'

    Returns:
    --------
    fig: matplotlib figure
    
    Examples:
    ---------
    Here is a minimalist example / template you can use for a single series:
    mt.tsgraph(
        xdata = ln_clean.index,
        ydata = {'Prime-age Employment Rate': ln_clean['LNS12300060']},
        format_info = {
            'title': 'Prime-age Employment Rate',
            'xlim': (master.GRAPH_START_DATE, master.GRAPH_END_DATE),
            'xinterval' : 6,
            'ylim': (0.80, 0.81), 'yticksize': 0.002, 'ylabel': 'Ratio', 'ytickformat': 'pctg', 'ydecimals': 1
        },
        save_file = master.FIGURES_DIR + 'paepop.png'
    )

    Here is an exmaple for two series:
    mt.tsgraph(
        xdata = ln_clean.index,
        ydata = {'Men': ln_clean['LNS12300061']},
        y2data = {'Women': ln_clean['LNS12300062']},
        format_info = {
            'title': 'Prime-age Employment Rate, by Gender', 'subtitle': 'Ratio, Employed to Population',
            'xlim': (master.GRAPH_START_DATE, master.GRAPH_END_DATE),
            'xinterval' : 6,
            'ylim': (0.83, 0.88), 'yticksize': 0.01, 'ylabel': 'Men', 'ytickformat': 'pctg', 'ydecimals': 0,        
            'y2lim': (0.73, 0.78), 'y2ticksize': 0.01, 'y2label': 'Woen', 'y2tickformat': 'pctg', 'y2decimals': 0,
            'legend': 'on'
        },
        save_file = master.FIGURES_DIR + 'paepop_gender.png'
    )

    ... Need to create example
    """
    
    ########################################
    # Call stylesheet and font
    ########################################
    stylefile = Path(__file__).parent / 'styles' / 'eastyle.mpstyle'
    fontfile = Path(__file__).parent / 'styles' / 'fonts' / 'Montserrat-Regular.ttf'
    fontprop = font_manager.FontProperties(fname=str(fontfile))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ########################################
    # Default format options
    ########################################

    # Title Positions
    if format_info:
        if 'legend' in format_info and 'subtitle' in format_info:
            def_title_y = 1.2
            def_subtitle_y = 1.12
        elif 'legend' in format_info and 'subtitle' not in format_info:
            def_title_y = 1.1
            def_subtitle_y = 1.03
        else:
            def_title_y = 1.1
            def_subtitle_y = 1.03
    else:
        def_title_y = 1.1
        def_subtitle_y = 1.03

    # Number of Series
    series_count = len(ydata) if isinstance(ydata, dict) else 1
    if y2data: series2_count = len(y2data) if isinstance(y2data, dict) else 1
    total_series = series_count + series2_count if y2data else series_count

    if isinstance(xdata, Dict) and len(xdata)!=total_series:
        raise Exception('Number of xdata series does not match numberf y and y2data series')
    if isinstance(xdata, Dict) and y2data:
        if ydata.keys() & y2data.keys(): raise Exception('You have overlapping series keys in ydata and y2data. Please use unique keys.')

    default_format = {
		'title': '',
		'title_y': def_title_y,
        'title_size': 20,
		'subtitle': '',
		'subtitle_y': def_subtitle_y,
        'subtitle_size': 14,
        'xlabel': '',
        'ylabel': '',
		'ytickformat': 'dec',
		'xticksize': None,
		'yticksize': None,
		'ydecimals': 0,
		'ylim': None,
        'xaxiscross': None,
		'figsize': (9, 5),
        'legend': 'off',
		'legend_loc': 'lower center',
        'legend_ncol': min(4, total_series),
		'line_style': '-',
		'line_width': 2.0,
		'colors': None,
		'xlim': None,
		'xinterval': None,
		'save_path': None,
		'dpi': 500,
        'y2label': '',
        'y2tickformat': 'dec',
        'y2ticksize': None,
        'y2decimals': 0,
        'y2lim': None,
        'colors2': None,
        'line2_style': '-',
        'line2_width': 2.0
    }
    
    # Merge user format_info with defaults
    if format_info is None:
        format_info = {}
    fmt = {**default_format, **format_info}
    
    with plt.style.context(str(stylefile)):

        plt.rcParams['font.family'] = fontprop.get_name()

        ########################################    
        # Figure and Axes
        ########################################
        fig, ax = plt.subplots(figsize=fmt['figsize'])
        if y2data:
            ax2 = fig.axes[0].twinx()

        ########################################    
        # Create Plots
        ########################################
        
        # Convert single values fmt to lists for iteration
        line_styles = [fmt['line_style']] * series_count if isinstance(fmt['line_style'], str) else fmt['line_style']
        line_widths = [fmt['line_width']] * series_count if isinstance(fmt['line_width'], (int, float)) else fmt['line_width']

        # If Colors are not specified, use default stylesheet colors
        if isinstance(fmt['colors'], list):
            colors = fmt['colors']
        elif isinstance(fmt['colors'], str):
            colors = [fmt['colors']] * series_count
        else: colors = None

        # Plot Left-hand Y-axis Series
        if not isinstance(ydata, dict):
            plot_kwargs = {'linestyle': line_styles[0], 'linewidth': line_widths[0]}            
            if colors and colors[0]: plot_kwargs['color'] = colors[i % len(colors)]
            ax.plot(xdata, ydata, **plot_kwargs)

        elif isinstance(ydata, dict):
            for i, (label, y) in enumerate(ydata.items()):
                plot_kwargs = {
                    'label': label,
                    'linestyle': line_styles[i] if i < len(line_styles) else '-',
                    'linewidth': line_widths[i] if i < len(line_widths) else 2
                }
                
                if colors and colors[i]:
                    plot_kwargs['color'] = colors[i % len(colors)]

                if isinstance(xdata, Dict):
                    ax.plot(xdata[label], y, **plot_kwargs)
                else:
                    ax.plot(xdata, y, **plot_kwargs)

        if y2data:

            line2_styles = [fmt['line2_style']] * series2_count if isinstance(fmt['line2_style'], str) else fmt['line2_style']
            line2_widths = [fmt['line2_width']] * series2_count if isinstance(fmt['line2_width'], (int, float)) else fmt['line2_width']

            if isinstance(fmt['colors2'], list):
                colors2 = fmt['colors2']
            elif isinstance(fmt['colors2'], str):
                colors2 = [fmt['colors2']] * series2_count
            else:
                colors2 = style_colors[series_count:min(series_count + series2_count, len(style_colors))]

            for i, (label2, y2) in enumerate(y2data.items()):
                plot_kwargs = {
                    'label': label2,
                    'linestyle': line2_styles[i] if i < len(line2_styles) else '-',
                    'linewidth': line2_widths[i] if i < len(line2_widths) else 2
                }
                
                if colors2 and colors2[i]:
                    plot_kwargs['color'] = colors2[i % len(colors2)]

                if isinstance(xdata, Dict):
                    ax2.plot(xdata[label2], y2, **plot_kwargs)
                else:
                    ax2.plot(xdata, y2, **plot_kwargs)
        
        ########################################
        # Apply Formatting
        ########################################

        # Apply Title Formatting
        if fmt['subtitle']:
        	ax.text(0.5, fmt['title_y'], fmt['title'], fontsize=fmt['title_size'], transform=ax.transAxes, ha='center')
        	ax.text(0.5, fmt['subtitle_y'], fmt['subtitle'], fontsize=fmt['subtitle_size'], transform=ax.transAxes, ha='center')
        else:
        	ax.text(0.5, fmt['title_y'], fmt['title'], fontsize=fmt['title_size'], transform=ax.transAxes, ha='center')

        # Apply axis formating
        if fmt['xlabel']:
            ax.set_xlabel(fmt['xlabel'])
        if fmt['ylabel']:
            ax.set_ylabel(fmt['ylabel'])
        if fmt['xlim']:
            if isinstance(fmt['xlim'][0], str) and isinstance(fmt['xlim'][1], str):
                ax.set_xlim(pd.to_datetime(fmt['xlim'][0]), pd.to_datetime(fmt['xlim'][1]))
            else:
                ax.set_xlim(fmt['xlim'])
        if fmt['ylim']:
            ax.set_ylim(fmt['ylim'])
        if fmt['yticksize']:
            ax.set_yticks(np.arange(fmt['ylim'][0], fmt['ylim'][1] + fmt['yticksize'] / 10, fmt['yticksize']))
        if fmt['ytickformat']=='pctg':
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=fmt['ydecimals']))
        elif fmt['ytickformat']=='dec':
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f'{{x:.{fmt['ydecimals']}f}}'))

        if fmt['xaxiscross'] is not None:
            ax.axhline(y=fmt['xaxiscross'], color='black', linewidth=0.8)
            ax.spines['bottom'].set_position(('outward', 0))

        # Will need to generalize this to quarters and years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        if fmt['xinterval']:
        	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=fmt['xinterval']))

        # Apply Second Y-axis Formating
        if y2data:
            ax2.spines[['right']].set_visible(True)
            if fmt['y2lim']:
                ax2.set_ylim(fmt['y2lim'])
            if fmt['y2label']:
                ax2.set_ylabel(fmt['y2label'])
            if fmt['y2tickformat']=='pctg': 
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=fmt['y2decimals']))
            elif fmt['y2tickformat']=='dec':
                ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            if fmt['y2ticksize']:
                ax2.set_yticks(np.arange(fmt['y2lim'][0], fmt['y2lim'][1] + fmt['y2ticksize'] / 10, fmt['y2ticksize']))

        # Apply Legend
        if fmt['legend']=='on':
            if y2data:
                h1, l1 = fig.axes[0].get_legend_handles_labels()
                h2, l2 = fig.axes[1].get_legend_handles_labels()
                ax.legend(h1+h2, l1+l2, bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'], frameon=False)
            else:
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'], frameon=False)
        
        # Save Figure
        if save_file:
            plt.savefig(save_file, bbox_inches='tight', edgecolor = fig.get_edgecolor(), dpi=500)

        return fig

# Annualized Growth Rate Function - by default assume monthly data
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
