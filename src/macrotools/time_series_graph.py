from typing import Union, List, Dict, Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.ticker as mtick

import pandas as pd
import datetime as dt
import warnings
from dateutil.relativedelta import relativedelta
from PIL import Image
from pathlib import Path

########################################
# COLORS
########################################

# Apply stylesheet
stylefile = Path(__file__).parent / 'styles' / 'eastyle.mplstyle'
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

def tsgraph(ydata: Union[List, np.ndarray, Dict], 
            y2data: Optional[Union[List, np.ndarray, Dict]] = None,
            xdata: Optional[Union[List, np.ndarray, Dict]] = None, 
            format_info: Optional[Dict] = None,
            save_file: Optional[str] = None):
    """
    Create a customizable graph with flexible formatting options in EA House Style.
    
    Parameters:
    -----------
    ydata : array-like or dict, optional
        Y-axis data points. Can be:
        - Single list/array for one series
        - Dict with {label: data} for one or multiple series
    y2data: dict, optional
        RHS Y-axis data points. Must be a Dictionary in the form of {label: data} even if there is only one series.
    xdata : list or array-like, optional
        X-axis data points
        - list if your x-axis is the same for all y series
        - Dict with {label: data} if not. labels must match labels used in ydata and be unique.
        - Optional argument--if you pass a DataFrame or Series, will use th index.
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

    Here is an example for two series:
    mt.tsgraph(
        xdata = ln_clean.index,
        ydata = {'Men': ln_clean['LNS12300061']},
        y2data = {'Women': ln_clean['LNS12300062']},
        format_info = {
            'title': 'Prime-age Employment Rate, by Gender', 'subtitle': 'Ratio, Employed to Population',
            'xlim': (master.GRAPH_START_DATE, master.GRAPH_END_DATE),
            'xinterval' : 6,
            'ylim': (0.83, 0.88), 'yticksize': 0.01, 'ylabel': 'Men', 'ytickformat': 'pctg', 'ydecimals': 0,        
            'y2lim': (0.73, 0.78), 'y2ticksize': 0.01, 'y2label': 'Women', 'y2tickformat': 'pctg', 'y2decimals': 0,
            'legend': 'on'
        },
        save_file = master.FIGURES_DIR + 'paepop_gender.png'
    )
    """
    
    ########################################
    # Call stylesheet and font
    ########################################
    stylefile = Path(__file__).parent / 'styles' / 'eastyle.mplstyle'
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

    # Number of Series: if DataFrame, use number of columns; if Dictionary, length, else, 1.
    if isinstance(ydata, dict):
        series_count = len(ydata)
    elif isinstance(ydata, pd.DataFrame):
        series_count = ydata.shape[1]
    else:
        series_count = 1
    
    if y2data is not None:
        if isinstance(y2data, dict):
            series2_count = len(y2data)
        elif isinstance(y2data, pd.DataFrame):
            series2_count = y2data.shape[1]
        else:
            series2_count = 1

    total_series = series_count + series2_count if y2data is not None else series_count

    # xdata - check that the number of xdata inputs is correct
    if xdata is not None:
        if isinstance(xdata, Dict) and len(xdata)!=total_series:
            raise Exception('Number of xdata series does not match number of y and y2data series')
        if isinstance(xdata, Dict) and y2data:
            if ydata.keys() & y2data.keys(): raise Exception('You have overlapping series keys in ydata and y2data. Please use unique keys.')

    # Formatting info
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
        'ydecimals': None,
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
        'xfreq': 'M',
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
        if y2data is not None:
            ax2 = fig.axes[0].twinx()

        ########################################    
        # Create Plots
        ########################################

        # Plot Data
        if y2data is not None:
            data_list = [ydata, y2data]
        else:
            data_list = [ydata]

        for i, (data) in enumerate(data_list):

            if i==0:
                sfx = ''
                plotaxs = ax
            elif i==1:
                sfx = '2'
                plotaxs = ax2

            # Create list of line styles, line widths
            line_styles = [fmt[f'line{sfx}_style']] * series_count if isinstance(fmt[f'line{sfx}_style'], str) else fmt[f'line{sfx}_style']
            line_widths = [fmt[f'line{sfx}_width']] * series_count if isinstance(fmt[f'line{sfx}_width'], (int, float)) else fmt[f'line{sfx}_width']

            # Create list of colors. If colors are not specified, use stylesheet colors
            if isinstance(fmt[f'colors{sfx}'], list):
                colors = fmt[f'colors{sfx}']
            elif isinstance(fmt['colors{sfx}'], str):
                colors = [fmt[f'colors{sfx}']] * series_count
            else:
                if i==0: colors = None
                if i==1: colors = style_colors[series_count:min(series_count + series2_count, len(style_colors))]

            # Obtain Graph limits -- convert to datetime if necessary
            if fmt['xlim'] is not None:
                xlim_min = pd.to_datetime(fmt['xlim'][0]) if isinstance(fmt['xlim'][0], str) else fmt['xlim'][0]
                xlim_max = pd.to_datetime(fmt['xlim'][1]) if isinstance(fmt['xlim'][1], str) else fmt['xlim'][1]

            # For DataFrame inputs, plot each column and label according to
            if isinstance(data, pd.DataFrame):

                for j, (col) in enumerate(data.columns):

                    plot_kwargs = {
                        'label': col,
                        'linestyle': line_styles[j] if j < len(line_styles) else '-',
                        'linewidth': line_widths[j] if j < len(line_widths) else 2
                    }
                    if colors and colors[j]:
                        plot_kwargs['color'] = colors[j % len(colors)]

                    # Mask data -- missing and xlims
                    mask = data[col].notna()
                    if xdata is not None:
                        if isinstance(xdata, Dict):
                            xdata_plot = xdata[col]
                        else:
                            xdata_plot = xdata
                    else:
                        xdata_plot = data.index
                    if fmt['xlim'] is not None:
                        mask = mask & (xdata_plot >= xlim_min) & (xdata_plot <= xlim_max)

                    # Plot data
                    plotaxs.plot(xdata_plot[mask], data[col][mask], **plot_kwargs)

            # For dictionary inputs, plot input with the label provided
            elif isinstance(data, dict):

                for j, (label, y) in enumerate(data.items()):

                    plot_kwargs = {
                        'label': label,
                        'linestyle': line_styles[j] if j < len(line_styles) else '-',
                        'linewidth': line_widths[j] if j < len(line_widths) else 2
                    }

                    if colors and colors[j]:
                        plot_kwargs['color'] = colors[j % len(colors)]

                    # Mask data -- missing and xlims
                    if isinstance(y, pd.Series):
                        mask = y.notna()
                    else:
                        mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y])

                    if fmt['xlim'] is not None:
                        if xdata is not None:
                            if isinstance(xdata, Dict):
                                xdata_to_mask = xdata[label]
                            else:
                                xdata_to_mask = xdata
                        else:
                            if isinstance(y, pd.Series):
                                xdata_to_mask = y.index
                            else:
                                xdata_to_mask = None

                        if xdata_to_mask is not None:
                            mask = mask & (xdata_to_mask >= xlim_min) & (xdata_to_mask <= xlim_max)

                    if xdata is not None:
                        if isinstance(xdata, Dict):
                            plotaxs.plot(xdata[label][mask], y[mask] if isinstance(y, pd.Series) else np.array(y)[mask], **plot_kwargs)
                        else:
                            plotaxs.plot(xdata[mask] if hasattr(xdata, '__getitem__') else xdata, y[mask] if isinstance(y, pd.Series) else np.array(y)[mask], **plot_kwargs)
                    else:
                        if isinstance(y, pd.Series):
                            plotaxs.plot(y.index[mask], y[mask], **plot_kwargs)
                        else:
                            raise Exception('No x-data detected. Did you mean to pass a pd.Series object? Or an xdata argument?')

            # For a single data series input, plot data labeled with the data series column
            elif isinstance(data, pd.Series):

                plot_kwargs = {'label': data.name, 'linestyle': line_styles[0], 'linewidth': line_widths[0]}

                if colors and colors[0]: plot_kwargs['color'] = colors[0]
                
                # Mask data
                mask = data.notna()
                if xdata is not None:
                    xdata_plot = xdata
                else:
                    xdata_plot = data.index
                if fmt['xlim'] is not None:
                    mask = mask & (xdata_plot >= xlim_min) & (xdata_plot <= xlim_max)

                plotaxs.plot(xdata_plot[mask], data[mask], **plot_kwargs)

            # Otherwise, plot the data
            else:

                plot_kwargs = {'linestyle': line_styles[0], 'linewidth': line_widths[0]}

                if colors and colors[0]: plot_kwargs['color'] = colors[0]
                if xdata is not None:
                    mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in data])
                    if fmt['xlim'] is not None:
                        mask = mask & (xdata >= xlim_min) & (xdata <= xlim_max)
                    plotaxs.plot(xdata[mask] if hasattr(xdata, '__getitem__') else xdata, np.array(data)[mask], **plot_kwargs)
                else:
                    raise Exception('No x-data detected. Did you mean to pass a pd.Series object? Or an xdata argument?')
        
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
            if fmt['ydecimals'] is None:
                ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
            else:
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f'{{x:,.{fmt["ydecimals"]}f}}'))

        if fmt['xaxiscross'] is not None:
            ax.axhline(y=fmt['xaxiscross'], color='black', linewidth=0.8)
            ax.spines['bottom'].set_position(('outward', 0))

        # Format x-axis based on data frequency
        if fmt['xfreq'] == 'Y':
            # Annual data format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if fmt['xinterval']:
                ax.xaxis.set_major_locator(mdates.YearLocator(fmt['xinterval']))
        elif fmt['xfreq'] == 'M':
            # Monthly data format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
            if fmt['xinterval']:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=fmt['xinterval']))

        # Apply Second Y-axis Formating
        if y2data is not None:
            ax2.spines[['right']].set_visible(True)
            if fmt['y2lim']:
                ax2.set_ylim(fmt['y2lim'])
            if fmt['y2label']:
                ax2.set_ylabel(fmt['y2label'])
            if fmt['y2tickformat']=='pctg': 
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=fmt['y2decimals']))
            elif fmt['y2tickformat']=='dec':
                if fmt['y2decimals'] is None:
                    ax2.yaxis.set_major_formatter(mtick.ScalarFormatter())
                else:
                    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter(f'{{x:,.{fmt["y2decimals"]}f}}'))
            if fmt['y2ticksize']:
                ax2.set_yticks(np.arange(fmt['y2lim'][0], fmt['y2lim'][1] + fmt['y2ticksize'] / 10, fmt['y2ticksize']))

        # Apply Legend
        if fmt['legend']=='on':
            if y2data is not None:
                h1, l1 = fig.axes[0].get_legend_handles_labels()
                h2, l2 = fig.axes[1].get_legend_handles_labels()
                ax.legend(h1+h2, l1+l2, bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'], frameon=False)
            else:
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'], frameon=False)
        
        # Save Figure
        if save_file:
            plt.savefig(save_file, bbox_inches='tight', edgecolor = fig.get_edgecolor(), dpi=500)

        return fig