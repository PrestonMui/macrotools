from typing import Union, List, Dict, Optional
import numpy as np
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.ticker as mtick

import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from PIL import Image
from pathlib import Path

########################################
# STYLES
########################################

_styles_dir = Path(__file__).parent / 'styles'
_default_stylefile = _styles_dir / 'defaultstyle.mplstyle'
_ea_stylefile = _styles_dir / 'eastyle.mplstyle'

# Apply default stylesheet at module level
plt.style.use(str(_default_stylefile))

########################################
# COLORS
########################################

# Default color palette (colorblind-friendly, cool-leaning)
default_colors = {
    'charcoal':     '#3C3C3C',
    'steel':        '#4A6FA5',
    'coral':        '#D64545',
    'dark_teal':    '#1B7C6F',
    'amber':        '#D4860B',
    'slate_purple': '#6E5E8B',
    'graphite':     '#6B7D8A',
}

# Default alert palette
default_alert_colors = {
    'alert_red':   '#E03131',
    'alert_blue':  '#1C7ED6',
    'alert_green': '#2F9E44',
    'alert_gold':  '#F08C00',
}

# EA color palette
class _EAColors(dict):
    """Color dict with deprecation warnings for legacy keys."""
    _deprecated = {
        'eaorange': ('alert_orange', '#FF591F'),
        'eayellow': ('alert_yellow', '#EAC14B'),
        'eamagenta': ('alert_magenta', '#B20066'),
        'eaviolet': ('alert_violet', '#8A2B9C'),
    }
    def __getitem__(self, key):
        if key in self._deprecated:
            new_name, color = self._deprecated[key]
            warnings.warn(
                f"eacolors['{key}'] is deprecated. "
                f"Use ea_alert_colors['{new_name}'] instead.",
                DeprecationWarning, stacklevel=2
            )
            return color
        return super().__getitem__(key)
    def __contains__(self, key):
        return key in self._deprecated or super().__contains__(key)

eacolors = _EAColors({
    'steel_blue': '#29679B',
    'green': '#008A6A',
    'dark_purple': '#2E2A73',
    'bright_blue': '#40B2FF',
    'slate': '#64748B',
    'burgundy': '#7A3B4E',
    'warm_gray': '#8C7E74',
    # Legacy aliases
    'eablue': '#29679B',
    'eagreen': '#008A6A',
    'eapurple': '#2E2A73',
})

# EA alert palette — reserved for thresholds, shock markers, annotations
ea_alert_colors = {
    'alert_orange': '#FF591F',
    'alert_magenta': '#B20066',
    'alert_violet': '#8A2B9C',
    'alert_yellow': '#EAC14B',
}

########################################
# GRAPHING FUNCTION
########################################

def tsgraph(ydata: Union[List, np.ndarray, Dict],
            y2data: Optional[Union[List, np.ndarray, Dict]] = None,
            xdata: Optional[Union[List, np.ndarray, Dict]] = None,
            format_info: Optional[Dict] = None,
            style: Optional[Union[str, Dict]] = 'default',
            save_file: Optional[str] = None):
    """
    Create a customizable time series graph.

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
        - 'bgcolor': str - Background color for figure and axes (e.g. 'white', '#FFFFFF'). Defaults to style color.

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

        Logo and footnote options:
        - 'logo': str - Path to a logo image file (PNG recommended). Placed in the bottom-right corner.
        - 'logo_scale': float - Height of logo as fraction of figure height (default: 0.06)
        - 'footnote': str - Footnote/attribution text in the bottom-left corner (e.g. 'Source: BLS')
        - 'footnote_fontsize': float - Font size for footnote text (default: 8)

        Colors:
        macrotools comes with two built-in colorblind-friendly color palette:
        - Default: macrotools.default_colors['colorname'] (charcoal, steel, coral, dark_teal, amber, slate_purple, graphite)
        - EA: macrotools.eacolors['colorname'] (steel_blue, green, dark_purple, bright_blue, slate, burgundy, warm_gray)
        Alert colors: macrotools.default_alert_colors or macrotools.ea_alert_colors
    
    style: str or dict, optional
        Style to use for the graph. Options:
        - 'default' (default): Clean, generic style with DejaVu Sans font
        - 'ea': Employ America house style with Montserrat/Lato fonts
        - Path to a .mplstyle file: Use a custom matplotlib stylesheet
        - Dict of rcParams: Apply overrides on top of the default style

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

    Here is an example using EA style:
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
        style = 'ea',
        save_file = master.FIGURES_DIR + 'paepop_gender.png'
    )
    """

    ########################################
    # Resolve style and fonts
    ########################################
    use_ea_fonts = False

    if isinstance(style, dict):
        # Dict of rcParams overrides on top of default style
        stylefile = str(_default_stylefile)
        style_overrides = style
    elif isinstance(style, str) and style.lower() == 'ea':
        stylefile = str(_ea_stylefile)
        style_overrides = None
        use_ea_fonts = True
        # Register EA fonts
        fontfile = _styles_dir / 'fonts' / 'Montserrat-Regular.ttf'
        fontfile_bold = _styles_dir / 'fonts' / 'Lato-Bold.ttf'
        font_manager.fontManager.addfont(str(fontfile))
        font_manager.fontManager.addfont(str(fontfile_bold))
        fontprop = font_manager.FontProperties(fname=str(fontfile))
        fontprop_bold = font_manager.FontProperties(fname=str(fontfile_bold), size=plt.rcParams['axes.labelsize'])
    elif isinstance(style, str) and Path(style).is_file():
        # Custom stylesheet path
        stylefile = style
        style_overrides = None
    else:
        # Default style
        stylefile = str(_default_stylefile)
        style_overrides = None

    ########################################
    # Default format options
    ########################################

    # Title Positions
    if format_info and 'legend' in format_info and 'subtitle' in format_info:
        def_title_y = 1.2
        def_subtitle_y = 1.12
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
    
    series2_count = 0
    if y2data is not None:
        if isinstance(y2data, dict):
            series2_count = len(y2data)
        elif isinstance(y2data, pd.DataFrame):
            series2_count = y2data.shape[1]
        else:
            series2_count = 1

    total_series = series_count + series2_count

    # xdata - check that the number of xdata inputs is correct
    if xdata is not None:
        if isinstance(xdata, dict) and len(xdata)!=total_series:
            raise Exception('Number of xdata series does not match number of y and y2data series')
        if isinstance(xdata, dict) and y2data:
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
        'figsize': None,
        'legend': 'off',
        'legend_loc': 'lower center',
        'legend_ncol': min(4, total_series),
        'line_style': '-',
        'line_width': plt.rcParams['lines.linewidth'],
        'colors': None,
        'xlim': None,
        'xfreq': 'M',
        'xinterval': None,
        'save_path': None,
        'y2label': '',
        'y2tickformat': 'dec',
        'y2ticksize': None,
        'y2decimals': 0,
        'y2lim': None,
        'colors2': None,
        'line2_style': '-',
        'line2_width': plt.rcParams['lines.linewidth'],
        'bgcolor': None,
        'logo': None,
        'logo_scale': 0.06,
        'footnote': None,
        'footnote_fontsize': 8
    }
    
    # Merge user format_info with defaults
    if format_info is None:
        format_info = {}
    fmt = {**default_format, **format_info}
    
    # Build style context: stylesheet + optional rcParams overrides
    style_context = [stylefile]
    if style_overrides:
        style_context.append(style_overrides)

    with plt.style.context(style_context):

        # Extract active color cycle for this style
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if use_ea_fonts:
            plt.rcParams['font.family'] = fontprop.get_name()

        ########################################    
        # Figure and Axes
        ########################################
        subplots_kwargs = {}
        if fmt['figsize'] is not None:
            subplots_kwargs['figsize'] = fmt['figsize']
        fig, ax = plt.subplots(**subplots_kwargs)
        if fmt['bgcolor'] is not None:
            fig.set_facecolor(fmt['bgcolor'])
            ax.set_facecolor(fmt['bgcolor'])
        if y2data is not None:
            ax2 = fig.axes[0].twinx()
            ax2.grid(False)

        ########################################
        # Per-axis grid styling
        ########################################
        ax.xaxis.grid(True, color='#D4D4D4', linewidth=0.25, alpha=1.0)

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
                line_styles = [fmt['line_style']] * series_count if isinstance(fmt['line_style'], str) else fmt[f'line_style']
                line_widths = [fmt[f'line_width']] * series_count if isinstance(fmt[f'line_width'], (int, float)) else fmt[f'line{sfx}_width']
            elif i==1:
                sfx = '2'
                plotaxs = ax2
                line_styles = [fmt['line2_style']] * series2_count if isinstance(fmt[f'line2_style'], str) else fmt[f'line2_style']
                line_widths = [fmt[f'line2_width']] * series2_count if isinstance(fmt[f'line2_width'], (int, float)) else fmt[f'line2_width']

            # Create list of colors. If colors are not specified, use stylesheet colors
            if isinstance(fmt[f'colors{sfx}'], list):
                colors = fmt[f'colors{sfx}']
            elif isinstance(fmt[f'colors{sfx}'], str):
                current_count = series_count if i == 0 else series2_count
                colors = [fmt[f'colors{sfx}']] * current_count
            else:
                if i==0: colors = None
                if i==1: colors = style_colors[series_count:min(series_count + series2_count, len(style_colors))]

            # Obtain Graph limits -- convert to datetime if necessary
            if fmt['xlim'] is not None:
                xlim_min = pd.to_datetime(fmt['xlim'][0]) if isinstance(fmt['xlim'][0], str) else fmt['xlim'][0]
                xlim_max = pd.to_datetime(fmt['xlim'][1]) if isinstance(fmt['xlim'][1], str) else fmt['xlim'][1]

            # For DataFrame inputs, plot each column and label; use index as xdata unless otherwise provided.
            if isinstance(data, pd.DataFrame):

                for j, (col) in enumerate(data.columns):

                    plot_kwargs = {
                        'label': col,
                        'linestyle': line_styles[j] if j < len(line_styles) else '-',
                        'linewidth': line_widths[j] if j < len(line_widths) else 2
                    }
                    if colors:
                        plot_kwargs['color'] = colors[j % len(colors)]

                    # Mask data -- missing and xlims
                    mask = data[col].notna()
                    if xdata is not None:
                        if isinstance(xdata, dict):
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
                            if isinstance(xdata, dict):
                                xdata_plot = xdata[label]
                            else:
                                xdata_plot = xdata
                        else:
                            if isinstance(y, pd.Series):
                                xdata_plot = y.index
                            else:
                                raise Exception('No x-data detected. Did you mean to pass a pd.Series object? Or an xdata argument?')

                        if xdata_plot is not None:
                            mask = mask & (xdata_plot >= xlim_min) & (xdata_plot <= xlim_max)

                    ydata_plot = y if isinstance(y, pd.Series) else np.array(y)

                    plotaxs.plot(xdata_plot[mask], ydata_plot[mask], **plot_kwargs)                           

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

        title_x = 0.0

        # Apply Logo (bottom-right) and Source text (bottom-left)
        logo_h = fmt['logo_scale'] if fmt['logo'] is not None else 0.04
        if fmt['logo'] is not None:
            logo_img = Image.open(fmt['logo'])
            logo_aspect = logo_img.width / logo_img.height
            pos = ax.get_position()
            logo_w = logo_h * logo_aspect * (fig.get_size_inches()[1] / fig.get_size_inches()[0])
            logo_x = pos.x0 + pos.width - logo_w
            logo_y = 0.0
            logo_ax = fig.add_axes([logo_x, logo_y, logo_w, logo_h])
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')

        if fmt['footnote'] is not None:
            fn_pos = ax.get_position()

            # Calculate available width in pixels
            if fmt['logo'] is not None:
                avail_width_frac = logo_x - fn_pos.x0 - 0.02
            else:
                avail_width_frac = fn_pos.width
            avail_px = avail_width_frac * fig.get_size_inches()[0] * fig.dpi

            # Renderer-based word wrapping (respects manual \n)
            renderer = fig.canvas.get_renderer()
            fn_segments = fmt['footnote'].split('\n')
            wrapped_lines = []
            for segment in fn_segments:
                words = segment.split()
                if not words:
                    wrapped_lines.append('')
                    continue
                current_line = words[0]
                for word in words[1:]:
                    candidate = current_line + ' ' + word
                    t = fig.text(0, 0, candidate, fontsize=fmt['footnote_fontsize'], fontstyle='italic')
                    w = t.get_window_extent(renderer).width
                    t.remove()
                    if w > avail_px:
                        wrapped_lines.append(current_line)
                        current_line = word
                    else:
                        current_line = candidate
                wrapped_lines.append(current_line)
            wrapped_fn = '\n'.join(wrapped_lines)

            fig.text(fn_pos.x0, logo_h / 2, wrapped_fn,
                     fontsize=fmt['footnote_fontsize'], color='#666666',
                     va='center', ha='left', fontstyle='italic')

        # Apply Title Formatting
        title_kwargs = {'fontweight': 'bold'}
        if use_ea_fonts:
            title_kwargs['fontproperties'] = fontprop_bold
        ax.text(title_x, fmt['title_y'], fmt['title'], fontsize=fmt['title_size'],
                transform=ax.transAxes, ha='left', **title_kwargs)
        if fmt['subtitle']:
            ax.text(title_x, fmt['subtitle_y'], fmt['subtitle'], fontsize=fmt['subtitle_size'],
                    transform=ax.transAxes, ha='left')

        # Apply axis formating
        if fmt['xlabel']:
            ax.set_xlabel(fmt['xlabel'])
        if fmt['ylabel']:
            ylabel_kwargs = {}
            if y2data is not None:
                if isinstance(fmt['colors'], list):
                    ylabel_kwargs['color'] = fmt['colors'][0]
                elif isinstance(fmt['colors'], str):
                    ylabel_kwargs['color'] = fmt['colors']
                else:
                    ylabel_kwargs['color'] = style_colors[0]
            ylabel_font_kwargs = {'fontweight': 'bold'}
            if use_ea_fonts:
                ylabel_font_kwargs['fontproperties'] = fontprop_bold
            ax.set_ylabel(fmt['ylabel'], **ylabel_font_kwargs, **ylabel_kwargs)
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
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
            if fmt['xinterval']:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=fmt['xinterval']))

        # Apply Second Y-axis Formating
        if y2data is not None:
            ax2.spines[['right']].set_visible(True)
            ax2.spines['right'].set_linewidth(1.0)
            ax2.spines['right'].set_color('#C8C8C8')
            if fmt['y2lim']:
                ax2.set_ylim(fmt['y2lim'])
            if fmt['y2label']:
                if isinstance(fmt['colors2'], list):
                    y2label_color = fmt['colors2'][0]
                elif isinstance(fmt['colors2'], str):
                    y2label_color = fmt['colors2']
                else:
                    y2label_color = style_colors[series_count] if series_count < len(style_colors) else style_colors[0]
                y2label_font_kwargs = {'fontweight': 'bold'}
                if use_ea_fonts:
                    y2label_font_kwargs['fontproperties'] = fontprop_bold
                ax2.set_ylabel(fmt['y2label'], color=y2label_color, **y2label_font_kwargs)
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
                ax.legend(h1+h2, l1+l2, bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'])
            else:
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc=fmt['legend_loc'], ncol = fmt['legend_ncol'])
        
        # Save Figure
        if save_file:
            plt.savefig(save_file, bbox_inches='tight', edgecolor = fig.get_edgecolor())

        return fig