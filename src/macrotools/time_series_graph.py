from typing import Union, Dict, Optional
import numpy as np
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.ticker as mtick

import pandas as pd
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

def tsgraph(series,
            xaxis: Optional[Dict] = None,
            yaxis: Optional[Dict] = None,
            yaxis_rhs: Optional[Dict] = None,
            title: Optional[Dict] = None,
            legend: Optional[Dict] = None,
            footnote: Optional[Dict] = None,
            hline=None,
            style: Optional[Union[str, Dict]] = 'default',
            figsize: Optional[tuple] = None,
            bgcolor: Optional[str] = None,
            save_path: Optional[str] = None):
    """
    Create a customizable time series graph.

    Parameters:
    -----------
    series : array-like, dict, pd.DataFrame, or list
        Data to plot. Accepts:
        - A bare value (pd.Series, np.ndarray, list) for a single series on the left axis
        - A pd.DataFrame — each column becomes a series (label=column name, x=index)
        - A dict describing one series, with keys:
            - 'y': (required) data to plot (pd.Series, array, list)
            - 'x': x-axis values; defaults to y.index if y is a pd.Series
            - 'label': legend label
            - 'axis': 'left' (default) or 'right'
            - 'color': line color; auto-assigned from style cycle if omitted
            - 'linestyle': '-', '--', '-.', or ':' (default '-')
            - 'linewidth': float (default from style)
        - A list of any combination of the above

    xaxis : dict, optional
        - 'label': str — axis label
        - 'lim': tuple — (min, max), strings or datetime objects
        - 'interval': int — number of periods between ticks
        - 'freq': 'M' (default) or 'Y' — tick frequency

    yaxis : dict, optional
        - 'label': str — axis label
        - 'lim': tuple — (min, max)
        - 'ticksize': float — tick interval
        - 'tickformat': 'dec' (default) or 'pctg'
        - 'decimals': int — decimal places

    yaxis_rhs : dict, optional
        Same keys as yaxis. Used when any series has axis='right'.

    title : dict, optional
        - 'title': str — graph title
        - 'subtitle': str — subtitle text
        - 'title_size': float (default 20)
        - 'subtitle_size': float (default 14)

    legend : dict, optional
        - 'show': bool (default False)
        - 'loc': str (default 'lower center')
        - 'ncol': int (default min(4, total_series))

    footnote : dict, optional
        - 'text': str — footnote text (e.g. 'Source: BLS')
        - 'fontsize': float (default 8)
        - 'logo': str — path to logo image file
        - 'logo_scale': float — logo height as fraction of figure height (default 0.07)

    hline : numeric, dict, or list of dicts, optional
        Draw horizontal reference line(s).
        - numeric: draws a line at that y-value with default styling
        - dict with keys:
          - 'y': (required) y-value for the line
          - 'color': str (default '#3C3C3C')
          - 'linewidth': float (default 1.0)
          - 'linestyle': str (default '-')
          - 'legend_label': str (default None)
          - 'callout': str — text annotation on the line (default None)
          - 'callout_align': 'above' or 'below' (default 'above')
          - 'callout_pos_x': float — 0.0 (left) to 1.0 (right) (default 1.0)
          - 'callout_fontsize': float (default 9)
        - list: draws multiple horizontal lines

    style : str or dict, optional
        - 'default': Clean, generic style with DejaVu Sans font
        - 'ea': Employ America house style with Montserrat/Lato fonts
        - Path to a .mplstyle file
        - Dict of rcParams overrides

    figsize : tuple, optional
        Figure size (width, height)

    bgcolor : str, optional
        Background color for figure and axes

    save_path : str, optional
        Filepath to save the graph as PNG (e.g. 'data/graph.png')

    Returns:
    --------
    fig : matplotlib figure

    Examples:
    ---------
    Single series:
    mt.tsgraph(
        series=ln_clean['LNS12300060'],
        xaxis={'lim': ('2019-01', '2025-06'), 'interval': 6},
        yaxis={'label': 'Ratio', 'lim': (0.80, 0.81), 'ticksize': 0.002,
               'tickformat': 'pctg', 'decimals': 1},
        title={'title': 'Prime-age Employment Rate'},
    )

    Dual-axis with EA style:
    mt.tsgraph(
        series=[
            {'y': ln_clean['LNS12300061'], 'label': 'Men', 'axis': 'left'},
            {'y': ln_clean['LNS12300062'], 'label': 'Women', 'axis': 'right'},
        ],
        xaxis={'lim': ('2019-01', '2025-06'), 'interval': 6},
        yaxis={'label': 'Men', 'lim': (0.83, 0.88), 'ticksize': 0.01,
               'tickformat': 'pctg', 'decimals': 0},
        yaxis_rhs={'label': 'Women', 'lim': (0.73, 0.78), 'ticksize': 0.01,
                   'tickformat': 'pctg', 'decimals': 0},
        title={'title': 'Prime-age Employment Rate, by Gender',
               'subtitle': 'Ratio, Employed to Population'},
        legend={'show': True},
        style='ea',
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
    # Normalize series input
    ########################################

    if isinstance(series, list):
        raw_list = series
    else:
        raw_list = [series]

    normalized = []
    for entry in raw_list:
        if isinstance(entry, pd.DataFrame):
            for col in entry.columns:
                normalized.append({'y': entry[col], 'x': entry.index, 'label': col,
                                   'axis': 'left', 'color': None, 'linestyle': None, 'linewidth': None})
        elif isinstance(entry, dict):
            if 'y' not in entry:
                raise ValueError("Series dict must contain a 'y' key with the data to plot.")
            normalized.append({
                'y': entry['y'], 'x': entry.get('x'), 'label': entry.get('label'),
                'axis': entry.get('axis', 'left'), 'color': entry.get('color'),
                'linestyle': entry.get('linestyle'), 'linewidth': entry.get('linewidth'),
            })
        elif isinstance(entry, pd.Series):
            normalized.append({'y': entry, 'x': None, 'label': entry.name,
                               'axis': 'left', 'color': None, 'linestyle': None, 'linewidth': None})
        else:
            normalized.append({'y': entry, 'x': None, 'label': None,
                               'axis': 'left', 'color': None, 'linestyle': None, 'linewidth': None})

    total_series = len(normalized)
    has_right_axis = any(s['axis'] == 'right' for s in normalized)

    ########################################
    # Merge option dicts with defaults
    ########################################

    xo = {**{'label': '', 'lim': None, 'interval': None, 'freq': 'M'}, **(xaxis or {})}
    yo = {**{'label': '', 'lim': None, 'ticksize': None, 'tickformat': 'dec', 'decimals': None}, **(yaxis or {})}
    yr = {**{'label': '', 'lim': None, 'ticksize': None, 'tickformat': 'dec', 'decimals': 0}, **(yaxis_rhs or {})}

    has_legend = (legend or {}).get('show', False)
    has_subtitle = bool((title or {}).get('subtitle'))
    def_title_y = 1.2 if (has_legend and has_subtitle) else 1.1
    def_subtitle_y = 1.12 if (has_legend and has_subtitle) else 1.03

    to = {**{'title': '', 'subtitle': '', 'title_size': 20, 'subtitle_size': 14,
             'title_y': def_title_y, 'subtitle_y': def_subtitle_y}, **(title or {})}
    lo = {**{'show': False, 'loc': 'lower center', 'ncol': min(4, total_series)}, **(legend or {})}
    fo = {**{'text': None, 'fontsize': 8, 'logo': None, 'logo_scale': 0.07}, **(footnote or {})}
    
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
        if figsize is not None:
            subplots_kwargs['figsize'] = figsize
        fig, ax = plt.subplots(**subplots_kwargs)
        if bgcolor is not None:
            fig.set_facecolor(bgcolor)
            ax.set_facecolor(bgcolor)
        if has_right_axis:
            ax2 = ax.twinx()
            ax2.grid(False)

        ########################################
        # Per-axis grid styling
        ########################################
        ax.xaxis.grid(True, color='#D4D4D4', linewidth=0.25, alpha=1.0)

        ########################################
        # Auto-assign colors from style cycle
        ########################################
        color_idx = 0
        for s in normalized:
            if s['color'] is None:
                s['color'] = style_colors[color_idx % len(style_colors)]
                color_idx += 1

        left_colors = [s['color'] for s in normalized if s['axis'] == 'left']
        right_colors = [s['color'] for s in normalized if s['axis'] == 'right']

        ########################################
        # Create Plots
        ########################################

        # Precompute xlim bounds
        if xo['lim'] is not None:
            xlim_min = pd.to_datetime(xo['lim'][0]) if isinstance(xo['lim'][0], str) else xo['lim'][0]
            xlim_max = pd.to_datetime(xo['lim'][1]) if isinstance(xo['lim'][1], str) else xo['lim'][1]

        for s in normalized:
            target_ax = ax2 if s['axis'] == 'right' else ax
            y = s['y']

            # Resolve x-data
            x = s['x'] if s['x'] is not None else (y.index if isinstance(y, pd.Series) else None)
            if x is None:
                raise ValueError('No x-data for series. Pass x in the series dict or use a pd.Series.')

            # Build plot kwargs
            plot_kwargs = {
                'linestyle': s['linestyle'] or '-',
                'linewidth': s['linewidth'] or plt.rcParams['lines.linewidth'],
                'color': s['color'],
            }
            if s['label'] is not None:
                plot_kwargs['label'] = s['label']

            # NaN mask
            if isinstance(y, pd.Series):
                mask = y.notna()
            else:
                y = np.array(y)
                mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y])

            # xlim mask
            if xo['lim'] is not None:
                mask = mask & (x >= xlim_min) & (x <= xlim_max)

            target_ax.plot(x[mask], y[mask], **plot_kwargs)

        ########################################
        # Apply Formatting
        ########################################

        title_x = 0.0

        # Apply Logo (bottom-right) and Footnote (bottom-left)
        logo_h = fo['logo_scale'] if fo['logo'] is not None else 0.04
        if fo['logo'] is not None:
            logo_img = Image.open(fo['logo'])
            logo_aspect = logo_img.width / logo_img.height
            pos = ax.get_position()
            logo_w = logo_h * logo_aspect * (fig.get_size_inches()[1] / fig.get_size_inches()[0])
            logo_x = pos.x0 + pos.width - logo_w
            logo_y = 0.0
            logo_ax = fig.add_axes([logo_x, logo_y, logo_w, logo_h])
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')

        if fo['text'] is not None:
            fn_pos = ax.get_position()

            # Calculate available width in pixels
            if fo['logo'] is not None:
                avail_width_frac = logo_x - fn_pos.x0 - 0.02
            else:
                avail_width_frac = fn_pos.width
            avail_px = avail_width_frac * fig.get_size_inches()[0] * fig.dpi

            # Renderer-based word wrapping (respects manual \n)
            renderer = fig.canvas.get_renderer()
            fn_segments = fo['text'].split('\n')
            wrapped_lines = []
            for segment in fn_segments:
                words = segment.split()
                if not words:
                    wrapped_lines.append('')
                    continue
                current_line = words[0]
                for word in words[1:]:
                    candidate = current_line + ' ' + word
                    t = fig.text(0, 0, candidate, fontsize=fo['fontsize'], fontstyle='italic')
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
                     fontsize=fo['fontsize'], color='#666666',
                     va='center', ha='left', fontstyle='italic')

        # Apply Title Formatting
        title_kwargs = {'fontweight': 'bold'}
        if use_ea_fonts:
            title_kwargs['fontproperties'] = fontprop_bold
        ax.text(title_x, to['title_y'], to['title'], fontsize=to['title_size'],
                transform=ax.transAxes, ha='left', **title_kwargs)
        if to['subtitle']:
            ax.text(title_x, to['subtitle_y'], to['subtitle'], fontsize=to['subtitle_size'],
                    transform=ax.transAxes, ha='left')

        # Apply x-axis formatting
        if xo['label']:
            ax.set_xlabel(xo['label'])
        if xo['lim']:
            if isinstance(xo['lim'][0], str) and isinstance(xo['lim'][1], str):
                ax.set_xlim(pd.to_datetime(xo['lim'][0]), pd.to_datetime(xo['lim'][1]))
            else:
                ax.set_xlim(xo['lim'])

        # Apply y-axis formatting (primary and secondary)
        axes_config = [(ax, yo, left_colors)]
        if has_right_axis:
            ax2.spines[['right']].set_visible(True)
            ax2.spines['right'].set_linewidth(1.0)
            ax2.spines['right'].set_color('#C8C8C8')
            axes_config.append((ax2, yr, right_colors))

        for target_ax, yopt, ax_colors in axes_config:
            if yopt['lim']:
                target_ax.set_ylim(yopt['lim'])
            if yopt['label']:
                ylabel_kwargs = {}
                if has_right_axis:
                    ylabel_kwargs['color'] = ax_colors[0] if ax_colors else style_colors[0]
                ylabel_font_kwargs = {'fontweight': 'bold'}
                if use_ea_fonts:
                    ylabel_font_kwargs['fontproperties'] = fontprop_bold
                target_ax.set_ylabel(yopt['label'], **ylabel_font_kwargs, **ylabel_kwargs)
            if yopt['ticksize']:
                target_ax.set_yticks(np.arange(
                    yopt['lim'][0],
                    yopt['lim'][1] + yopt['ticksize'] / 10,
                    yopt['ticksize']
                ))
            if yopt['tickformat'] == 'pctg':
                target_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=yopt['decimals']))
            elif yopt['tickformat'] == 'dec':
                if yopt['decimals'] is None:
                    target_ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
                else:
                    target_ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f'{{x:,.{yopt["decimals"]}f}}'))

        # Draw horizontal reference lines
        if hline is not None:
            hlines = hline if isinstance(hline, list) else [hline]
            for hl in hlines:
                if isinstance(hl, (int, float)):
                    ax.axhline(y=hl, color='#3C3C3C', linewidth=1.0, linestyle='-')
                elif isinstance(hl, dict):
                    hl_color = hl.get('color', '#3C3C3C')
                    ax.axhline(
                        y=hl['y'],
                        color=hl_color,
                        linewidth=hl.get('linewidth', 1.0),
                        linestyle=hl.get('linestyle', '-'),
                        label=hl.get('legend_label', None),
                    )
                    if hl.get('callout'):
                        callout_align = hl.get('callout_align', 'above')
                        va = 'bottom' if callout_align == 'above' else 'top'
                        y_offset = 1 if callout_align == 'above' else -1
                        callout_x = hl.get('callout_pos_x', 1.0)
                        ha = 'right' if callout_x >= 0.5 else 'left'
                        ax.annotate(
                            hl['callout'],
                            xy=(callout_x, hl['y']),
                            xytext=(0, y_offset),
                            textcoords='offset points',
                            xycoords=ax.get_yaxis_transform(),
                            va=va, ha=ha,
                            fontsize=hl.get('callout_fontsize', 9),
                            color=hl_color,
                        )

        # Format x-axis based on data frequency
        if xo['freq'] == 'Y':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if xo['interval']:
                ax.xaxis.set_major_locator(mdates.YearLocator(xo['interval']))
        elif xo['freq'] == 'M':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
            if xo['interval']:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=xo['interval']))

        # Apply Legend
        if lo['show']:
            if has_right_axis:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1+h2, l1+l2, bbox_to_anchor=(0.5, 1.0), loc=lo['loc'], ncol=lo['ncol'])
            else:
                ax.legend(bbox_to_anchor=(0.5, 1.0), loc=lo['loc'], ncol=lo['ncol'])

        # Save Figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', edgecolor=fig.get_edgecolor())

        return fig