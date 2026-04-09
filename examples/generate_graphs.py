"""
Generate a visual gallery of tsgraph() output for manual inspection.
Run from the repo root: python examples/generate_graphs.py
Output saved to examples/output/
"""
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import macrotools as mt

data = pd.read_pickle('examples/lndata.pkl')
output_dir = Path('examples/output')
output_dir.mkdir(exist_ok=True)

# 1. Single series
mt.tsgraph(
    series=data['LNS12300060'] / 100,
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'label': 'Ratio', 'lim': (0.76, 0.82), 'ticksize': 0.01,
           'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Prime-Age Employment Rate'},
    save_path=str(output_dir / '01_single_series.png'),
)
plt.close()
print(' 1/12  01_single_series.png')

# 2. Multiple series, one axis
mt.tsgraph(
    series=[
        {'y': data['LNS12300061'] / 100, 'label': 'Men'},
        {'y': data['LNS12300062'] / 100, 'label': 'Women'},
    ],
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'label': 'Ratio', 'lim': (0.70, 0.90), 'ticksize': 0.05,
           'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Prime-Age EPOP by Gender',
           'subtitle': 'Ratio, Employed to Population'},
    legend={'show': True},
    save_path=str(output_dir / '02_multi_series.png'),
)
plt.close()
print(' 2/12  02_multi_series.png')

# 3. Dual axes (EA style)
mt.tsgraph(
    series=[
        {'y': data['LNS12300061'] / 100, 'label': 'Men', 'axis': 'left'},
        {'y': data['LNS12300062'] / 100, 'label': 'Women', 'axis': 'right', 'linestyle': '--'},
    ],
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'label': 'Men', 'lim': (0.83, 0.88), 'ticksize': 0.01,
           'tickformat': 'pctg', 'decimals': 0},
    yaxis_rhs={'label': 'Women', 'lim': (0.73, 0.78), 'ticksize': 0.01,
               'tickformat': 'pctg', 'decimals': 0},
    legend={'show': True},
    style='ea',
    save_path=str(output_dir / '03_dual_axis_ea.png'),
)
plt.close()
print(' 3/12  03_dual_axis_ea.png')

# 4. DataFrame input
mt.tsgraph(
    series=data[['LNS12300061', 'LNS12300062']] / 100,
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'lim': (0.70, 0.90), 'ticksize': 0.05, 'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'DataFrame Input'},
    legend={'show': True},
    save_path=str(output_dir / '04_dataframe.png'),
)
plt.close()
print(' 4/12  04_dataframe.png')

# 5. hline with callout
mt.tsgraph(
    series=data['LNS12300060'] / 100,
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'label': 'Ratio', 'lim': (0.76, 0.82), 'ticksize': 0.01,
           'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'hline with Callout'},
    hline={'y': 0.80, 'callout': '80% Target', 'callout_align': 'below',
           'color': mt.default_alert_colors['alert_red']},
    save_path=str(output_dir / '05_hline_callout.png'),
)
plt.close()
print(' 5/12  05_hline_callout.png')

# 6. Per-series styling
mt.tsgraph(
    series=[
        {'y': data['LNS12300061'] / 100, 'label': 'Men',
         'color': mt.eacolors['steel_blue'], 'linewidth': 2.5},
        {'y': data['LNS12300062'] / 100, 'label': 'Women',
         'color': mt.eacolors['green'], 'linestyle': '--'},
    ],
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'lim': (0.70, 0.90), 'ticksize': 0.05, 'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Per-series Styling'},
    legend={'show': True},
    style='ea',
    save_path=str(output_dir / '06_per_series_styling.png'),
)
plt.close()
print(' 6/12  06_per_series_styling.png')

# 7. Mixed input types
mt.tsgraph(
    series=[
        data['LNS12300061'] / 100,
        {'y': data['LNS12300062'] / 100, 'label': 'Women', 'linestyle': ':'},
    ],
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'lim': (0.70, 0.90), 'ticksize': 0.05, 'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Mixed Input Types'},
    legend={'show': True},
    save_path=str(output_dir / '07_mixed_input.png'),
)
plt.close()
print(' 7/12  07_mixed_input.png')

# 8. Multiple hlines with callouts
mt.tsgraph(
    series=data['LNS12300060'] / 100,
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'label': 'Ratio', 'lim': (0.76, 0.82), 'ticksize': 0.01,
           'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Multiple hlines'},
    hline=[
        {'y': 0.80, 'callout': '80%', 'callout_pos_x': 0.0,
         'color': mt.default_alert_colors['alert_red']},
        {'y': 0.78, 'callout': 'Pre-pandemic avg', 'callout_align': 'below',
         'color': mt.default_alert_colors['alert_blue']},
    ],
    save_path=str(output_dir / '08_multiple_hlines.png'),
)
plt.close()
print(' 8/12  08_multiple_hlines.png')

# 9. Callout with auto-formatted value
mt.tsgraph(
    series={'y': data['LNS12300060'] / 100, 'label': 'EPOP',
            'callout': {'x': 'last', 'align': 'right'}},
    xaxis={'lim': ('2022-01', '2027-1'), 'interval': 6},
    yaxis={'lim': (0.79, 0.82), 'ticksize': 0.005, 'tickformat': 'pctg', 'decimals': 1},
    title={'title': 'Callout: Auto Value'},
    save_path=str(output_dir / '09_callout_auto_value.png'),
)
plt.close()
print(' 9/12  09_callout_auto_value.png')

# 10. Multiple callouts on one series
mt.tsgraph(
    series={'y': data['LNS12300060'] / 100, 'label': 'EPOP',
            'callout': [
                {'x': '2020-04', 'text': 'COVID trough', 'align': 'below'},
                {'x': 'last', 'align': 'right'},
            ]},
    xaxis={'lim': ('2019-01', '2027-1'), 'interval': 6},
    yaxis={'lim': (0.69, 0.82), 'ticksize': 0.02, 'tickformat': 'pctg', 'decimals': 0},
    title={'title': 'Callout: Multiple'},
    save_path=str(output_dir / '10_callout_multiple.png'),
)
plt.close()
print('10/12  10_callout_multiple.png')

# 11. Callout with custom text
mt.tsgraph(
    series={'y': data['LNS12300060'] / 100,
            'callout': {'x': 'last', 'text': 'Latest', 'align': 'above'}},
    xaxis={'lim': ('2022-01', '2027-1')},
    yaxis={'tickformat': 'pctg'},
    title={'title': 'Callout: Custom Text'},
    save_path=str(output_dir / '11_callout_custom_text.png'),
)
plt.close()
print('11/12  11_callout_custom_text.png')

# 12. Callout with decimal format
mt.tsgraph(
    series={'y': data['LNS12300060'],
            'callout': {'x': 'last', 'align': 'right', 'decimals': 0}},
    xaxis={'lim': ('2022-01', '2027-1')},
    yaxis={'tickformat': 'dec'},
    title={'title': 'Callout: Decimal Format'},
    save_path=str(output_dir / '12_callout_dec_format.png'),
)
plt.close()
print('12_callout_dec_format.png')

print(f'\nAll graphs saved to {output_dir.resolve()}')

# 13. Callout, bottom align
mt.tsgraph(
    series={'y': data['LNS12300060']/ 100,
            'callout': {'x': 'last', 'align': 'below'}},
    xaxis={'lim': ('2022-01', '2027-1')},
    yaxis={'tickformat': 'pctg'},
    title={'title': 'Callout: Decimal Format'},
    save_path=str(output_dir / '13_callout_bottom.png'),
)
plt.close()
print('13_callout_bottom.png')

print(f'\nAll graphs saved to {output_dir.resolve()}')
