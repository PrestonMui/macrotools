import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import macrotools as mt


# ============================================================
# Smoke tests — function runs and returns a Figure
# ============================================================

def test_bare_series(sample_data):
    fig = mt.tsgraph(series=sample_data['LNS12300060'] / 100)
    assert isinstance(fig, plt.Figure)


def test_dict_series(sample_data):
    fig = mt.tsgraph(series={'y': sample_data['LNS12300060'] / 100, 'label': 'EPOP'})
    assert isinstance(fig, plt.Figure)


def test_list_of_dicts(sample_data):
    fig = mt.tsgraph(series=[
        {'y': sample_data['LNS12300061'] / 100, 'label': 'Men'},
        {'y': sample_data['LNS12300062'] / 100, 'label': 'Women'},
    ])
    assert isinstance(fig, plt.Figure)


def test_dataframe(sample_data):
    df = sample_data[['LNS12300061', 'LNS12300062']] / 100
    fig = mt.tsgraph(series=df)
    assert isinstance(fig, plt.Figure)


def test_mixed_list(sample_data):
    fig = mt.tsgraph(series=[
        sample_data['LNS12300061'] / 100,
        {'y': sample_data['LNS12300062'] / 100, 'label': 'Women'},
    ])
    assert isinstance(fig, plt.Figure)


def test_dual_axis(sample_data):
    fig = mt.tsgraph(series=[
        {'y': sample_data['LNS12300061'] / 100, 'label': 'Men', 'axis': 'left'},
        {'y': sample_data['LNS12300062'] / 100, 'label': 'Women', 'axis': 'right'},
    ])
    assert isinstance(fig, plt.Figure)


def test_ea_style(sample_data):
    fig = mt.tsgraph(
        series=sample_data['LNS12300060'] / 100,
        style='ea',
    )
    assert isinstance(fig, plt.Figure)


def test_hline_numeric(sample_data):
    fig = mt.tsgraph(
        series=sample_data['LNS12300060'] / 100,
        hline=0.80,
    )
    assert isinstance(fig, plt.Figure)


def test_hline_dict_with_callout(sample_data):
    fig = mt.tsgraph(
        series=sample_data['LNS12300060'] / 100,
        hline={'y': 0.80, 'callout': '80%', 'callout_align': 'below',
               'color': mt.default_alert_colors['alert_red']},
    )
    assert isinstance(fig, plt.Figure)


def test_all_options(sample_data):
    """Kitchen sink — every parameter populated."""
    fig = mt.tsgraph(
        series=[
            {'y': sample_data['LNS12300061'] / 100, 'label': 'Men', 'axis': 'left',
             'color': mt.eacolors['steel_blue'], 'linewidth': 2.5},
            {'y': sample_data['LNS12300062'] / 100, 'label': 'Women', 'axis': 'right',
             'linestyle': '--'},
        ],
        xaxis={'lim': ('2019-01', '2025-06'), 'interval': 6, 'label': 'Date', 'freq': 'M'},
        yaxis={'label': 'Men', 'lim': (0.83, 0.88), 'ticksize': 0.01,
               'tickformat': 'pctg', 'decimals': 0},
        yaxis_rhs={'label': 'Women', 'lim': (0.73, 0.78), 'ticksize': 0.01,
                   'tickformat': 'pctg', 'decimals': 0},
        title={'title': 'Employment Rate', 'subtitle': 'By Gender'},
        legend={'show': True, 'ncol': 2},
        hline=[
            {'y': 0.85, 'callout': 'Target', 'callout_pos_x': 0.0},
            0.86,
        ],
        style='ea',
        figsize=(10, 6),
        bgcolor='white',
    )
    assert isinstance(fig, plt.Figure)


# ============================================================
# Property tests — verify specific graph attributes
# ============================================================

def test_single_axis_has_one_axis(sample_data):
    fig = mt.tsgraph(series=sample_data['LNS12300060'] / 100)
    assert len(fig.axes) == 1


def test_dual_axis_has_two_axes(sample_data):
    fig = mt.tsgraph(series=[
        {'y': sample_data['LNS12300061'] / 100, 'axis': 'left'},
        {'y': sample_data['LNS12300062'] / 100, 'axis': 'right'},
    ])
    assert len(fig.axes) == 2


def test_ylim_applied(sample_data):
    fig = mt.tsgraph(
        series=sample_data['LNS12300060'] / 100,
        yaxis={'lim': (0.76, 0.82)},
    )
    ax = fig.axes[0]
    assert ax.get_ylim() == pytest.approx((0.76, 0.82))


def test_legend_shows_labels(sample_data):
    fig = mt.tsgraph(
        series=[
            {'y': sample_data['LNS12300061'] / 100, 'label': 'Men'},
            {'y': sample_data['LNS12300062'] / 100, 'label': 'Women'},
        ],
        legend={'show': True},
    )
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert 'Men' in labels
    assert 'Women' in labels


def test_title_text(sample_data):
    fig = mt.tsgraph(
        series=sample_data['LNS12300060'] / 100,
        title={'title': 'My Test Title'},
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert 'My Test Title' in texts


def test_color_assignment(sample_data):
    """Two series should get distinct auto-assigned colors."""
    fig = mt.tsgraph(series=[
        {'y': sample_data['LNS12300061'] / 100, 'label': 'A'},
        {'y': sample_data['LNS12300062'] / 100, 'label': 'B'},
    ])
    lines = fig.axes[0].get_lines()
    assert len(lines) >= 2
    assert lines[0].get_color() != lines[1].get_color()


def test_explicit_color(sample_data):
    fig = mt.tsgraph(series={'y': sample_data['LNS12300060'] / 100, 'color': '#FF0000'})
    lines = fig.axes[0].get_lines()
    assert lines[0].get_color() == '#FF0000'


def test_no_label_none_in_legend(sample_data):
    """Series without a label should not show 'None' in the legend."""
    fig = mt.tsgraph(
        series=[
            sample_data['LNS12300061'] / 100,  # has .name from pd.Series
            {'y': sample_data['LNS12300062'] / 100},  # no label
        ],
        legend={'show': True},
    )
    ax = fig.axes[0]
    legend = ax.get_legend()
    if legend is not None:
        labels = [t.get_text() for t in legend.get_texts()]
        assert 'None' not in labels


# ============================================================
# Error tests
# ============================================================

def test_error_on_missing_y_key():
    with pytest.raises(ValueError, match="'y' key"):
        mt.tsgraph(series={'label': 'oops', 'color': 'red'})


def test_error_on_no_xdata():
    with pytest.raises(ValueError, match="x-data"):
        mt.tsgraph(series=[1, 2, 3])
