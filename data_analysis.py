import os
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps


TAB10 = colormaps['tab10']
TAB20 = colormaps['tab20']

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')


class Data:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.name = os.path.split(csv_path)[-1][:-4]

        with open(csv_path + '.tag', 'r') as f:
            r = [i.replace('\n', '') for i in f.readlines()]

        self.arch, self.mp, self.cell, *_ = r
        if self.cell == 'True' or self.cell == 'cell':
            self.cell = True
        elif self.cell == 'False' or self.cell == 'no_cell':
            self.cell = False
        else:
            raise Exception(f'{csv_path}; cell={self.cell}')


def is_diagonal(arr):
    if isinstance(arr, str):
        arr = np.array(arr.split()).astype(int)

    try:
        if len(arr) == 3:
            return True
    except TypeError:
        if np.isnan(arr):
            return True
        else:
            raise

    arr = arr.reshape((3, 3))

    return np.allclose(arr - (np.diag(arr) * np.eye(3)), 0)


def get_specific_data(data: list[Data], arch: str, model_path: str, cell: str) -> Data:
    for d in data:
        if d.arch == arch and d.mp == model_path and d.cell == cell:
            return d


class ignore(catch_warnings):
    def __enter__(self):
        super().__enter__()
        simplefilter('ignore')


def plot_column_pie(df: pd.DataFrame, ax, column: str, show_total=False, ref_labels=None, show_zero=False):
    opt = df[column].dropna().value_counts().sort_index()
    if ref_labels is None or not show_zero:
        labels = list(opt.index)
        values = opt.values
    else:
        values = np.array([opt.get(label) if opt.get(label) is not None else 0 for label in ref_labels])
        labels = ref_labels

    total = values.sum()
    fmt = (lambda x: f'{x:.2f}%\n({total * x / 100:.0f})') if show_total else '%1.1f%%'

    if ref_labels is None or show_zero:
        ax.pie(values, labels=labels, autopct=fmt)
    else:
        cs = [TAB10(i) for i, l in enumerate(sorted(ref_labels)) if l in labels]
        ax.pie(values, labels=labels, autopct=fmt, colors=cs)


def plot_hist_pie(df: pd.DataFrame, ax, column: str, bins, show_total=False):
    values, _ = np.histogram(df[column].dropna(), bins)

    total = values.sum()
    fmt = (lambda x: f'{x:.2f}%\n({total * x / 100:.0f})') if show_total else '%1.1f%%'

    all_labels = [f'<{edge:.1f},{bins[i+1]:.1f})' for i, edge in enumerate(bins[:-1])]
    vals, cs, labels = [], [], []
    for i, value in enumerate(values):
        if value > 0:
            vals.append(value)
            cs.append(TAB20(i))
            labels.append(all_labels[i])

    ax.pie(vals, labels=labels, autopct=fmt, colors=cs)


def plot_basic_bar(series, ax=None, xlabel=''):
    if ax is None:
        fig, ax = plt.subplots()

    opt = series.dropna().value_counts()
    labels = list(opt.index)
    values = opt.values

    ax.bar(labels, values)

    ax.set_ylabel('Number of structures')
    if xlabel:
        ax.set_xlabel(xlabel)

    if ax is None:
        plt.show()


def get_displacements(n: int, bar_width: float = 0.9):
    width = bar_width / n
    half = 0.5 * bar_width
    return width, np.linspace(- (half - 0.5 * width), half - 0.5 * width, n)


def value_counts(series):
    return series.value_counts()


def value_counts_sorted(series):
    return series.value_counts().sort_index()


def create_range_counts(ranges, convert_to_float=True):
    def range_counts(series):
        series = series.astype(float) if convert_to_float else series
        return series.groupby(pd.cut(series, ranges), observed=True).count()

    return range_counts


def select_equal(df, name, label):
    return df[name] == label


def select_interval(df, name, label):
    return df[name].between(label.left, label.right)


def plot_nested_pie(df,
                    ax,
                    primary_column,
                    secondary_column,
                    primary_func=value_counts_sorted,
                    secondary_func=value_counts,
                    select_func=select_equal,
                    size=0.3):
    rdf = df[df[primary_column].notna()]

    opt = primary_func(rdf[primary_column])
    labels = list(opt.index)
    values = opt.values

    possible_hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    n_major_colours = len(labels)
    new_labels = list(secondary_func(rdf[secondary_column]).index)
    n_minor_colours = len(new_labels)

    tab10 = plt.color_sequences["tab10"]
    colour_pool = [tab10[i] for i in range(n_major_colours, n_major_colours + n_minor_colours)]

    wedges, *_ = ax.pie(values, radius=1, labels=labels, autopct='%1.1f%%', pctdistance=0.85,
                        # hatch=possible_hatches[:n_major_colours],
                        wedgeprops=dict(width=size, edgecolor='w'))

    counts = []
    colours, edge_colours = [], []
    hatches = []
    for i, label in enumerate(labels):
        sub_df = rdf[select_func(rdf, primary_column, label)]
        vals = list(secondary_func(sub_df[secondary_column]).values)

        counts.extend(vals)
        colours.extend(colour_pool[:len(vals)])
        hatches.extend([possible_hatches[i]] * len(vals))
        edge_colours.extend([tab10[i]] * len(vals))

    wedges, *_ = ax.pie(counts, radius=1 - size, colors=colours,  # hatch=hatches,
                        wedgeprops=dict(width=size, edgecolor='w'))

    j = 0
    for wedge, ec in zip(wedges, edge_colours):
        wedge.set(ec=ec)

    legend_elements = [Patch(facecolor=c, edgecolor=c, label=lab) for c, lab in
                       zip(colour_pool, new_labels)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5))


def plot_nested_histogram(df, ax, primary_column, secondary_column):
    rdf = df[df[primary_column].notna()]

    opt = rdf[primary_column].value_counts()
    labels = opt.index.array

    secondary_labels = df[secondary_column].dropna().value_counts().index
    n_slabel = len(secondary_labels)

    subdata = []
    for label in labels:
        sub_df = rdf[rdf[primary_column] == label]
        vals = sub_df[secondary_column].dropna().value_counts()

        if len(vals) == n_slabel:
            subdata.append(list(vals.values))
        else:
            new_vals = [vals[l] if l in vals.index else 0 for l in secondary_labels]
            subdata.append(new_vals)

    subdata = np.array(subdata).T
    width, displacements = get_displacements(n_slabel)

    for label, values, disp in zip(secondary_labels, subdata, displacements):
        ax.bar(labels + disp, values, width=width, label=label)

    ax.legend()

    ax.set_xlabel(primary_column)