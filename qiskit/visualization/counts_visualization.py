# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,import-error

"""
Visualization functions for measurement counts.
"""

from collections import Counter, OrderedDict
import functools
import numpy as np
from .matplotlib import HAS_MATPLOTLIB
from .exceptions import VisualizationError

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two bit strings

    Args:
        str1 (str): First string.
        str2 (str): Second string.
    Returns:
        int: Distance between strings.
    Raises:
        VisualizationError: Strings not same length
    """
    if len(str1) != len(str2):
        raise VisualizationError('Strings not same length.')
    return sum(s1 != s2 for s1, s2 in zip(str1, str2))


VALID_SORTS = ['asc', 'desc', 'hamming']
DIST_MEAS = {'hamming': hamming_distance}


def plot_histogram(data, figsize=(7, 5), color=None, number_to_keep=None,
                   sort='asc', target_string=None,
                   legend=None, bar_labels=True, title=None):
    """Plot a histogram of data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        figsize (tuple): Figure size in inches.
        color (list or str): String or list of strings for histogram bar colors.
        number_to_keep (int): The number of terms to plot and rest
            is made into a single bar called 'rest'.
        sort (string): Could be 'asc', 'desc', or 'hamming'.
        target_string (str): Target string if 'sort' is a distance measure.
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the length of data (if data is a
            list or 1 if it's a dict)
        bar_labels (bool): Label each bar in histogram with probability value.
        title (str): A string to use for the plot title

    Returns:
        matplotlib.Figure: A figure for the rendered histogram.

    Raises:
        ImportError: Matplotlib not available.
        VisualizationError: When legend is provided and the length doesn't
            match the input data.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if sort not in VALID_SORTS:
        raise VisualizationError("Value of sort option, %s, isn't a "
                                 "valid choice. Must be 'asc', "
                                 "'desc', or 'hamming'")
    if sort in DIST_MEAS.keys() and target_string is None:
        err_msg = 'Must define target_string when using distance measure.'
        raise VisualizationError(err_msg)

    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError("Length of legendL (%s) doesn't match "
                                 "number of input executions: %s" %
                                 (len(legend), len(data)))

    fig, ax = plt.subplots(figsize=figsize)
    labels = list(sorted(
        functools.reduce(lambda x, y: x.union(y.keys()), data, set())))
    if number_to_keep is not None:
        labels.append('rest')

    if sort in DIST_MEAS.keys():
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string))

        labels = [list(x) for x in zip(*sorted(zip(dist, labels),
                                               key=lambda pair: pair[0]))][1]

    labels_dict = OrderedDict()

    # Set bar colors
    if color is None:
        color = ['#648fff', '#dc267f', '#785ef0', '#ffb000', '#fe6100']
    elif isinstance(color, str):
        color = [color]

    all_pvalues = []
    length = len(data)
    for item, execution in enumerate(data):
        if number_to_keep is not None:
            data_temp = dict(Counter(execution).most_common(number_to_keep))
            data_temp["rest"] = sum(execution.values()) - sum(data_temp.values())
            execution = data_temp
        values = []
        for key in labels:
            if key not in execution:
                if number_to_keep is None:
                    labels_dict[key] = 1
                    values.append(0)
                else:
                    values.append(-1)
            else:
                labels_dict[key] = 1
                values.append(execution[key])
        values = np.array(values, dtype=float)
        where_idx = np.where(values >= 0)[0]
        pvalues = values[where_idx] / sum(values[where_idx])
        for value in pvalues:
            all_pvalues.append(value)
        numelem = len(values[where_idx])
        ind = np.arange(numelem)  # the x locations for the groups
        width = 1/(len(data)+1)  # the width of the bars
        rects = []
        for idx, val in enumerate(pvalues):
            label = None
            if not idx and legend:
                label = legend[item]
            if val >= 0:
                rects.append(ax.bar(idx+item*width, val, width, label=label,
                                    color=color[item % len(color)],
                                    zorder=2))
        bar_center = (width / 2) * (length - 1)
        ax.set_xticks(ind + bar_center)
        ax.set_xticklabels(labels_dict.keys(), fontsize=14, rotation=70)
        # attach some text labels
        if bar_labels:
            for rect in rects:
                for rec in rect:
                    height = rec.get_height()
                    if height >= 1e-3:
                        ax.text(rec.get_x() + rec.get_width() / 2., 1.05 * height,
                                '%.3f' % float(height),
                                ha='center', va='bottom', zorder=3)
                    else:
                        ax.text(rec.get_x() + rec.get_width() / 2., 1.05 * height,
                                '0',
                                ha='center', va='bottom', zorder=3)

    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Probabilities', fontsize=14)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in all_pvalues])])])
    if sort == 'desc':
        ax.invert_xaxis()

    ax.yaxis.set_major_locator(MaxNLocator(5))
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_facecolor('#eeeeee')
    plt.grid(which='major', axis='y', zorder=0, linestyle='--')
    if title:
        plt.title(title)

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1,
                  borderaxespad=0, frameon=True, fontsize=12)
    if fig:
        plt.close(fig)
    return fig
