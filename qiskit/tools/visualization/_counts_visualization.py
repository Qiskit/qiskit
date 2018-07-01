# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""
Visualization functions for measurement counts.
"""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data, number_to_keep=False):
    """Plot a histogram of data.

    data is a dictionary of  {'000': 5, '010': 113, ...}
    number_to_keep is the number of terms to plot and rest is made into a
    single bar called other values
    """
    if number_to_keep is not False:
        data_temp = dict(Counter(data).most_common(number_to_keep))
        data_temp["rest"] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = sorted(data)
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    numelem = len(values)
    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.35  # the width of the bars
    _, ax = plt.subplots()
    rects = ax.bar(ind, pvalues, width, color='seagreen')
    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=12, rotation=70)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in pvalues])])])
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')
    plt.show()
