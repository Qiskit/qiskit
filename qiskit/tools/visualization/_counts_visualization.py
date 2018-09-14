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
import warnings

import numpy as np
import matplotlib.pyplot as plt

from ._error import VisualizationError


def plot_histogram(data, number_to_keep=False, legend=None, options=None):
    """Plot a histogram of data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        number_to_keep (int): DEPRECATED the number of terms to plot and rest
            is made into a single bar called other values
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the lenght of data (if data is a
            list or 1 if it's a dict)
        options (dict): Representation settings containing
            - number_to_keep (integer): groups max values
            - show_legend (bool): show legend of graph content
    Raises:
        VisualizationError: When legend is provided and the length doesn't
            match the input data.
    """
    if options is None:
        options = {}

    if number_to_keep is not False:
        warnings.warn("number_to_keep has been deprecated, use the options "
                      "dictionary and set a number_to_keep key instead",
                      DeprecationWarning)

    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError("Length of legendL (%s) doesn't match "
                                 "number of input executions: %s" %
                                 (len(legend), len(data)))

    _, ax = plt.subplots()
    for item, execution in enumerate(data):
        if number_to_keep is not False or (
                'number_to_keep' in options and options['number_to_keep']):
            data_temp = dict(Counter(execution).most_common(number_to_keep))
            data_temp["rest"] = sum(execution.values()) - sum(data_temp.values())
            execution = data_temp

        labels = sorted(execution)
        values = np.array([execution[key] for key in labels], dtype=float)
        pvalues = values / sum(values)
        numelem = len(values)
        ind = np.arange(numelem)  # the x locations for the groups
        width = 0.35  # the width of the bars
        label = None
        if legend:
            label = legend[item]
        adj = width * item
        rects = ax.bar(ind+adj, pvalues, width, label=label)
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
    if legend and (
            'show_legend' not in options or options['show_legend'] is True):
        plt.legend()
    plt.show()
