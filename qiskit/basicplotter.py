# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Basic plotting methods using matplotlib.

These include methods to plot Bloch vectors, histograms, and quantum spheres.

Author: Andrew Cross, Jay Gambetta
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_histogram(data, number_to_keep=None):
    """Plot a histogram of data.

    data is a dictionary of  {'000': 5, '010': 113, ...}
    number_to_keep is the number of terms to plot and rest is made into a
    single bar called other values
    """
    if number_to_keep is not None:
        data_temp = dict(Counter(data).most_common(number_to_keep))
        data_temp["rest"] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = sorted(data)
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    numelem = len(values)
    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects = ax.bar(ind, pvalues, width, color='seagreen')
    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in pvalues])])])
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')
    plt.show()


# Functions used by randomized benchmarking. This we become basic curve fitting
# exp, cosine, linear etc.

def plot_rb_data(xdata, ydatas, yavg, fit, survival_prob):
    """Plot randomized benchmarking data.

    xdata = list of subsequence lengths
    ydatas = list of lists of survival probabilities for each sequence
    yavg = mean of the survival probabilities at each sequence length
    fit = list of fitting parameters [a, b, alpha]
    survival_prob = function that computes survival probability
    """
    # Plot the result for each sequence
    for ydata in ydatas:
        plt.plot(xdata, ydata, 'rx')
    # Plot the mean
    plt.plot(xdata, yavg, 'bo')
    # Plot the fit
    plt.plot(xdata, survival_prob(xdata, *fit), 'b-')
    plt.show()
