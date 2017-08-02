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
"""
import matplotlib.pyplot as plt
import numpy as np


# function used to fit the exponetial decay
def exp_fit_fun(x, a, tau, c):
    return a * np.exp(-x/tau) + c


# function used to fit the decay cosine
def osc_fit_fun(x, a, tau, f, phi, c):
    return a * np.exp(-x/tau)*np.cos(2*np.pi*f*x+phi) + c


# Functions used by randomized benchmarking.
def plot_coherence(xdata, ydata, std_error, fit, fit_function, xunit, exp_str,
                   qubit_label):
    """Plot coherence data.

    Args:
        xdata
        ydata
        std_error
        fit
        fit_function
        xunit
        exp_str
        qubit_label
    """
    plt.errorbar(xdata, ydata, std_error, marker='.',
                 markersize=9, c='b', linestyle='')
    plt.plot(xdata, fit_function(xdata, *fit), c='r', linestyle='--',
             label=(exp_str + '= %s %s' % (str(round(fit[1])), xunit)))
    plt.xticks(fontsize=14, rotation=70)
    plt.yticks(fontsize=14)
    plt.xlabel('time [%s]' % (xunit), fontsize=16)
    plt.ylabel('P(1)', fontsize=16)
    plt.title(exp_str + 'measurments of Q%s' % (str(qubit_label)), fontsize=18)
    plt.legend(fontsize=12)
    plt.show()


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
