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

# function used to fit the exponetial decay
def exp_fit_fun(x,a,tau,c):
    return a * np.exp(-x/tau) + c

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
