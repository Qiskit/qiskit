# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Basic plotting methods using matplotlib.

These include methods to plot Bloch vectors, histograms, and quantum spheres.
"""

import numpy as np

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def exp_fit_fun(x, a, tau, c):
    """Function used to fit the exponential decay."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) + c


def osc_fit_fun(x, a, tau, f, phi, c):
    """Function used to fit the decay cosine."""
    # pylint: disable=invalid-name
    return a * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + c


def rb_fit_fun(x, a, alpha, b):
    """Function used to fit rb."""
    # pylint: disable=invalid-name
    return a * alpha ** x + b


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
    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('The function plot_coherence needs matplotlib. '
                          'Run "pip install matplotlib" before.')
    plt.errorbar(xdata, ydata, std_error, marker='.',
                 markersize=9, c='b', linestyle='')
    plt.plot(xdata, fit_function(xdata, *fit), c='r', linestyle='--',
             label=(exp_str + '= %s %s' % (str(round(fit[1])), xunit)))
    plt.xticks(fontsize=14, rotation=70)
    plt.yticks(fontsize=14)
    plt.xlabel('time [%s]' % (xunit), fontsize=16)
    plt.ylabel('P(1)', fontsize=16)
    plt.title(exp_str + ' measurement of Q$_{%s}$' % (str(qubit_label)), fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def shape_rb_data(raw_rb):
    """Take the raw rb data and convert it into averages and std dev

    Args:
        raw_rb (numpy.array): m x n x l list where m is the number of seeds, n
            is the number of Clifford sequences and l is the number of qubits

    Return:
        numpy_array: 2 x n x l list where index 0 is the mean over seeds, 1 is
            the std dev overseeds
    """
    rb_data = []
    rb_data.append(np.mean(raw_rb, 0))
    rb_data.append(np.std(raw_rb, 0))

    return rb_data


def rb_epc(fit, rb_pattern):
    """Take the rb fit data and convert it into EPC (error per Clifford)

    Args:
        fit (dict): dictionary of the fit quantities (A, alpha, B) with the
            keys 'qn' where n is  the qubit and subkeys 'fit', e.g.
            {'q0':{'fit': [1, 0, 0.9], 'fiterr': [0, 0, 0]}}}
        rb_pattern (list): (see randomized benchmarking functions). Pattern
            which specifies which qubits performing RB with which qubits. E.g.
            [[1],[0,2]] is Q1  doing 1Q RB simultaneously with Q0/Q2 doing
            2Q RB

    Return:
        dict: updates the passed in fit dictionary with the epc
    """
    for patterns in rb_pattern:
        for qubit in patterns:
            fitalpha = fit['q%d' % qubit]['fit'][1]
            fitalphaerr = fit['q%d' % qubit]['fiterr'][1]
            nrb = 2 ** len(patterns)

            fit['q%d' % qubit]['fit_calcs'] = {}
            fit['q%d' % qubit]['fit_calcs']['epc'] = [(nrb - 1) / nrb * (1 - fitalpha),
                                                      fitalphaerr / fitalpha]
            fit['q%d' % qubit]['fit_calcs']['epc'][1] *= fit['q%d' % qubit]['fit_calcs']['epc'][0]

    return fit


def plot_rb_data(xdata, ydatas, yavg, yerr, fit, survival_prob, ax=None,
                 show_plt=True):
    """Plot randomized benchmarking data.

    Args:
        xdata (list): list of subsequence lengths
        ydatas (list): list of lists of survival probabilities for each
            sequence
        yavg (list): mean of the survival probabilities at each sequence
            length
        yerr (list): error of the survival
        fit (list): fit parameters
        survival_prob (callable): function that computes survival probability
        ax (Axes or None): plot axis (if passed in)
        show_plt (bool): display the plot.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    # pylint: disable=invalid-name

    if not HAS_MATPLOTLIB:
        raise ImportError('The function plot_rb_data needs matplotlib. '
                          'Run "pip install matplotlib" before.')
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot the result for each sequence
    for ydata in ydatas:
        ax.plot(xdata, ydata, color='gray', linestyle='none', marker='x')
    # Plot the mean with error bars
    ax.errorbar(xdata, yavg, yerr=yerr, color='r', linestyle='--', linewidth=3)

    # Plot the fit
    ax.plot(xdata, survival_prob(xdata, *fit), color='blue', linestyle='-', linewidth=2)
    ax.tick_params(labelsize=14)
    # ax.tick_params(axis='x',labelrotation=70)

    ax.set_xlabel('Clifford Length', fontsize=16)
    ax.set_ylabel('Z', fontsize=16)
    ax.grid(True)

    if show_plt:
        plt.show()
