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
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from collections import Counter
from functools import reduce


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


# Functions used for plotting on the qsphere.
#
# See:
# lex_index:
#      https://msdn.microsoft.com/en-us/library/aa289166%28v=vs.71%29.aspx
# n_choose_k: http://stackoverflow.com/questions/
#              2096573/counting-combinations-and-permutations-efficiently


class Arrow3D(FancyArrowPatch):
    """Standard 3D arrow."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Create arrow."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def compliment(value):
    """Swap 1 and 0 in a vector."""
    return ''.join(COMPLEMENT[x] for x in value)


COMPLEMENT = {'1': '0', '0': '1'}


def n_choose_k(n, k):
    """Return the number of combinations."""
    if n == 0:
        return 0.0
    else:
        return reduce(lambda x, y: x * y[0] / y[1],
                      zip(range(n - k + 1, n + 1),
                          range(1, k + 1)), 1)


def lex_index(n, k, lst):
    """Return the index of a combination."""
    assert len(lst) == k, "list should have length k"
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    m = dualm
    return int(m)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    assert s.count("0") == n - k, "s must be a string of 0 and 1"
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def plot_qsphere(data, number_to_keep, number_of_qubits):
    """Plot the qsphere of data."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axes.grid(False)
    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='k', alpha=0.05,
                    linewidth=0)
    # wireframe
    # Get rid of the panes
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    d = number_of_qubits
    total_values = sum(data.values())
    for key in data:
        weight = key.count("1")
        zvalue = -2 * weight / d + 1
        number_of_divisions = n_choose_k(d, weight)
        weight_order = bit_string_index(key)
        if weight_order >= number_of_divisions / 2:
            com_key = compliment(key)
            weight_order_temp = bit_string_index(com_key)
            weight_order = np.floor(
                number_of_divisions / 2) + weight_order_temp + 1
        print(key + "  " + str(weight_order))
        angle = (weight_order) * 2 * np.pi / number_of_divisions
        xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
        yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)
        linewidth = 5 * data.get(key) / total_values
        print([xvalue, yvalue, zvalue])
        a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue], mutation_scale=20,
                    lw=linewidth, arrowstyle="->", color="k")
        ax.add_artist(a)
    for weight in range(d + 1):
        theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        z = -2 * weight / d + 1
        if weight == 0:
            z = z - 0.001
        if weight == d:
            z = z + 0.001
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, z, 'k')
    plt.show()


# Functions used for plotting tomography.


def plot_bloch_vector(bloch, title=""):
    """Plot a Bloch vector.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.
    bloch is a 3-tuple (x, y, z)
    title is a string, the plot title
    """
    # Set arrow lengths
    arlen = 1.3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")

    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="b", alpha=0.1)

    # Plot arrows (axes, Bloch vector, its projections)
    xa = Arrow3D([0, arlen], [0, 0], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    ya = Arrow3D([0, 0], [0, arlen], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    za = Arrow3D([0, 0], [0, 0], [0, arlen], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    a = Arrow3D([0, bloch[0]], [0, bloch[1]], [0, bloch[2]], mutation_scale=20,
                lw=2, arrowstyle="simple", color="k")
    bax = Arrow3D([0, bloch[0]], [0, 0], [0, 0], mutation_scale=20, lw=2,
                  arrowstyle="-", color="r")
    bay = Arrow3D([0, 0], [0, bloch[1]], [0, 0], mutation_scale=20, lw=2,
                  arrowstyle="-", color="g")
    baz = Arrow3D([0, 0], [0, 0], [0, bloch[2]], mutation_scale=20, lw=2,
                  arrowstyle="-", color="b")
    arrowlist = [xa, ya, za, a, bax, bay, baz]
    for arr in arrowlist:
        ax.add_artist(arr)

    # Rotate the view
    ax.view_init(30, 30)

    # Annotate the axes, shifts are ad-hoc for this (30, 30) view
    xp, yp, _ = proj3d.proj_transform(arlen, 0, 0, ax.get_proj())
    plt.annotate("x", xy=(xp, yp), xytext=(-3, -8),
                 textcoords='offset points', ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, arlen, 0, ax.get_proj())
    plt.annotate("y", xy=(xp, yp), xytext=(6, -5),
                 textcoords='offset points', ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, 0, arlen, ax.get_proj())
    plt.annotate("z", xy=(xp, yp), xytext=(2, 0),
                 textcoords='offset points', ha='right', va='bottom')

    plt.title(title)
    plt.show()


# Functions used by randomized benchmarking.


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
