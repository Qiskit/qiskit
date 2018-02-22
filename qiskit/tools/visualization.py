# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,anomalous-backslash-in-string

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
Visualization functions for quantum states.
"""

import itertools
import operator
import re
import os
import subprocess
import tempfile
import logging
from collections import Counter, OrderedDict
from functools import reduce
from io import StringIO

import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None
    ImageChops = None

from qiskit import qasm, unroll, QISKitError
from qiskit.tools.qi.pauli import pauli_group, pauli_singles

logger = logging.getLogger(__name__)

###############################################################
# Plotting histogram
###############################################################


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


###############################################################
# Plotting states
###############################################################

class Arrow3D(FancyArrowPatch):
    """Standard 3D arrow."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Create arrow."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_bloch_vector(bloch, title=""):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        bloch (list[double]): array of three elements where [<x>, <y>,<z>]
        title (str): a string that represents the plot title
    """
    # Set arrow lengths
    arlen = 1.3

    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    ax.plot_surface(x, y, z, color=(.5, .5, .5), alpha=0.1)

    # Plot arrows (axes, Bloch vector, its projections)
    xa = Arrow3D([0, arlen], [0, 0], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color=(.5, .5, .5))
    ya = Arrow3D([0, 0], [0, arlen], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color=(.5, .5, .5))
    za = Arrow3D([0, 0], [0, 0], [0, arlen], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color=(.5, .5, .5))
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
    plt.annotate("x", xy=(xp, yp), xytext=(-3, -8), textcoords='offset points',
                 ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, arlen, 0, ax.get_proj())
    plt.annotate("y", xy=(xp, yp), xytext=(6, -5), textcoords='offset points',
                 ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, 0, arlen, ax.get_proj())
    plt.annotate("z", xy=(xp, yp), xytext=(2, 0), textcoords='offset points',
                 ha='right', va='bottom')

    plt.title(title)
    plt.show()


def plot_state_city(rho, title=""):
    """Plot the cityscape of quantum state.

    Plot two 3d bargraphs (two dimenstional) of the mixed state rho

    Args:
        rho (np.array[[complex]]): array of dimensions 2**n x 2**nn complex
                                   numbers
        title (str): a string that represents the plot title
    """
    num = int(np.log2(len(rho)))

    # get the real and imag parts of rho
    datareal = np.real(rho)
    dataimag = np.imag(rho)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]

    lx = len(datareal[0])            # Work out matrix dimensions
    ly = len(datareal[:, 0])
    xpos = np.arange(0, lx, 1)    # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dzr, color="g", alpha=0.5)
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dzi, color="g", alpha=0.5)

    ax1.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax1.set_yticks(np.arange(0.5, ly+0.5, 1))
    ax1.axes.set_zlim3d(-1.0, 1.0001)
    ax1.set_zticks(np.arange(-1, 1, 0.5))
    ax1.w_xaxis.set_ticklabels(row_names, fontsize=12, rotation=45)
    ax1.w_yaxis.set_ticklabels(column_names, fontsize=12, rotation=-22.5)
    # ax1.set_xlabel('basis state', fontsize=12)
    # ax1.set_ylabel('basis state', fontsize=12)
    ax1.set_zlabel("Real[rho]")

    ax2.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax2.set_yticks(np.arange(0.5, ly+0.5, 1))
    ax2.axes.set_zlim3d(-1.0, 1.0001)
    ax2.set_zticks(np.arange(-1, 1, 0.5))
    ax2.w_xaxis.set_ticklabels(row_names, fontsize=12, rotation=45)
    ax2.w_yaxis.set_ticklabels(column_names, fontsize=12, rotation=-22.5)
    # ax2.set_xlabel('basis state', fontsize=12)
    # ax2.set_ylabel('basis state', fontsize=12)
    ax2.set_zlabel("Imag[rho]")
    plt.title(title)
    plt.show()


def plot_state_paulivec(rho, title=""):
    """Plot the paulivec representation of a quantum state.

    Plot a bargraph of the mixed state rho over the pauli matricies

    Args:
        rho (np.array[[complex]]): array of dimensions 2**n x 2**nn complex
                                   numbers
        title (str): a string that represents the plot title
    """
    num = int(np.log2(len(rho)))
    labels = list(map(lambda x: x.to_label(), pauli_group(num)))
    values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                      pauli_group(num)))
    numelem = len(values)
    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.5  # the width of the bars
    _, ax = plt.subplots()
    ax.grid(zorder=0)
    ax.bar(ind, values, width, color='seagreen')

    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Expectation value', fontsize=12)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=12, rotation=70)
    ax.set_xlabel('Pauli', fontsize=12)
    ax.set_ylim([-1, 1])
    plt.title(title)
    plt.show()


def n_choose_k(n, k):
    """Return the number of combinations for n choose k.

    Args:
        n (int): the total number of options .
        k (int): The number of elements.

    Returns:
        int: returns the binomial coefficient
    """
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1],
                  zip(range(n - k + 1, n + 1),
                      range(1, k + 1)), 1)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    """
    assert len(lst) == k, "list should have length k"
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    assert s.count("0") == n - k, "s must be a string of 0 and 1"
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_color_wheel(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to a color wheel with blue at zero phase.
    """
    angles = np.angle(complex_number)
    angle_round = int(((angles + 2 * np.pi) % (2 * np.pi))/np.pi*6)
    # print(angleround)
    color_map = {
        0: (0, 0, 1),  # blue,
        1: (0.5, 0, 1),  # blue-violet
        2: (1, 0, 1),  # violet
        3: (1, 0, 0.5),  # red-violet,
        4: (1, 0, 0),  # red
        5: (1, 0.5, 0),  # red-oranage,
        6: (1, 1, 0),  # orange
        7: (0.5, 1, 0),  # orange-yellow
        8: (0, 1, 0),  # yellow,
        9: (0, 1, 0.5),  # yellow-green,
        10: (0, 1, 1),  # green,
        11: (0, 0.5, 1)  # green-blue,
    }
    return color_map[angle_round]


def plot_state_qsphere(rho):
    """Plot the qsphere representation of a quantum state."""
    num = int(np.log2(len(rho)))
    # get the eigenvectors and egivenvalues
    we, stateall = la.eigh(rho)
    for k in range(2**num):
        # start with the max
        probmix = we.max()
        prob_location = we.argmax()
        if probmix > 0.001:
            print("The " + str(k) + "th eigenvalue = " + str(probmix))
            # get the max eigenvalue
            state = stateall[:, prob_location]
            loc = np.absolute(state).argmax()
            # get the element location closes to lowest bin representation.
            for j in range(2**num):
                test = np.absolute(np.absolute(state[j])
                                   - np.absolute(state[loc]))
                if test < 0.001:
                    loc = j
                    break
            # remove the global phase
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j*angles)
            # print(state)
            # print(angles)
            state = angleset*state
            # print(state)
            state.flatten()
            # start the plotting
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
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='k',
                            alpha=0.05, linewidth=0)
            # wireframe
            # Get rid of the panes
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Get rid of the spines
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # Get rid of the ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            d = num
            for i in range(2**num):
                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                # if weight_order >= number_of_divisions / 2:
                #    com_key = compliment(element)
                #    weight_order_temp = bit_string_index(com_key)
                #    weight_order = np.floor(
                #        number_of_divisions / 2) + weight_order_temp + 1
                angle = weight_order * 2 * np.pi / number_of_divisions
                xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)
                ax.plot([xvalue], [yvalue], [zvalue],
                        markerfacecolor=(.5, .5, .5),
                        markeredgecolor=(.5, .5, .5),
                        marker='o', markersize=10, alpha=1)
                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                colorstate = phase_to_color_wheel(state[i])
                a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue],
                            mutation_scale=20, alpha=prob, arrowstyle="-",
                            color=colorstate, lw=10)
                ax.add_artist(a)
            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z**2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(.5, .5, .5))
            # add center point
            ax.plot([0], [0], [0], markerfacecolor=(.5, .5, .5),
                    markeredgecolor=(.5, .5, .5), marker='o', markersize=10,
                    alpha=1)
            plt.show()
            we[prob_location] = 0
        else:
            break


def plot_state(rho, method='city'):
    """Plot the quantum state."""
    num = int(np.log2(len(rho)))
    # Need updating to check its a matrix
    if method == 'city':
        plot_state_city(rho)
    elif method == "paulivec":
        plot_state_paulivec(rho)
    elif method == "qsphere":
        plot_state_qsphere(rho)
    elif method == "bloch":
        for i in range(num):
            bloch_state = list(
                map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                    pauli_singles(i, num)))
            plot_bloch_vector(bloch_state, "qubit " + str(i))
    elif method == "wigner":
        plot_wigner_function(rho)


###############################################################
# Plotting Wigner functions
###############################################################

def plot_wigner_function(state, res=100):
    """Plot the equal angle slice spin Wigner function of an arbitrary
    quantum state.

    Args:
        state (np.matrix[[complex]]):
            - Matrix of 2**n x 2**n complex numbers
            - State Vector of 2**n x 1 complex numbers
        res (int) : number of theta and phi values in meshgrid
            on sphere (creates a res x res grid of points)

    References:
        [1] T. Tilma, M. J. Everitt, J. H. Samson, W. J. Munro,
        and K. Nemoto, Phys. Rev. Lett. 117, 180401 (2016).
        [2] R. P. Rundle, P. W. Mills, T. Tilma, J. H. Samson, and
        M. J. Everitt, Phys. Rev. A 96, 022117 (2017).
    """
    state = np.array(state)
    if state.ndim == 1:
        state = np.outer(state,
                         state)  # turns state vector to a density matrix
    state = np.matrix(state)
    num = int(np.log2(len(state)))  # number of qubits
    phi_vals = np.linspace(0, np.pi, num=res,
                           dtype=np.complex_)
    theta_vals = np.linspace(0, 0.5*np.pi, num=res,
                             dtype=np.complex_)  # phi and theta values for WF
    w = np.empty([res, res])
    harr = np.sqrt(3)
    delta_su2 = np.zeros((2, 2), dtype=np.complex_)

    # create the spin Wigner function
    for theta in range(res):
        costheta = harr*np.cos(2*theta_vals[theta])
        sintheta = harr*np.sin(2*theta_vals[theta])

        for phi in range(res):
            delta_su2[0, 0] = 0.5*(1+costheta)
            delta_su2[0, 1] = -0.5*(np.exp(2j*phi_vals[phi])*sintheta)
            delta_su2[1, 0] = -0.5*(np.exp(-2j*phi_vals[phi])*sintheta)
            delta_su2[1, 1] = 0.5*(1-costheta)
            kernel = 1
            for _ in range(num):
                kernel = np.kron(kernel,
                                 delta_su2)  # creates phase point kernel

            w[phi, theta] = np.real(np.trace(state*kernel))  # Wigner function

    # Plot a sphere (x,y,z) with Wigner function facecolor data stored in Wc
    fig = plt.figure(figsize=(11, 9))
    ax = fig.gca(projection='3d')
    w_max = np.amax(w)
    # Color data for plotting
    w_c = cm.seismic_r((w+w_max)/(2*w_max))  # color data for sphere
    w_c2 = cm.seismic_r((w[0:res, int(res/2):res]+w_max)/(2*w_max))  # bottom
    w_c3 = cm.seismic_r((w[int(res/4):int(3*res/4), 0:res]+w_max) /
                        (2*w_max))  # side
    w_c4 = cm.seismic_r((w[int(res/2):res, 0:res]+w_max)/(2*w_max))  # back

    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))  # creates a sphere mesh

    ax.plot_surface(x, y, z, facecolors=w_c,
                    vmin=-w_max, vmax=w_max,
                    rcount=res, ccount=res,
                    linewidth=0, zorder=0.5,
                    antialiased=False)  # plots Wigner Bloch sphere

    ax.plot_surface(x[0:res, int(res/2):res],
                    y[0:res, int(res/2):res],
                    -1.5*np.ones((res, int(res/2))),
                    facecolors=w_c2,
                    vmin=-w_max, vmax=w_max,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False)  # plots bottom reflection

    ax.plot_surface(-1.5*np.ones((int(res/2), res)),
                    y[int(res/4):int(3*res/4), 0:res],
                    z[int(res/4):int(3*res/4), 0:res],
                    facecolors=w_c3,
                    vmin=-w_max, vmax=w_max,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False)  # plots side reflection

    ax.plot_surface(x[int(res/2):res, 0:res],
                    1.5*np.ones((int(res/2), res)),
                    z[int(res/2):res, 0:res],
                    facecolors=w_c4,
                    vmin=-w_max, vmax=w_max,
                    rcount=res/2, ccount=res/2,
                    linewidth=0, zorder=0.5,
                    antialiased=False)  # plots back reflection

    ax.w_xaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.w_yaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.w_zaxis.set_pane_color((0.4, 0.4, 0.4, 1.0))
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    m = cm.ScalarMappable(cmap=cm.seismic_r)
    m.set_array([-w_max, w_max])
    plt.colorbar(m, shrink=0.5, aspect=10)

    plt.show()


def plot_wigner_curve(wigner_data, xaxis=None):
    """Plots a curve for points in phase space of the spin Wigner function.

    Args:
        wigner_data(np.array): an array of points to plot as a 2d curve
        xaxis (np.array):  the range of the x axis
    """
    if not xaxis:
        xaxis = np.linspace(0, len(wigner_data)-1, num=len(wigner_data))

    plt.plot(xaxis, wigner_data)
    plt.show()


def plot_wigner_plaquette(wigner_data, max_wigner='local'):
    """Plots plaquette of wigner function data, the plaquette will
    consist of cicles each colored to match the value of the Wigner
    function at the given point in phase space.

    Args:
        wigner_data (matrix): array of Wigner function data where the
                            rows are plotted along the x axis and the
                            columns are plotted along the y axis
        max_wigner (str or float):
            - 'local' puts the maximum value to maximum of the points
            - 'unit' sets maximum to 1
            - float for a custom maximum.
    """
    wigner_data = np.matrix(wigner_data)
    dim = wigner_data.shape

    if max_wigner == 'local':
        w_max = np.amax(wigner_data)
    elif max_wigner == 'unit':
        w_max = 1
    else:
        w_max = max_wigner  # For a float input
    w_max = float(w_max)

    cmap = plt.cm.get_cmap('seismic_r')

    xax = dim[1]-0.5
    yax = dim[0]-0.5
    norm = np.amax(dim)

    fig = plt.figure(figsize=((xax+0.5)*6/norm, (yax+0.5)*6/norm))
    ax = fig.gca()

    for x in range(int(dim[1])):
        for y in range(int(dim[0])):
            circle = plt.Circle(
                (x, y), 0.49, color=cmap((wigner_data[y, x]+w_max)/(2*w_max)))
            ax.add_artist(circle)

    ax.set_xlim(-1, xax+0.5)
    ax.set_ylim(-1, yax+0.5)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    m = cm.ScalarMappable(cmap=cm.seismic_r)
    m.set_array([-w_max, w_max])
    plt.colorbar(m, shrink=0.5, aspect=10)
    plt.show()


def plot_wigner_data(wigner_data, phis=None, method=None):
    """Plots Wigner results in appropriate format.

    Args:
        wigner_data (numpy.array): Output returned from the wigner_data
            function
        phis (numpy.array): Values of phi
        method (str or None): how the data is to be plotted, methods are:
            point: a single point in phase space
            curve: a two dimensional curve
            plaquette: points plotted as circles
    """
    if not method:
        wig_dim = len(np.shape(wigner_data))
        if wig_dim == 1:
            if np.shape(wigner_data) == 1:
                method = 'point'
            else:
                method = 'curve'
        elif wig_dim == 2:
            method = 'plaquette'

    if method == 'curve':
        plot_wigner_curve(wigner_data, xaxis=phis)
    elif method == 'plaquette':
        plot_wigner_plaquette(wigner_data)
    elif method == 'state':
        plot_wigner_function(wigner_data)
    elif method == 'point':
        plot_wigner_plaquette(wigner_data)
        print('point in phase space is '+str(wigner_data))
    else:
        print("No method given")

###############################################################
# Plotting circuit
###############################################################


def plot_circuit(circuit, basis="u1,u2,u3,id,cx,x,y,z,h,s,t,rx,ry,rz"):
    """Plot and show circuit (opens new window, cannot inline in Jupyter)
    Defaults to an overcomplete basis, in order to not alter gates.
    Requires pdflatex installed (to compile Latex)
    Requires Qcircuit latex package (to compile latex)
    Requires poppler installed (to convert pdf to png)
    Requires pillow python package to handle images
    """
    im = circuit_drawer(circuit, basis)
    if im:
        im.show()


def circuit_drawer(circuit, basis="u1,u2,u3,id,cx,x,y,z,h,s,t,rx,ry,rz"):
    """Obtain the circuit in image format (output can be inlined in Jupyter)
    Defaults to an overcomplete basis, in order to not alter gates.
    Requires pdflatex installed (to compile Latex)
    Requires Qcircuit latex package (to compile latex)
    Requires poppler installed (to convert pdf to png)
    Requires pillow python package to handle images
    """
    filename = 'circuit'
    with tempfile.TemporaryDirectory() as tmpdirname:
        latex_drawer(circuit, os.path.join(tmpdirname, filename + '.tex'), basis=basis)
        im = None
        try:
            subprocess.run(["pdflatex", "-interaction=batchmode",
                            "-output-directory={}".format(tmpdirname),
                            "{}".format(filename + '.tex')],
                           stdout=subprocess.DEVNULL, check=True)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                logger.warning('WARNING: Unable to compile latex; `pdflatex` not installed. '
                               'Skipping circuit drawing...')
        except subprocess.CalledProcessError as e:
            logger.warning('WARNING: Unable to compile latex; `Qcircuit` package not installed. '
                           'Skipping circuit drawing...')
        else:
            try:
                subprocess.run(["pdftocairo", "-singlefile", "-png", "-q",
                                "{}".format(os.path.join(tmpdirname, filename + '.pdf'))])
                im = Image.open(filename + ".png")
                im = trim(im)
                os.remove(filename + ".png")
            except OSError as e:
                if e.errno == os.errno.ENOENT:
                    logger.warning('WARNING: Unable to convert pdf; `poppler` not installed. '
                                   'Skipping circuit drawing...')
                else:
                    raise
            except AttributeError:
                logger.warning('WARNING: `pillow` package not installed. '
                               'Skipping circuit drawing...')
    return im


def trim(im):
    """Trim image and remove white space
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    return im


def latex_drawer(circuit, filename=None, basis="u1,u2,u3,cx"):
    """Convert QuantumCircuit to LaTeX string.

    Args:
        circuit (QuantumCircuit): input circuit
        filename (str): optional filename to write latex
        basis (str): optional comma-separated list of gate names

    Returns:
        str: Latex string appropriate for writing to file.
    """
    ast = qasm.Qasm(data=circuit.qasm()).parse()
    if basis:
        # Split basis only if it is not the empty string.
        basis = basis.split(',')
    u = unroll.Unroller(ast, unroll.JsonBackend(basis))
    u.execute()
    json_circuit = u.backend.circuit
    qcimg = QCircuitImage(json_circuit)
    latex = qcimg.latex()
    if filename:
        with open(filename, 'w') as latex_file:
            latex_file.write(latex)
    return latex


class QCircuitImage(object):
    """This class contains methods to create \LaTeX circuit images.

    The class targets the \LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for QISKit.
    """
    def __init__(self, circuit, aliases=None):
        """
        Args:
            circuit (dict): compiled_circuit from qobj
            aliases (dict): dict mapping the current qubits in the circuit to
                new qubit names.
        """
        # compiled qobj circuit
        self.circuit = circuit

        # Map of qregs to sizes
        self.qregs = {}

        # Map of cregs to sizes
        self.cregs = {}

        # List of qregs and cregs in order of appearance in code and image
        self.ordered_regs = []

        # Map from registers to the list they appear in the image
        self.img_regs = {}

        # Array to hold the \LaTeX commands to generate a circuit image.
        self._latex = []

        # Variable to hold image depth (width)
        self.img_depth = 0

        # Variable to hold image width (height)
        self.img_width = 0

        #################################
        self.header = self.circuit['header']
        self.qregs = OrderedDict(_get_register_specs(
            self.header['qubit_labels']))
        self.qubit_list = []
        for qr in self.qregs:
            for i in range(self.qregs[qr]):
                self.qubit_list.append((qr, i))
        # TODO: clbit_labels has different format than qubit_labels
        # in qobj?
        self.cregs = OrderedDict()
        for item in self.header['clbit_labels']:
            self.cregs[item[0]] = item[1]
        self.clbit_list = []
        for cr in self.cregs:
            for i in range(self.cregs[cr]):
                self.clbit_list.append((cr, i))
        self.ordered_regs = [(item[0], item[1]) for
                             item in self.header['qubit_labels']]
        for clabel in self.header['clbit_labels']:
            for cind in range(clabel[1]):
                self.ordered_regs.append((clabel[0], cind))
        self.img_regs = {bit: ind for ind, bit in
                         enumerate(self.ordered_regs)}
        self.img_width = len(self.img_regs)
        self.wire_type = {}
        for key, value in self.ordered_regs:
            self.wire_type[(key, value)] = key in self.cregs.keys()

        self._initialize_latex_array(aliases=aliases)
        self._build_latex_array(aliases=aliases)

    def latex(self, aliases=None):
        """Return LaTeX string representation of circuit.

        This method uses the LaTeX Qconfig package to create a graphical
        representation of the circuit.

        Returns:
            string: for writing to a LaTeX file.
        """
        self._initialize_latex_array(aliases)
        self._build_latex_array(aliases)
        header_1 = r"""% \documentclass[preview]{standalone}
% If the image is too large to fit on this documentclass use
\documentclass[final]{beamer}
"""
        beamer_line = "\\usepackage[size=custom,width=%d,height=%d]{beamerposter}\n"
        header_2 = r"""% instead and customize the height and width (in cm) to fit.
% Large images may run out of memory quickly.
% To fix this use the LuaLaTeX compiler, which dynamically
% allocates memory.
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
% \usepackage[landscape]{geometry}
% Comment out the above line if using the beamer documentclass.
% Defines a measurement target symbol
\newcommand{\ctarg}{*+<.02em,.02em>{
\xy ="i","i"-<.39em,0em>;"i"+<.39em,0em> **\dir{-},
"i"-<0em,.39em>;"i"+<0em,.39em> **\dir{-},"i"*\xycircle<.4em>{} \endxy}}
\begin{document}
\begin{equation*}
    \Qcircuit @C=.5em @R=0em @!R {
"""
        output = StringIO()
        output.write(header_1)
        output.write('%% img_depth = %d, img_width = %d\n'
                     % (self.img_depth, self.img_width))
        # TODO: the scaling in the next line is arbitrary
        # TODO: and should be computed based on the column width
        output.write(beamer_line %
                     (3*self.img_depth, 1.7*self.img_width))
        output.write(header_2)
        for i in range(self.img_width):
            output.write("\t \t")
            for j in range(self.img_depth + 1):
                cell_str = self._latex[i][j]
                # floats can cause "Dimension too large" latex error in xymatrix
                # this truncates floats to avoid issue.
                cell_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}', _truncate_float,
                                  cell_str)
                output.write(cell_str)
                if j != self.img_depth:
                    output.write(" & ")
                elif j == self.img_depth:
                    output.write(r'\\'+'\n')
                else:
                    output.write('\n')
        output.write('\t }\n')
        output.write('\end{equation*}\n\n')
        output.write('\end{document}')
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self, aliases=None):
        # pylint: disable=unused-argument
        self.img_depth = len(self.circuit['operations'])
        self._latex = [
            ["\\cw" if self.wire_type[self.ordered_regs[j]]
             else "\\qw" for i in range(self.img_depth + 1)]
            for j in range(self.img_width)]
        self._latex.append([" "] * (self.img_depth + 1))
        for i in range(self.img_width):
            if self.wire_type[self.ordered_regs[i]]:
                self._latex[i][0] = "\\lstick{" + self.ordered_regs[i][0] + \
                                    "_" + str(self.ordered_regs[i][1]) + "}"
            else:
                self._latex[i][0] = "\\lstick{\\ket{" + \
                                    self.ordered_regs[i][0] + "_" + \
                                    str(self.ordered_regs[i][1]) + "}}"

    def _get_image_depth(self, aliases=None):
        columns = 2
        is_occupied = [False] * self.img_width
        for op in self.circuit['operations']:
            if 'clbits' not in op:
                if op['name'] != 'barrier':
                    qarglist = [self.qubit_list[i] for i in op['qubits']]
                    if aliases is not None:
                        qarglist = map(lambda x: aliases[x], qarglist)
                    if len(qarglist) == 1:
                        pos_1 = self.img_regs[(qarglist[0][0],
                                               qarglist[0][1])]
                        if 'conditional' in op:
                            mask = int(op['conditional']['mask'], 16)
                            cl_reg = self.clbit_list[self._ffs(mask)]
                            if_reg = cl_reg[0]
                            pos_2 = self.img_regs[cl_reg]
                            for i in range(pos_1, pos_2 + self.cregs[if_reg] + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_1, pos_2 + 1):
                                        is_occupied[j] = True
                                    break
                        else:
                            if is_occupied[pos_1] is False:
                                is_occupied[pos_1] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                is_occupied[pos_1] = True
                    elif len(qarglist) == 2:
                        pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                        pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                        if 'conditional' in op:
                            mask = int(op['conditional']['mask'], 16)
                            cl_reg = self.clbit_list[self._ffs(mask)]
                            if_reg = cl_reg[0]
                            pos_3 = self.img_regs[(if_reg, 0)]
                            if pos_1 > pos_2:
                                for i in range(pos_2, pos_3 + self.cregs[if_reg] + 1):
                                    if is_occupied[i] is False:
                                        is_occupied[i] = True
                                    else:
                                        columns += 1
                                        is_occupied = [False] * self.img_width
                                        for j in range(pos_2, pos_3 + 1):
                                            is_occupied[j] = True
                                        break
                            else:
                                for i in range(pos_1, pos_3 + self.cregs[if_reg] + 1):
                                    if is_occupied[i] is False:
                                        is_occupied[i] = True
                                    else:
                                        columns += 1
                                        is_occupied = [False] * self.img_width
                                        for j in range(pos_1, pos_3 + 1):
                                            is_occupied[j] = True
                                        break
                        else:
                            temp = [pos_1, pos_2]
                            temp.sort(key=int)
                            top = temp[0]
                            bottom = temp[1]

                            for i in range(top, bottom + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(top, bottom + 1):
                                        is_occupied[j] = True
                                    break
            else:
                if op['name'] == "measure":
                    assert len(op['clbits']) == 1 and len(op['qubits']) == 1
                    if 'conditional' in op:
                        assert False,\
                            "If controlled measures currently not supported."
                    qname, qindex = self.total_2_register_index(
                        op['qubits'][0], self.qregs)
                    cname, cindex = self.total_2_register_index(
                        op['clbits'][0], self.cregs)
                    if aliases:
                        newq = aliases[(qname, qindex)]
                        qname = newq[0]
                        qindex = newq[1]
                    pos_1 = self.img_regs[(qname, qindex)]
                    pos_2 = self.img_regs[(cname, cindex)]
                    temp = [pos_1, pos_2]
                    temp.sort(key=int)
                    [pos_1, pos_2] = temp
                    for i in range(pos_1, pos_2 + 1):
                        if is_occupied[i] is False:
                            is_occupied[i] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            for j in range(pos_1, pos_2 + 1):
                                is_occupied[j] = True
                            break

                else:
                    assert False, "bad node data"
        return columns+1

    def total_2_register_index(self, index, registers):
        """Get register name for qubit index.

        This function uses the self.qregs ordered dictionary, which looks like
        {'qr1': 2, 'qr2', 3}
        to get the register name for the total qubit index. For the above example,
        index in [0,1] returns 'qr1' and index in [2,4] returns 'qr2'.

        Args:
            index (int): total qubit index among all quantum registers
            registers (OrderedDict): OrderedDict as described above.
        Returns:
            str: name of register associated with qubit index.
        Raises:
            ValueError: if the qubit index lies outside the range of qubit
                registers.
        """
        count = 0
        for name, size in registers.items():
            if count + size > index:
                return name, index - count
            else:
                count += size
        raise ValueError('qubit index lies outside range of qubit registers')

    def _build_latex_array(self, aliases=None):
        """Returns an array of strings containing \LaTeX for this circuit.

        If aliases is not None, aliases contains a dict mapping
        the current qubits in the circuit to new qubit names.
        We will deduce the register names and sizes from aliases.
        """
        columns = 1
        is_occupied = [False] * self.img_width

        # Rename qregs if necessary
        if aliases:
            qregdata = {}
            for q in aliases.values():
                if q[0] not in qregdata:
                    qregdata[q[0]] = q[1] + 1
                elif qregdata[q[0]] < q[1] + 1:
                    qregdata[q[0]] = q[1] + 1
        else:
            qregdata = self.qregs

        for _, op in enumerate(self.circuit['operations']):
            # print(iop, op)
            if 'conditional' in op:
                mask = int(op['conditional']['mask'], 16)
                cl_reg = self.clbit_list[self._ffs(mask)]
                if_reg = cl_reg[0]
                pos_2 = self.img_regs[cl_reg]
                if_value = format(int(op['conditional']['val'], 16),
                                  'b').zfill(self.cregs[if_reg])[::-1]
            if 'clbits' not in op:
                nm = op['name']
                if nm != 'barrier':
                    qarglist = [self.qubit_list[i] for i in op['qubits']]
                    if aliases is not None:
                        qarglist = map(lambda x: aliases[x], qarglist)
                    if len(qarglist) == 1:
                        pos_1 = self.img_regs[(qarglist[0][0],
                                               qarglist[0][1])]
                        if 'conditional' in op:
                            mask = int(op['conditional']['mask'], 16)
                            cl_reg = self.clbit_list[self._ffs(mask)]
                            if_reg = cl_reg[0]
                            pos_2 = self.img_regs[cl_reg]
                            for i in range(pos_1, pos_2 + self.cregs[if_reg] + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_1, pos_2 + 1):
                                        is_occupied[j] = True
                                    break

                            if nm == "x":
                                self._latex[pos_1][columns] = "\\gate{X}"
                            elif nm == "y":
                                self._latex[pos_1][columns] = "\\gate{Y}"
                            elif nm == "z":
                                self._latex[pos_1][columns] = "\\gate{Z}"
                            elif nm == "h":
                                self._latex[pos_1][columns] = "\\gate{H}"
                            elif nm == "s":
                                self._latex[pos_1][columns] = "\\gate{S}"
                            elif nm == "sdg":
                                self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                            elif nm == "t":
                                self._latex[pos_1][columns] = "\\gate{T}"
                            elif nm == "tdg":
                                self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                            elif nm == "u1":
                                self._latex[pos_1][columns] = "\\gate{U(0,0,%s)}" % (
                                    op["texparams"][0])
                            elif nm == "u2":
                                self._latex[pos_1][columns] =\
                                    "\\gate{U\\left(\\frac{\pi}{2},%s,%s\\right)}" % (
                                        op["texparams"][0], op["texparams"][1])
                            elif nm == "u3":
                                self._latex[pos_1][columns] = "\\gate{U(%s,%s,%s)}" \
                                    % (op["texparams"][0], op["texparams"][1], op["texparams"][2])
                            elif nm == "rx":
                                self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                    op["texparams"][0])
                            elif nm == "ry":
                                self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                    op["texparams"][0])
                            elif nm == "rz":
                                self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                    op["texparams"][0])

                            gap = pos_2 - pos_1
                            for i in range(self.cregs[if_reg]):
                                if if_value[i] == '1':
                                    self._latex[pos_2 + i][columns] = \
                                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                                else:
                                    self._latex[pos_2 + i][columns] = \
                                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1

                        else:
                            if not is_occupied[pos_1]:
                                is_occupied[pos_1] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                is_occupied[pos_1] = True

                            if nm == "x":
                                self._latex[pos_1][columns] = "\\gate{X}"
                            elif nm == "y":
                                self._latex[pos_1][columns] = "\\gate{Y}"
                            elif nm == "z":
                                self._latex[pos_1][columns] = "\\gate{Z}"
                            elif nm == "h":
                                self._latex[pos_1][columns] = "\\gate{H}"
                            elif nm == "s":
                                self._latex[pos_1][columns] = "\\gate{S}"
                            elif nm == "sdg":
                                self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                            elif nm == "t":
                                self._latex[pos_1][columns] = "\\gate{T}"
                            elif nm == "tdg":
                                self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                            elif nm == "u1":
                                self._latex[pos_1][columns] = "\\gate{U(0,0,%s)}" % (
                                    op["texparams"][0])
                            elif nm == "u2":
                                self._latex[pos_1][columns] = \
                                    "\\gate{U\\left(\\frac{\pi}{2},%s,%s\\right)}" % (
                                        op["texparams"][0], op["texparams"][1])
                            elif nm == "u3":
                                self._latex[pos_1][columns] = "\\gate{U(%s,%s,%s)}" \
                                    % (op["texparams"][0], op["texparams"][1], op["texparams"][2])
                            elif nm == "rx":
                                self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                    op["texparams"][0])
                            elif nm == "ry":
                                self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                    op["texparams"][0])
                            elif nm == "rz":
                                self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                    op["texparams"][0])

                    elif len(qarglist) == 2:
                        pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                        pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                        if 'conditional' in op:
                            pos_3 = self.img_regs[(if_reg, 0)]

                            if pos_1 > pos_2:
                                for i in range(pos_2, pos_3 + self.cregs[if_reg] + 1):
                                    if is_occupied[i] is False:
                                        is_occupied[i] = True
                                    else:
                                        columns += 1
                                        is_occupied = [False] * self.img_width
                                        for j in range(pos_2, pos_3 + 1):
                                            is_occupied[j] = True
                                        break

                                if nm == "cx":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = "\\targ"
                                elif nm == "cz":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = "\\gate{Z}"
                                elif nm == "cy":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = "\\gate{Y}"
                                elif nm == "ch":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = "\\gate{H}"
                                elif nm == "swap":
                                    self._latex[pos_1][columns] = "\\qswap"
                                    self._latex[pos_2][columns] = "\\qwx"
                                elif nm == "crz":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = \
                                        "\\gate{R_z(%s)}" % (op["texparams"][0])
                                elif nm == "cu1":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = \
                                        "\\gate{U(0,0,%s)}" % (op["texparams"][0])
                                elif nm == "cu3":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                    self._latex[pos_2][columns] = \
                                        "\\gate{U(%s,%s,%s)}" % (op["texparams"][0],
                                                                 op["texparams"][1],
                                                                 op["texparams"][2])

                                gap = pos_3 - pos_1
                                for i in range(self.cregs[if_reg]):
                                    if if_value[i] == '1':
                                        self._latex[pos_3 + i][columns] = \
                                            "\\control \\cw \\cwx[-" + str(gap) + "]"
                                        gap = 1
                                    else:
                                        self._latex[pos_3 + i][columns] = \
                                            "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                        gap = 1
                            else:
                                for i in range(pos_1, pos_3 + self.cregs[if_reg]):
                                    if is_occupied[i] is False:
                                        is_occupied[i] = True
                                    else:
                                        columns += 1
                                        is_occupied = [False] * self.img_width
                                        for j in range(pos_1, pos_3 + 1):
                                            is_occupied[j] = True
                                        break

                                if nm == "cx":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = "\\targ"
                                elif nm == "cz":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = "\\gate{Z}"
                                elif nm == "cy":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = "\\gate{Y}"
                                elif nm == "ch":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = "\\gate{H}"
                                elif nm == "swap":
                                    self._latex[pos_1][columns] = "\\qswap"
                                    self._latex[pos_2][columns] = "\\qwx"
                                elif nm == "crz":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = \
                                        "\\gate{R_z(%s)}" % (op["texparams"][0])
                                elif nm == "cu1":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = \
                                        "\\gate{U(0,0,%s)}" % (op["texparams"][0])
                                elif nm == "cu3":
                                    self._latex[pos_1][columns] = \
                                        "\\ctrl{" + str(pos_1 - pos_2) + "}"
                                    self._latex[pos_2][columns] = "\\gate{U(%s,%s,%s)}" \
                                        % (op["texparams"][0], op["texparams"][1],
                                           op["texparams"][2])

                                gap = pos_3 - pos_2
                                for i in range(self.cregs[if_reg]):
                                    if if_value[i] == '1':
                                        self._latex[pos_3 + i][columns] = \
                                            "\\control \\cw \\cwx[-" + str(gap) + "]"
                                        gap = 1
                                    else:
                                        self._latex[pos_3 + i][columns] = \
                                            "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                        gap = 1

                        else:
                            temp = [pos_1, pos_2]
                            temp.sort(key=int)
                            top = temp[0]
                            bottom = temp[1]

                            for i in range(top, bottom + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(top, bottom + 1):
                                        is_occupied[j] = True
                                    break

                            if nm == "cx":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\targ"
                            elif nm == "cz":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\gate{Z}"
                            elif nm == "cy":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\gate{Y}"
                            elif nm == "ch":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\gate{H}"
                            elif nm == "swap":
                                self._latex[pos_1][columns] = "\\qswap"
                                self._latex[pos_2][columns] = "\\qwx"
                            elif nm == "crz":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = \
                                    "\\gate{R_z(%s)}" % (op["texparams"][0])
                            elif nm == "cu1":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = \
                                    "\\gate{U(0,0,%s)}" % (op["texparams"][0])
                            elif nm == "cu3":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\gate{U(%s,%s,%s)}" \
                                    % (op["texparams"][0], op["texparams"][1], op["texparams"][2])

                    elif len(qarglist) == 3:
                        pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                        pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]
                        pos_3 = self.img_regs[(qarglist[2][0], qarglist[2][1])]

                        if 'conditional' in op:
                            pos_4 = self.img_regs[(if_reg, 0)]

                            temp = [pos_1, pos_2, pos_3, pos_4]
                            temp.sort(key=int)
                            top = temp[0]
                            bottom = temp[2]

                            for i in range(top, pos_4 + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(top, pos_4 + 1):
                                        is_occupied[j] = True
                                    break

                            gap = pos_4 - bottom
                            for i in range(self.cregs[if_reg]):
                                if if_value[i] == '1':
                                    self._latex[pos_4 + i][columns] = \
                                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                                else:
                                    self._latex[pos_4 + i][columns] = \
                                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                        else:
                            temp = [pos_1, pos_2, pos_3]
                            temp.sort(key=int)
                            top = temp[0]
                            bottom = temp[2]

                            for i in range(top, bottom + 1):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(top, bottom + 1):
                                        is_occupied[j] = True
                                    break

                            if nm == "ccx":
                                self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][columns] = "\\ctrl{" + str(pos_3 - pos_2) + "}"
                                self._latex[pos_3][columns] = "\\targ"

            else:
                if op["name"] == "measure":
                    assert len(op['clbits']) == 1 and \
                        len(op['qubits']) == 1 and \
                        'params' not in op, "bad operation record"

                    if 'conditional' in op:
                        assert False, "If controlled measures currently not supported."
                    qname, qindex = self.total_2_register_index(
                        op['qubits'][0], self.qregs)
                    cname, cindex = self.total_2_register_index(
                        op['clbits'][0], self.cregs)

                    if aliases:
                        newq = aliases[(qname, qindex)]
                        qname = newq[0]
                        qindex = newq[1]

                    pos_1 = self.img_regs[(qname, qindex)]
                    pos_2 = self.img_regs[(cname, cindex)]

                    for i in range(pos_1, pos_2 + 1):
                        if is_occupied[i] is False:
                            is_occupied[i] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            for j in range(pos_1, pos_2 + 1):
                                is_occupied[j] = True
                            break

                    try:
                        self._latex[pos_1][columns] = "\\meter"
                        self._latex[pos_2][columns] = \
                            "\\ctarg \\cw \\cwx[-" + str(pos_2 - pos_1) + "]"
                    except Exception as e:
                        raise QISKitError('Error during Latex building: %s' %
                                          str(e))
                else:
                    assert False, "bad node data"

    def _ffs(self, mask):
        """Find index of first set bit.

        Args:
            mask (int): integer to search
        Returns:
            int: index of the first set bit.
        """
        return (mask & (-mask)).bit_length() - 1


def _get_register_specs(bit_labels):
    """
    Get the number and size of unique registers from bit_labels list.

    TODO: this function also appears in _projectq_simulator.py. Perhaps it
    should be placed in _quantumcircuit.py or tools.

    Args:
        bit_labels (list): this list is of the form::

            [['reg1', 0], ['reg1', 1], ['reg2', 0]]

            which indicates a register named "reg1" of size 2
            and a register named "reg2" of size 1. This is the
            format of classic and quantum bit labels in qobj
            header.

    Yields:
        tuple: iterator of register_name:size pairs.
    """
    it = itertools.groupby(bit_labels, operator.itemgetter(0))
    for register_name, sub_it in it:
        yield register_name, max(ind[1] for ind in sub_it) + 1


def _truncate_float(matchobj, format_str='0.2g'):
    """Truncate long floats

    Args:
        matchobj (re.Match): contains original float
        format_str (str): format specifier
    Returns:
       str: returns truncated float
    """
    if matchobj.group(0):
        return format(float(matchobj.group(0)), format_str)
    return ''
