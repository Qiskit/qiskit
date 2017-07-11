# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# This file is intended only for use during the USEQIP Summer School 2017.
# Do not distribute.
# It is provided without warranty or conditions of any kind, either express or
# implied.
# An open source version of this file will be included in QISKIT-DEV-PY
# reposity in the future. Keep an eye on the Github repository for updates!
# https://github.com/IBM/qiskit-sdk-py
# =============================================================================

"""
Quantum state tomography using the maximum likelihood reconstruction method
from Smolin, Gambetta, Smith Phys. Rev. Lett. 108, 070502  (arXiv: 1106.5458)

Author: Christopher J. Wood <cjwood@us.ibm.com>
        Jay Gambetta
        Andrew Cross
"""
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from tools.pauli import pauli_group


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


COMPLEMENT = {'1': '0', '0': '1'}


def compliment(value):
    """Swap 1 and 0 in a vector."""
    return ''.join(COMPLEMENT[x] for x in value)


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


def plot_qsphere(state, number_of_qubits):
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
    for i in range(2**number_of_qubits):
        element = bin(i)[2:].zfill(number_of_qubits)
        weight = element.count("1")
        zvalue = -2 * weight / d + 1
        number_of_divisions = n_choose_k(d, weight)
        weight_order = bit_string_index(element)
        #if weight_order >= number_of_divisions / 2:
        #    com_key = compliment(element)
        #    weight_order_temp = bit_string_index(com_key)
        #    weight_order = np.floor(
        #        number_of_divisions / 2) + weight_order_temp + 1
        angle = (weight_order) * 2 * np.pi / number_of_divisions
        xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
        yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)
        linewidth = np.real(np.dot(state[i], state[i].conj()))
        angleround = int(np.angle(state[i])/(2*np.pi)*12)
        if angleround == 0:
            colorstate = (1, 0, 0)
        elif angleround == 1:
            colorstate = (1, 0.5, 0)
        elif angleround == 2:
            colorstate = (1, 1, 0)
        elif angleround == 3:
            colorstate = (0.5, 1, 0)
        elif angleround == 4:
            colorstate = (0, 1, 0)
        elif angleround == 5:
            colorstate = (0, 1, 0.5)
        elif angleround == 6:
            colorstate = (0, 1, 1)
        elif angleround == 7:
            colorstate = (0, 0.5, 1)
        elif angleround == 8:
            colorstate = (0, 0, 1)
        elif angleround == 9:
            colorstate = (0.5, 0, 1)
        elif angleround == 10:
            colorstate = (1, 0, 1)
        elif angleround == 11:
            colorstate = (1, 0, 0.5)
        # print("outcome = " + element + " weight " + str(weight) + " angle " + str(angle) + " amp " + str(linewidth))
        a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue], mutation_scale=20,
                    alpha=linewidth, arrowstyle="-|>", color=colorstate)
        ax.plot([xvalue], [yvalue], [zvalue], markerfacecolor=(.5,.5,.5), markeredgecolor=(.5,.5,.5), marker='o', markersize=10, alpha=1)
        ax.add_artist(a)
    for weight in range(d + 1):
        theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        z = -2 * weight / d + 1
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, z, color=(.5,.5,.5))
    ax.plot([0], [0], [0], markerfacecolor=(.5,.5,.5), markeredgecolor=(.5,.5,.5), marker='o', markersize=10, alpha=1)
    plt.show()


def plot_bloch_vector(bloch, number_of_qubits, title=""):
    """Plot a Bloch vector.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.
    bloch is a array j*3+i where i is x y and z for qubit j
    title is a string, the plot title
    """
    # Set arrow lengths
    arlen = 1.3

    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    for j in range(number_of_qubits):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("equal")
        ax.plot_surface(x, y, z, color="b", alpha=0.1)

        # Plot arrows (axes, Bloch vector, its projections)
        xa = Arrow3D([0, arlen], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ya = Arrow3D([0, 0], [0, arlen], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        za = Arrow3D([0, 0], [0, 0], [0, arlen], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        a = Arrow3D([0, bloch[j*3+0]], [0, bloch[j*3+1]], [0, bloch[j*3+2]], mutation_scale=20, lw=2, arrowstyle="simple", color="k")
        bax = Arrow3D([0, bloch[j*3+0]], [0, 0], [0, 0], mutation_scale=20, lw=2, arrowstyle="-", color="r")
        bay = Arrow3D([0, 0], [0, bloch[j*3+1]], [0, 0], mutation_scale=20, lw=2, arrowstyle="-", color="g")
        baz = Arrow3D([0, 0], [0, 0], [0, bloch[j*3+2]], mutation_scale=20, lw=2, arrowstyle="-", color="b")
        arrowlist = [xa, ya, za, a, bax, bay, baz]
        for arr in arrowlist:
            ax.add_artist(arr)

        # Rotate the view
        ax.view_init(30, 30)

        # Annotate the axes, shifts are ad-hoc for this (30, 30) view
        xp, yp, _ = proj3d.proj_transform(arlen, 0, 0, ax.get_proj())
        plt.annotate("x", xy=(xp, yp), xytext=(-3, -8), textcoords='offset points', ha='right', va='bottom')
        xp, yp, _ = proj3d.proj_transform(0, arlen, 0, ax.get_proj())
        plt.annotate("y", xy=(xp, yp), xytext=(6, -5), textcoords='offset points', ha='right', va='bottom')
        xp, yp, _ = proj3d.proj_transform(0, 0, arlen, ax.get_proj())
        plt.annotate("z", xy=(xp, yp), xytext=(2, 0), textcoords='offset points', ha='right', va='bottom')

        plt.title("qubit " + str(j))
        plt.show()


def plot_state(rho, number_of_qubits, method='city'):
    """Plot the cityscape of quantum state."""
    if int(np.log2(len(rho))) == number_of_qubits:
        # Need updating to check its a matrix
        if method == 'city':
            datareal = np.real(rho)
            dataimag = np.imag(rho)

            column_names = [bin(i)[2:].zfill(number_of_qubits) for i in range(2**number_of_qubits)]
            row_names = [bin(i)[2:].zfill(number_of_qubits) for i in range(2**number_of_qubits)]

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
            ax1.set_zticks(np.arange(-1, 1, 0.5))
            ax1.w_xaxis.set_ticklabels(column_names, fontsize=12)
            ax1.w_yaxis.set_ticklabels(row_names, fontsize=12)
            ax1.set_xlabel('basis state', fontsize=12)
            ax1.set_ylabel('basis state', fontsize=12)
            ax1.set_zlabel('probability', fontsize=12)
            ax1.set_title("Real[rho]")

            ax2.set_xticks(np.arange(0.5, lx+0.5, 1))
            ax2.set_yticks(np.arange(0.5, ly+0.5, 1))
            ax2.set_zticks(np.arange(-1, 1, 0.5))
            ax2.w_xaxis.set_ticklabels(column_names, fontsize=12)
            ax2.w_yaxis.set_ticklabels(row_names, fontsize=12)
            ax2.set_xlabel('basis state', fontsize=12)
            ax2.set_ylabel('basis state', fontsize=12)
            ax2.set_zlabel('probability', fontsize=12)
            ax2.set_title("Imag[rho]")

            plt.show()
        elif method == "paulivec":
            labels = list(map(lambda x: x.to_label(), pauli_group(number_of_qubits)))
            values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))), pauli_group(number_of_qubits)))
            numelem = len(values)
            ind = np.arange(numelem)  # the x locations for the groups
            width = 0.5  # the width of the bars
            fig, ax = plt.subplots()
            ax.grid(zorder=0)
            ax.bar(ind, values, width, color='seagreen')

            # add some text for labels, title, and axes ticks
            ax.set_ylabel('Expectation value', fontsize=12)
            ax.set_xticks(ind)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_xlabel('Pauli', fontsize=12)
            ax.set_ylim([-1, 1])
            plt.show()
        else:
            print("No method given")
    else:
        print("size of rho is not equal to number of qubits")
