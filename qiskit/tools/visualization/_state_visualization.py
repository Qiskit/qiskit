# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string,ungrouped-imports,import-error

"""
Visualization functions for quantum states.
"""

import warnings
from functools import reduce
import numpy as np
from scipy import linalg
from qiskit.quantum_info import pauli_group, Pauli
from ._matplotlib import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from matplotlib import cm
    from matplotlib.ticker import MaxNLocator
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.colors as mcolors
    from matplotlib.colors import Normalize, LightSource
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from qiskit.tools.visualization._error import VisualizationError
    from qiskit.tools.visualization._bloch import Bloch
    from qiskit.tools.visualization._utils import _validate_input_state

if HAS_MATPLOTLIB:
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


def plot_state_hinton(rho, title='', figsize=None):
    """Plot a hinton diagram for the quanum state.

    Args:
        rho (ndarray): Numpy array for state vector or density matrix.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
    Returns:
         matplotlib.Figure: The matplotlib.Figure of the visualization

    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    rho = _validate_input_state(rho)
    if figsize is None:
        figsize = (8, 5)
    num = int(np.log2(len(rho)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    max_weight = 2 ** np.ceil(np.log(np.abs(rho).max()) / np.log(2))
    datareal = np.real(rho)
    dataimag = np.imag(rho)
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    lx = len(datareal[0])            # Work out matrix dimensions
    ly = len(datareal[:, 0])
    # Real
    ax1.patch.set_facecolor('gray')
    ax1.set_aspect('equal', 'box')
    ax1.xaxis.set_major_locator(plt.NullLocator())
    ax1.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(datareal):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax1.add_patch(rect)

    ax1.set_xticks(np.arange(0, lx+0.5, 1))
    ax1.set_yticks(np.arange(0, ly+0.5, 1))
    ax1.set_yticklabels(row_names, fontsize=14)
    ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
    ax1.autoscale_view()
    ax1.invert_yaxis()
    ax1.set_title('Real[rho]', fontsize=14)
    # Imaginary
    ax2.patch.set_facecolor('gray')
    ax2.set_aspect('equal', 'box')
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(dataimag):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax2.add_patch(rect)
    if np.any(dataimag != 0):
        ax2.set_xticks(np.arange(0, lx+0.5, 1))
        ax2.set_yticks(np.arange(0, ly+0.5, 1))
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
    ax2.autoscale_view()
    ax2.invert_yaxis()
    ax2.set_title('Imag[rho]', fontsize=14)
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_bloch_vector(bloch, title="", ax=None, figsize=None):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        bloch (list[double]): array of three elements where [<x>, <y>, <z>]
        title (str): a string that represents the plot title
        ax (matplotlib.Axes): An Axes to use for rendering the bloch sphere
        figsize (tuple): Figure size in inches. Has no effect is passing `ax`.

    Returns:
        Figure: A matplotlib figure instance if `ax = None`.

    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if figsize is None:
        figsize = (5, 5)
    B = Bloch(axes=ax)
    B.add_vectors(bloch)
    B.render(title=title)
    if ax is None:
        fig = B.fig
        fig.set_size_inches(figsize[0], figsize[1])
        plt.close(fig)
        return fig
    return None


def plot_bloch_multivector(rho, title='', figsize=None):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        rho (ndarray): Numpy array for state vector or density matrix.
        title (str): a string that represents the plot title
        figsize (tuple): Has no effect, here for compatibility only.

    Returns:
        Figure: A matplotlib figure instance if `ax = None`.

    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    rho = _validate_input_state(rho)
    num = int(np.log2(len(rho)))
    width, height = plt.figaspect(1/num)
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1, projection='3d')
        pauli_singles = [
            Pauli.pauli_single(num, i, 'X'),
            Pauli.pauli_single(num, i, 'Y'),
            Pauli.pauli_single(num, i, 'Z')
        ]
        bloch_state = list(
            map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                pauli_singles))
        plot_bloch_vector(bloch_state, "qubit " + str(i), ax=ax,
                          figsize=figsize)
    fig.suptitle(title, fontsize=16)
    plt.close(fig)
    return fig


def plot_state_city(rho, title="", figsize=None, color=None,
                    alpha=1):
    """Plot the cityscape of quantum state.

    Plot two 3d bar graphs (two dimensional) of the real and imaginary
    part of the density matrix rho.

    Args:
        rho (ndarray): Numpy array for state vector or density matrix.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list): A list of len=2 giving colors for real and
        imaginary components of matrix elements.
        alpha (float): Transparency value for bars
    Returns:
         matplotlib.Figure: The matplotlib.Figure of the visualization

    Raises:
        ImportError: Requires matplotlib.
        ValueError: When 'color' is not a list of len=2.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    rho = _validate_input_state(rho)

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

    if color is None:
        color = ["#648fff", "#648fff"]
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        if color[0] is None:
            color[0] = "#648fff"
        if color[1] is None:
            color[1] = "#648fff"

    # set default figure size
    if figsize is None:
        figsize = (15, 5)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    x = [0, max(xpos)+0.5, max(xpos)+0.5, 0]
    y = [0, 0, max(ypos)+0.5, max(ypos)+0.5]
    z = [0, 0, 0, 0]
    verts = [list(zip(x, y, z))]

    fc1 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzr, color[0])
    for idx in range(len(zpos)):  # pylint: disable=consider-using-enumerate
        if dzr[idx] > 0:
            zorder = 2
        else:
            zorder = 0
        b1 = ax1.bar3d(xpos[idx], ypos[idx], zpos[idx],
                       dx[idx], dy[idx], dzr[idx],
                       alpha=alpha, zorder=zorder)
        b1.set_facecolors(fc1[6*idx:6*idx+6])

    pc1 = Poly3DCollection(verts, alpha=0.15, facecolor='k',
                           linewidths=1, zorder=1)

    if min(dzr) < 0 and max(dzr) > 0:
        ax1.add_collection3d(pc1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
    for idx in range(len(zpos)):  # pylint: disable=consider-using-enumerate
        if dzi[idx] > 0:
            zorder = 2
        else:
            zorder = 0
        b2 = ax2.bar3d(xpos[idx], ypos[idx], zpos[idx],
                       dx[idx], dy[idx], dzi[idx],
                       alpha=alpha, zorder=zorder)
        b2.set_facecolors(fc2[6*idx:6*idx+6])

    pc2 = Poly3DCollection(verts, alpha=0.2, facecolor='k',
                           linewidths=1, zorder=1)

    if min(dzi) < 0 and max(dzi) > 0:
        ax2.add_collection3d(pc2)

    ax1.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax1.set_yticks(np.arange(0.5, ly+0.5, 1))
    ax1.axes.set_zlim3d(np.min(dzr), np.max(dzr)+1e-9)
    ax1.zaxis.set_major_locator(MaxNLocator(5))
    ax1.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45)
    ax1.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5)
    ax1.set_zlabel("Real[rho]", fontsize=14)
    for tick in ax1.zaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax2.set_yticks(np.arange(0.5, ly+0.5, 1))
    if np.min(dzi) != np.max(dzi):
        eps = 0
        ax2.zaxis.set_major_locator(MaxNLocator(5))
    else:
        ax2.set_zticks([0])
        eps = 1e-9
    ax2.axes.set_zlim3d(np.min(dzi), np.max(dzi)+eps)
    ax2.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45)
    ax2.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5)
    # ax2.set_xlabel('basis state', fontsize=12)
    # ax2.set_ylabel('basis state', fontsize=12)
    ax2.set_zlabel("Imag[rho]", fontsize=14)
    for tick in ax2.zaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_state_paulivec(rho, title="", figsize=None, color=None):
    """Plot the paulivec representation of a quantum state.

    Plot a bargraph of the mixed state rho over the pauli matrices

    Args:
        rho (ndarray): Numpy array for state vector or density matrix
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list or str): Color of the expectation value bars.
    Returns:
         matplotlib.Figure: The matplotlib.Figure of the visualization
    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    rho = _validate_input_state(rho)
    if figsize is None:
        figsize = (7, 5)
    num = int(np.log2(len(rho)))
    labels = list(map(lambda x: x.to_label(), pauli_group(num)))
    values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                      pauli_group(num)))
    numelem = len(values)
    if color is None:
        color = "#648fff"

    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(zorder=0, linewidth=1, linestyle='--')
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color='k')
    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Expectation value', fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel('Pauli', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor('#eeeeee')
    for tick in ax.xaxis.get_major_ticks()+ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    plt.close(fig)
    return fig


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

    Raises:
        VisualizationError: if length of list is not equal to k
    """
    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_color_wheel(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to a color wheel with blue at zero phase.
    """
    angles = np.angle(complex_number)
    angle_round = int(((angles + 2 * np.pi) % (2 * np.pi))/np.pi*6)
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


def plot_state_qsphere(rho, figsize=None):
    """Plot the qsphere representation of a quantum state.

    Args:
        rho (ndarray): State vector or density matrix representation.
        of quantum state.
        figsize (tuple): Figure size in inches.

    Returns:
        Figure: A matplotlib figure instance.

    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    rho = _validate_input_state(rho)
    if figsize is None:
        figsize = (7, 7)
    num = int(np.log2(len(rho)))
    # get the eigenvectors and eigenvalues
    we, stateall = linalg.eigh(rho)
    for _ in range(2**num):
        # start with the max
        probmix = we.max()
        prob_location = we.argmax()
        if probmix > 0.001:
            # get the max eigenvalue
            state = stateall[:, prob_location]
            loc = np.absolute(state).argmax()
            # get the element location closes to lowest bin representation.
            for j in range(2**num):
                test = np.absolute(np.absolute(state[j]) -
                                   np.absolute(state[loc]))
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
            fig = plt.figure(figsize=figsize)
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
            we[prob_location] = 0
        else:
            break
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_state(quantum_state, method='city', figsize=None):
    """Plot the quantum state.

    Args:
        quantum_state (ndarray): statevector or density matrix
                                 representation of a quantum state.
        method (str): Plotting method to use.
        figsize (tuple): Figure size in inches,

    Returns:
         matplotlib.Figure: The matplotlib.Figure of the visualization
    Raises:
        ImportError: Requires matplotlib.
        VisualizationError: if the input is not a statevector or density
        matrix, or if the state is not an multi-qubit quantum state.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    warnings.warn("plot_state is deprecated, and will be removed in \
                  the 0.9 release. Use the plot_state_ * functions \
                  instead.",
                  DeprecationWarning)
    # Check if input is a statevector, and convert to density matrix
    rho = _validate_input_state(quantum_state)
    fig = None
    if method == 'city':
        fig = plot_state_city(rho, figsize=figsize)
    elif method == "paulivec":
        fig = plot_state_paulivec(rho, figsize=figsize)
    elif method == "qsphere":
        fig = plot_state_qsphere(rho, figsize=figsize)
    elif method == "bloch":
        plot_bloch_multivector(rho, figsize=figsize)
    elif method == "wigner":
        fig = plot_wigner_function(rho)
    elif method == "hinton":
        fig = plot_state_hinton(rho, figsize=figsize)
    return fig


###############################################################
# Plotting Wigner functions
###############################################################

def plot_wigner_function(state, res=100, figsize=None):
    """Plot the equal angle slice spin Wigner function of an arbitrary
    quantum state.

    Args:
        state (np.matrix[[complex]]):
            - Matrix of 2**n x 2**n complex numbers
            - State Vector of 2**n x 1 complex numbers
        res (int) : number of theta and phi values in meshgrid
            on sphere (creates a res x res grid of points)
        figsize (tuple): Figure size in inches.
    Returns:
         matplotlib.Figure: The matplotlib.Figure of the visualization
    Raises:
        ImportError: Requires matplotlib.

    References:
        [1] T. Tilma, M. J. Everitt, J. H. Samson, W. J. Munro,
        and K. Nemoto, Phys. Rev. Lett. 117, 180401 (2016).
        [2] R. P. Rundle, P. W. Mills, T. Tilma, J. H. Samson, and
        M. J. Everitt, Phys. Rev. A 96, 022117 (2017).
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if figsize is None:
        figsize = (11, 9)

    state = np.asarray(state)
    if state.ndim == 1:
        state = np.outer(state,
                         state)  # turns state vector to a density matrix
    num = int(np.log2(state.shape[0]))  # number of qubits
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

            w[phi, theta] = np.real(np.trace(state.dot(kernel)))  # Wigner function

    # Plot a sphere (x,y,z) with Wigner function facecolor data stored in Wc
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    w_max = np.amax(w)
    # Color data for plotting
    w_c = cm.RdBu((w+w_max)/(2*w_max))  # color data for sphere
    w_c2 = cm.RdBu((w[0:res, int(res/2):res]+w_max)/(2*w_max))  # bottom
    w_c3 = cm.RdBu((w[int(res/4):int(3*res/4), 0:res]+w_max) /
                   (2*w_max))  # side
    w_c4 = cm.RdBu((w[int(res/2):res, 0:res]+w_max)/(2*w_max))  # back

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

    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    m = cm.ScalarMappable(cmap=cm.RdBu)
    m.set_array([-w_max, w_max])
    cbar = plt.colorbar(m, shrink=0.5, aspect=10,
                        ticks=[-1, -0.5, 0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=14)
    plt.close(fig)
    return fig


def plot_wigner_curve(wigner_data, xaxis=None, filename=None):
    """Plots a curve for points in phase space of the spin Wigner function.

    Args:
        wigner_data(np.array): an array of points to plot as a 2d curve
        xaxis (np.array):  the range of the x axis
        filename (str): the output file to save the plot as. If specified it
            will save and exit and not open up the plot in a new window.
    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
    if not xaxis:
        xaxis = np.linspace(0, len(wigner_data)-1, num=len(wigner_data))

    plt.plot(xaxis, wigner_data)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_wigner_plaquette(wigner_data, max_wigner='local', filename=None):
    """Plots plaquette of wigner function data, the plaquette will
    consist of circles each colored to match the value of the Wigner
    function at the given point in phase space.

    Args:
        wigner_data (matrix): array of Wigner function data where the
                            rows are plotted along the x axis and the
                            columns are plotted along the y axis
        max_wigner (str or float):
            - 'local' puts the maximum value to maximum of the points
            - 'unit' sets maximum to 1
            - float for a custom maximum.
        filename (str): the output file to save the plot as. If specified it
            will save and exit and not open up the plot in a new window.
    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
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
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_wigner_data(wigner_data, phis=None, method=None, filename=None):
    """Plots Wigner results in appropriate format.

    Args:
        wigner_data (numpy.array): Output returned from the wigner_data
            function
        phis (numpy.array): Values of phi
        method (str or None): how the data is to be plotted, methods are:
            point: a single point in phase space
            curve: a two dimensional curve
            plaquette: points plotted as circles
        filename (str): the output file to save the plot as. If specified it
            will save and exit and not open up the plot in a new window.
    Raises:
        ImportError: Requires matplotlib.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed.')
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
        plot_wigner_curve(wigner_data, xaxis=phis, filename=filename)
    elif method == 'plaquette':
        plot_wigner_plaquette(wigner_data, filename=filename)
    elif method == 'state':
        plot_wigner_function(wigner_data)
    elif method == 'point':
        plot_wigner_plaquette(wigner_data, filename=filename)
        print('point in phase space is '+str(wigner_data))
    else:
        print("No method given")


def generate_facecolors(x, y, z, dx, dy, dz, color):
    """Generates shaded facecolors for shaded bars.

    This is here to work around a Matplotlib bug
    where alpha does not work in Bar3D.

    Args:
        x (array_like): The x- coordinates of the anchor point of the bars.
        y (array_like): The y- coordinates of the anchor point of the bars.
        z (array_like): The z- coordinates of the anchor point of the bars.
        dx (array_like): Width of bars.
        dy (array_like): Depth of bars.
        dz (array_like): Height of bars.
        color (array_like): sequence of valid color specifications, optional

    Returns:
        list: Shaded colors for bars.
    """
    cuboid = np.array([
        # -z
        (
            (0, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0),
        ),
        # +z
        (
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        # -y
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 1),
            (0, 0, 1),
        ),
        # +y
        (
            (0, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 1, 0),
        ),
        # -x
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
        ),
        # +x
        (
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 0, 1),
        ),
    ])

    # indexed by [bar, face, vertex, coord]
    polys = np.empty(x.shape + cuboid.shape)
    # handle each coordinate separately
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]

    # collapse the first two axes
    polys = polys.reshape((-1,) + polys.shape[2:])

    facecolors = []
    if len(color) == len(x):
        # bar colors specified, need to expand to number of faces
        for c in color:
            facecolors.extend([c] * 6)
    else:
        # a single color specified, or face colors specified explicitly
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= (6 * len(x))

    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)


def _generate_normals(polygons):
    """
    Takes a list of polygons and return an array of their normals.
    Normals point towards the viewer for a face with its vertices in
    counterclockwise order, following the right hand rule.
    Uses three points equally spaced around the polygon.
    This normal of course might not make sense for polygons with more than
    three points not lying in a plane, but it's a plausible and fast
    approximation.

    Args:
        polygons (list): list of (M_i, 3) array_like, or (..., M, 3) array_like
            A sequence of polygons to compute normals for, which can have
            varying numbers of vertices. If the polygons all have the same
            number of vertices and array is passed, then the operation will
            be vectorized.
    Returns:
        normals: (..., 3) array_like
            A normal vector estimated for the polygon.
    """
    if isinstance(polygons, np.ndarray):
        # optimization: polygons all have the same number of points, so can
        # vectorize
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n//3, 2*n//3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # The subtraction doesn't vectorize because polygons is jagged.
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n//3, 2*n//3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]

    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*.
    *color* can also be an array of the same length as *normals*.
    """
    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    shade = np.array([np.dot(n / proj3d.mod(n), lightsource.direction)
                      if proj3d.mod(n) else np.nan
                      for n in normals])
    mask = ~np.isnan(shade)

    if mask.any():
        norm = Normalize(min(shade[mask]), max(shade[mask]))
        shade[~mask] = min(shade[mask])
        color = mcolors.to_rgba_array(color)
        # shape of color should be (M, 4) (where M is number of faces)
        # shape of shade should be (M,)
        # colors should have final shape of (M, 4)
        alpha = color[:, 3]
        colors = (0.5 + norm(shade)[:, np.newaxis] * 0.5) * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors
