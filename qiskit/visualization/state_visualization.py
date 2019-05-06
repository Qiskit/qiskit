# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,anomalous-backslash-in-string,ungrouped-imports,import-error

"""
Visualization functions for quantum states.
"""

from functools import reduce
import numpy as np
from scipy import linalg
from qiskit.quantum_info.operators.pauli import pauli_group, Pauli
from .matplotlib import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from matplotlib.ticker import MaxNLocator
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.colors as mcolors
    from matplotlib.colors import Normalize, LightSource
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from qiskit.visualization.exceptions import VisualizationError
    from qiskit.visualization.bloch import Bloch
    from qiskit.visualization.utils import _validate_input_state

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
    for idx, cur_zpos in enumerate(zpos):
        if dzr[idx] > 0:
            zorder = 2
        else:
            zorder = 0
        b1 = ax1.bar3d(xpos[idx], ypos[idx], cur_zpos,
                       dx[idx], dy[idx], dzr[idx],
                       alpha=alpha, zorder=zorder)
        b1.set_facecolors(fc1[6*idx:6*idx+6])

    pc1 = Poly3DCollection(verts, alpha=0.15, facecolor='k',
                           linewidths=1, zorder=1)

    if min(dzr) < 0 < max(dzr):
        ax1.add_collection3d(pc1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
    for idx, cur_zpos in enumerate(zpos):
        if dzi[idx] > 0:
            zorder = 2
        else:
            zorder = 0
        b2 = ax2.bar3d(xpos[idx], ypos[idx], cur_zpos,
                       dx[idx], dy[idx], dzi[idx],
                       alpha=alpha, zorder=zorder)
        b2.set_facecolors(fc2[6*idx:6*idx+6])

    pc2 = Poly3DCollection(verts, alpha=0.2, facecolor='k',
                           linewidths=1, zorder=1)

    if min(dzi) < 0 < max(dzi):
        ax2.add_collection3d(pc2)
    ax1.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax1.set_yticks(np.arange(0.5, ly+0.5, 1))
    max_dzr = max(dzr)
    min_dzr = min(dzr)
    if max_dzr != min_dzr:
        ax1.axes.set_zlim3d(np.min(dzr), np.max(dzr)+1e-9)
    else:
        if min_dzr == 0:
            ax1.axes.set_zlim3d(np.min(dzr), np.max(dzr)+1e-9)
        else:
            ax1.axes.set_zlim3d(auto=True)
    ax1.zaxis.set_major_locator(MaxNLocator(5))
    ax1.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45)
    ax1.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5)
    ax1.set_zlabel("Real[rho]", fontsize=14)
    for tick in ax1.zaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2.set_xticks(np.arange(0.5, lx+0.5, 1))
    ax2.set_yticks(np.arange(0.5, ly+0.5, 1))
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)
    if min_dzi != max_dzi:
        eps = 0
        ax2.zaxis.set_major_locator(MaxNLocator(5))
        ax2.axes.set_zlim3d(np.min(dzi), np.max(dzi)+eps)
    else:
        if min_dzi == 0:
            ax2.set_zticks([0])
            eps = 1e-9
            ax2.axes.set_zlim3d(np.min(dzi), np.max(dzi)+eps)
        else:
            ax2.axes.set_zlim3d(auto=True)
    ax2.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45)
    ax2.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5)
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
