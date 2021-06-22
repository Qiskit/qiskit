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

# pylint: disable=invalid-name
# pylint: disable=inconsistent-return-statements
# pylint: disable=missing-param-doc,missing-type-doc,unused-argument

"""
Visualization functions for quantum states.
"""

from functools import reduce
import colorsys
import numpy as np
from scipy import linalg
from qiskit import user_config
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.visualization.array import array_to_latex
from qiskit.utils.deprecation import deprecate_arguments
from qiskit.visualization.matplotlib import HAS_MATPLOTLIB
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.utils import _bloch_multivector_data, _paulivec_data
from qiskit.circuit.tools.pi_check import pi_check


@deprecate_arguments({"rho": "state"})
def plot_state_hinton(state, title="", figsize=None, ax_real=None, ax_imag=None, *, rho=None):
    """Plot a hinton diagram for the density matrix of a quantum state.

    Args:
        state (Statevector or DensityMatrix or ndarray): An N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if
            neither ax_real or ax_imag is set.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import DensityMatrix
            from qiskit.visualization import plot_state_hinton
            %matplotlib inline

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)

            state = DensityMatrix.from_instruction(qc)
            plot_state_hinton(state, title="New Hinton Plot")
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_hinton",
            pip_install="pip install matplotlib",
        )
    from matplotlib import pyplot as plt
    from matplotlib import get_backend

    # Figure data
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    max_weight = 2 ** np.ceil(np.log(np.abs(rho.data).max()) / np.log(2))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    if figsize is None:
        figsize = (8, 5)
    if not ax_real and not ax_imag:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax_real:
            fig = ax_real.get_figure()
        else:
            fig = ax_imag.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    ly, lx = datareal.shape
    # Real
    if ax1:
        ax1.patch.set_facecolor("gray")
        ax1.set_aspect("equal", "box")
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(datareal):
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + x - size / 2, 0.5 + y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax1.add_patch(rect)

        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([ly, 0])
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax1.invert_yaxis()
        ax1.set_title("Re[$\\rho$]", fontsize=14)
    # Imaginary
    if ax2:
        ax2.patch.set_facecolor("gray")
        ax2.set_aspect("equal", "box")
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(dataimag):
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + x - size / 2, 0.5 + y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax2.add_patch(rect)

        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([ly, 0])
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax2.invert_yaxis()
        ax2.set_title("Im[$\\rho$]", fontsize=14)
    if title:
        fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
            plt.close(fig)
        return fig


def plot_bloch_vector(bloch, title="", ax=None, figsize=None, coord_type="cartesian"):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        bloch (list[double]): array of three elements where [<x>, <y>, <z>] (Cartesian)
            or [<r>, <theta>, <phi>] (spherical in radians)
            <theta> is inclination angle from +z direction
            <phi> is azimuth from +x direction
        title (str): a string that represents the plot title
        ax (matplotlib.axes.Axes): An Axes to use for rendering the bloch
            sphere
        figsize (tuple): Figure size in inches. Has no effect is passing ``ax``.
        coord_type (str): a string that specifies coordinate type for bloch
            (Cartesian or spherical), default is Cartesian

    Returns:
        Figure: A matplotlib figure instance if ``ax = None``.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.

    Example:
        .. jupyter-execute::

           from qiskit.visualization import plot_bloch_vector
           %matplotlib inline

           plot_bloch_vector([0,1,0], title="New Bloch Sphere")
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_bloch_vector",
            pip_install="pip install matplotlib",
        )
    from qiskit.visualization.bloch import Bloch
    from matplotlib import get_backend
    from matplotlib import pyplot as plt

    if figsize is None:
        figsize = (5, 5)
    B = Bloch(axes=ax)
    if coord_type == "spherical":
        r, theta, phi = bloch[0], bloch[1], bloch[2]
        bloch[0] = r * np.sin(theta) * np.cos(phi)
        bloch[1] = r * np.sin(theta) * np.sin(phi)
        bloch[2] = r * np.cos(theta)
    B.add_vectors(bloch)
    B.render(title=title)
    if ax is None:
        fig = B.fig
        fig.set_size_inches(figsize[0], figsize[1])
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
            plt.close(fig)
        return fig
    return None


@deprecate_arguments({"rho": "state"})
def plot_bloch_multivector(state, title="", figsize=None, *, rho=None, reverse_bits=False):
    """Plot the Bloch sphere.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Has no effect, here for compatibility only.
        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].

    Returns:
        matplotlib.Figure:
            A matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.visualization import plot_bloch_multivector
            %matplotlib inline

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)

            state = Statevector.from_instruction(qc)
            plot_bloch_multivector(state, title="New Bloch Multivector", reverse_bits=False)
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_bloch_multivector",
            pip_install="pip install matplotlib",
        )
    from matplotlib import get_backend
    from matplotlib import pyplot as plt

    # Data
    bloch_data = (
        _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    )
    num = len(bloch_data)
    width, height = plt.figaspect(1 / num)
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(1, num, i + 1, projection="3d")
        plot_bloch_vector(bloch_data[i], "qubit " + str(pos), ax=ax, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
        plt.close(fig)
    return fig


@deprecate_arguments({"rho": "state"})
def plot_state_city(
    state, title="", figsize=None, color=None, alpha=1, ax_real=None, ax_imag=None, *, rho=None
):
    """Plot the cityscape of quantum state.

    Plot two 3d bar graphs (two dimensional) of the real and imaginary
    part of the density matrix rho.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list): A list of len=2 giving colors for real and
            imaginary components of matrix elements.
        alpha (float): Transparency value for bars
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_real only the imaginary component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if the
            ``ax_real`` and ``ax_imag`` kwargs are not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        ValueError: When 'color' is not a list of len=2.
        VisualizationError: if input is not a valid N-qubit state.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import DensityMatrix
           from qiskit.visualization import plot_state_city
           %matplotlib inline

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = DensityMatrix.from_instruction(qc)
           plot_state_city(state, color=['midnightblue', 'midnightblue'],
                title="New State City")
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_city",
            pip_install="pip install matplotlib",
        )
    from matplotlib import get_backend
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")

    # get the real and imag parts of rho
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]

    lx = len(datareal[0])  # Work out matrix dimensions
    ly = len(datareal[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

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
    if ax_real is None and ax_imag is None:
        # set default figure size
        if figsize is None:
            figsize = (15, 5)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag

    max_dzr = max(dzr)
    min_dzr = min(dzr)
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)

    if ax1 is not None:
        fc1 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzr, color[0])
        for idx, cur_zpos in enumerate(zpos):
            if dzr[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b1 = ax1.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzr[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b1.set_facecolors(fc1[6 * idx : 6 * idx + 6])

        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc1 = Poly3DCollection(verts, alpha=0.15, facecolor="k", linewidths=1, zorder=1)

        if min(dzr) < 0 < max(dzr):
            ax1.add_collection3d(pc1)
        ax1.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax1.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dzr != min_dzr:
            ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, max_dzi))
        else:
            if min_dzr == 0:
                ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, np.max(dzi)))
            else:
                ax1.axes.set_zlim3d(auto=True)
        ax1.get_autoscalez_on()
        ax1.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax1.w_yaxis.set_ticklabels(
            column_names, fontsize=14, rotation=-22.5, ha="left", va="center"
        )
        ax1.set_zlabel("Re[$\\rho$]", fontsize=14)
        for tick in ax1.zaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    if ax2 is not None:
        fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
        for idx, cur_zpos in enumerate(zpos):
            if dzi[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b2 = ax2.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzi[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b2.set_facecolors(fc2[6 * idx : 6 * idx + 6])

        xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc2 = Poly3DCollection(verts, alpha=0.2, facecolor="k", linewidths=1, zorder=1)

        if min(dzi) < 0 < max(dzi):
            ax2.add_collection3d(pc2)
        ax2.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax2.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if min_dzi != max_dzi:
            eps = 0
            ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
        else:
            if min_dzi == 0:
                ax2.set_zticks([0])
                eps = 1e-9
                ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
            else:
                ax2.axes.set_zlim3d(auto=True)

        ax2.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax2.w_yaxis.set_ticklabels(
            column_names, fontsize=14, rotation=-22.5, ha="left", va="center"
        )
        ax2.set_zlabel("Im[$\\rho$]", fontsize=14)
        for tick in ax2.zaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        ax2.get_autoscalez_on()

    fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
            plt.close(fig)
        return fig


@deprecate_arguments({"rho": "state"})
def plot_state_paulivec(state, title="", figsize=None, color=None, ax=None, *, rho=None):
    """Plot the paulivec representation of a quantum state.

    Plot a bargraph of the mixed state rho over the pauli matrices

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list or str): Color of the expectation value bars.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if the
            ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_paulivec
           %matplotlib inline

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector.from_instruction(qc)
           plot_state_paulivec(state, color='midnightblue',
                title="New PauliVec plot")
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_paulivec",
            pip_install="pip install matplotlib",
        )
    from matplotlib import get_backend
    from matplotlib import pyplot as plt

    labels, values = _paulivec_data(state)
    numelem = len(values)

    if figsize is None:
        figsize = (7, 5)
    if color is None:
        color = "#648fff"

    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.5  # the width of the bars
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    ax.grid(zorder=0, linewidth=1, linestyle="--")
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color="k")
    # add some text for labels, title, and axes ticks
    ax.set_ylabel("Expectation value", fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel("Pauli", fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor("#eeeeee")
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    if return_fig:
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
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
    return reduce(lambda x, y: x * y[0] / y[1], zip(range(n - k + 1, n + 1), range(1, k + 1)), 1)


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
    dualm = sum(n_choose_k(comb[k - 1 - i], i + 1) for i in range(k))
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_rgb(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to the HSL color wheel
    """
    angles = (np.angle(complex_number) + (np.pi * 5 / 4)) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb


@deprecate_arguments({"rho": "state"})
def plot_state_qsphere(
    state,
    figsize=None,
    ax=None,
    show_state_labels=True,
    show_state_phases=False,
    use_degrees=False,
    *,
    rho=None,
):
    """Plot the qsphere representation of a quantum state.
    Here, the size of the points is proportional to the probability
    of the corresponding term in the state and the color represents
    the phase.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        figsize (tuple): Figure size in inches.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        show_state_labels (bool): An optional boolean indicating whether to
            show labels for each basis state.
        show_state_phases (bool): An optional boolean indicating whether to
            show the phase for each basis state.
        use_degrees (bool): An optional boolean indicating whether to use
            radians or degrees for the phase values in the plot.

    Returns:
        Figure: A matplotlib figure instance if the ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

        QiskitError: Input statevector does not have valid dimensions.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_qsphere
           %matplotlib inline

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector.from_instruction(qc)
           plot_state_qsphere(state)
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_qsphere",
            pip_install="pip install matplotlib",
        )

    import matplotlib.gridspec as gridspec
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib import get_backend
    from qiskit.visualization.bloch import Arrow3D

    try:
        import seaborn as sns
    except ImportError as ex:
        raise MissingOptionalLibraryError(
            libname="seaborn",
            name="plot_state_qsphere",
            pip_install="pip install seaborn",
        ) from ex
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    # get the eigenvectors and eigenvalues
    eigvals, eigvecs = linalg.eigh(rho.data)

    if figsize is None:
        figsize = (7, 7)

    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()

    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax = fig.add_subplot(gs[0:3, 0:3], projection="3d")
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)

    # Force aspect ratio
    # MPL 3.2 or previous do not have set_box_aspect
    if hasattr(ax.axes, "set_box_aspect"):
        ax.axes.set_box_aspect((1, 1, 1))

    # start the plotting
    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color=plt.rcParams["grid.color"], alpha=0.2, linewidth=0
    )

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

    # traversing the eigvals/vecs backward as sorted low->high
    for idx in range(eigvals.shape[0] - 1, -1, -1):
        if eigvals[idx] > 0.001:
            # get the max eigenvalue
            state = eigvecs[:, idx]
            loc = np.absolute(state).argmax()
            # remove the global phase from max element
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state

            d = num
            for i in range(2 ** num):
                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = (float(weight) / d) * (np.pi * 2) + (
                    weight_order * 2 * (np.pi / number_of_divisions)
                )

                if (weight > d / 2) or (
                    (weight == d / 2) and (weight_order >= number_of_divisions / 2)
                ):
                    angle = np.pi - angle - (2 * np.pi / number_of_divisions)

                xvalue = np.sqrt(1 - zvalue ** 2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue ** 2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                prob = min(prob, 1)  # See https://github.com/Qiskit/qiskit-terra/issues/4666
                colorstate = phase_to_rgb(state[i])

                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue

                if not np.isclose(prob, 0) and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue ** 2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = "$\\vert" + element + "\\rangle$"
                    if show_state_phases:
                        element_angle = (np.angle(state[i]) + (np.pi * 4)) % (np.pi * 2)
                        if use_degrees:
                            element_text += "\n$%.1f^\\circ$" % (element_angle * 180 / np.pi)
                        else:
                            element_angle = pi_check(element_angle, ndigits=3).replace("pi", "\\pi")
                            element_text += "\n$%s$" % (element_angle)
                    ax.text(
                        xvalue_text,
                        yvalue_text,
                        zvalue_text,
                        element_text,
                        ha="center",
                        va="center",
                        size=12,
                    )

                ax.plot(
                    [xvalue],
                    [yvalue],
                    [zvalue],
                    markerfacecolor=colorstate,
                    markeredgecolor=colorstate,
                    marker="o",
                    markersize=np.sqrt(prob) * 30,
                    alpha=alfa,
                )

                a = Arrow3D(
                    [0, xvalue],
                    [0, yvalue],
                    [0, zvalue],
                    mutation_scale=20,
                    alpha=prob,
                    arrowstyle="-",
                    color=colorstate,
                    lw=2,
                )
                ax.add_artist(a)

            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z ** 2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5), lw=1, ls=":", alpha=0.5)

            # add center point
            ax.plot(
                [0],
                [0],
                [0],
                markerfacecolor=(0.5, 0.5, 0.5),
                markeredgecolor=(0.5, 0.5, 0.5),
                marker="o",
                markersize=3,
                alpha=1,
            )
        else:
            break

    n = 64
    theta = np.ones(n)
    colors = sns.hls_palette(n)

    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=colors[5 * n // 8 :] + colors[: 5 * n // 8], radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color="white", zorder=1))
    offset = 0.95  # since radius of sphere is one.

    if use_degrees:
        labels = ["Phase\n(Deg)", "0", "90", "180   ", "270"]
    else:
        labels = ["Phase", "$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"]

    ax2.text(0, 0, labels[0], horizontalalignment="center", verticalalignment="center", fontsize=14)
    ax2.text(
        offset, 0, labels[1], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, offset, labels[2], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        -offset, 0, labels[3], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, -offset, labels[4], horizontalalignment="center", verticalalignment="center", fontsize=14
    )

    if return_fig:
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
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
    Raises:
        MissingOptionalLibraryError: If matplotlib is not installed
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_city",
            pip_install="pip install matplotlib",
        )
    import matplotlib.colors as mcolors

    cuboid = np.array(
        [
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
        ]
    )

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
            facecolors *= 6 * len(x)

    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)


def _generate_normals(polygons):
    """Takes a list of polygons and return an array of their normals.

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
        i1, i2, i3 = 0, n // 3, 2 * n // 3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # The subtraction doesn't vectorize because polygons is jagged.
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n // 3, 2 * n // 3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]

    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*.
    *color* can also be an array of the same length as *normals*.
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_state_city",
            pip_install="pip install matplotlib",
        )

    from matplotlib.colors import Normalize, LightSource
    import matplotlib.colors as mcolors

    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    def mod(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    shade = np.array(
        [np.dot(n / mod(n), lightsource.direction) if mod(n) else np.nan for n in normals]
    )
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


def state_to_latex(state, dims=None, **args):
    """Return a Latex representation of a state. Wrapper function
    for `qiskit.visualization.array_to_latex` to add dims if necessary.
    Intended for use within `state_drawer`.

    Args:
        state (`Statevector` or `DensityMatrix`): State to be drawn
        dims (bool): Whether to display the state's `dims`
        *args: Arguments to be passed directly to `array_to_latex`

    Returns:
        `str`: Latex representation of the state
    """
    if dims is None:  # show dims if state is not only qubits
        if set(state.dims()) == {2}:
            dims = False
        else:
            dims = True

    prefix = ""
    suffix = ""
    if dims:
        prefix = "\\begin{align}\n"
        dims_str = state._op_shape.dims_l()
        suffix = f"\\\\\n\\text{{dims={dims_str}}}\n\\end{{align}}"
    latex_str = array_to_latex(state._data, source=True, **args)
    return prefix + latex_str + suffix


class TextMatrix:
    """Text representation of an array, with `__str__` method so it
    displays nicely in Jupyter notebooks"""

    def __init__(self, state, max_size=8, dims=None, prefix="", suffix=""):
        self.state = state
        self.max_size = max_size
        if dims is None:  # show dims if state is not only qubits
            if set(state.dims()) == {2}:
                dims = False
            else:
                dims = True
        self.dims = dims
        self.prefix = prefix
        self.suffix = suffix
        if isinstance(max_size, int):
            self.max_size = max_size
        elif isinstance(state, DensityMatrix):
            # density matrices are square, so threshold for
            # summarization is shortest side squared
            self.max_size = min(max_size) ** 2
        else:
            self.max_size = max_size[0]

    def __str__(self):
        threshold = self.max_size
        data = np.array2string(
            self.state._data, prefix=self.prefix, threshold=threshold, separator=","
        )
        dimstr = ""
        if self.dims:
            data += ",\n"
            dimstr += " " * len(self.prefix)
            dimstr += f"dims={self.state._op_shape.dims_l()}"
        return self.prefix + data + dimstr + self.suffix

    def __repr__(self):
        return self.__str__()


def state_drawer(state, output=None, **drawer_args):
    """Returns a visualization of the state.

    **repr**: ASCII TextMatrix of the state's ``_repr_``.

    **text**: ASCII TextMatrix that can be printed in the console.

    **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

    **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

    **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.

    **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.

    **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.

    **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.

    **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.

    Args:
        output (str): Select the output method to use for drawing the
            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,
            ``qsphere``, ``hinton``, ``bloch``, ``city`` or ``paulivec``.
            Default is `'text`'.
        drawer_args: Arguments to be passed to the relevant drawer. For
            'latex' and 'latex_source' see ``array_to_latex``

    Returns:
        :class:`matplotlib.figure` or :class:`str` or
        :class:`TextMatrix` or :class:`IPython.display.Latex`:
        Drawing of the state.

    Raises:
        MissingOptionalLibraryError: when `output` is `latex` and IPython is not installed.
        ValueError: when `output` is not a valid selection.
    """
    config = user_config.get_config()
    # Get default 'output' from config file else use 'repr'
    default_output = "repr"
    if output is None:
        if config:
            default_output = config.get("state_drawer", "repr")
        output = default_output
    output = output.lower()

    # Choose drawing backend:
    drawers = {
        "text": TextMatrix,
        "latex_source": state_to_latex,
        "qsphere": plot_state_qsphere,
        "hinton": plot_state_hinton,
        "bloch": plot_bloch_multivector,
        "city": plot_state_city,
        "paulivec": plot_state_paulivec,
    }
    if output == "latex":
        try:
            from IPython.display import Latex
        except ImportError as err:
            raise MissingOptionalLibraryError(
                libname="IPython",
                name="state_drawer",
                pip_install="\"pip install ipython\", or set output='latex_source' "
                "instead for an ASCII string.",
            ) from err
        else:
            draw_func = drawers["latex_source"]
            return Latex(f"$${draw_func(state, **drawer_args)}$$")

    if output == "repr":
        return state.__repr__()

    try:
        draw_func = drawers[output]
        return draw_func(state, **drawer_args)
    except KeyError as err:
        raise ValueError(
            """'{}' is not a valid option for drawing {} objects. Please choose from:
            'text', 'latex', 'latex_source', 'qsphere', 'hinton',
            'bloch', 'city' or 'paulivec'.""".format(
                output, type(state).__name__
            )
        ) from err
