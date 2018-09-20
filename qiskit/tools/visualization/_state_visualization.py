# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string

"""
Visualization functions for quantum states.
"""

from functools import reduce
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from qiskit.tools.qi.pauli import pauli_group, pauli_singles
from qiskit.tools.visualization import VisualizationError
from qiskit.tools.visualization.bloch import Bloch


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
    B = Bloch()
    B.add_vectors(bloch)
    B.show(title=title)


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
    we, stateall = linalg.eigh(rho)
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


def plot_state(quantum_state, method='city'):
    """Plot the quantum state.

    Args:
        quantum_state (ndarray): statevector or density matrix
                                 representation of a quantum state.
        method (str): Plotting method to use.

    Raises:
        VisualizationError: if the input is not a statevector or density
        matrix, or if the state is not an multi-qubit quantum state.
    """

    # Check if input is a statevector, and convert to density matrix
    rho = np.array(quantum_state)
    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))
    # Check the shape of the input is a square matrix
    shape = np.shape(rho)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise VisualizationError("Input is not a valid quantum state.")
    # Check state is an n-qubit state
    num = int(np.log2(len(rho)))
    if 2 ** num != len(rho):
        raise VisualizationError("Input is not a multi-qubit quantum state.")

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
