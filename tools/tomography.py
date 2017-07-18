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
from scipy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from tools.pauli import pauli_group, pauli_singles

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


def plot_bloch_vector(bloch, title=""):
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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    ax.plot_surface(x, y, z, color=(.5,.5,.5), alpha=0.1)

    # Plot arrows (axes, Bloch vector, its projections)
    xa = Arrow3D([0, arlen], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color=(.5,.5,.5))
    ya = Arrow3D([0, 0], [0, arlen], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color=(.5,.5,.5))
    za = Arrow3D([0, 0], [0, 0], [0, arlen], mutation_scale=20, lw=1, arrowstyle="-|>", color=(.5,.5,.5))
    a = Arrow3D([0, bloch[0]], [0, bloch[1]], [0, bloch[2]], mutation_scale=20, lw=2, arrowstyle="simple", color="k")
    bax = Arrow3D([0, bloch[0]], [0, 0], [0, 0], mutation_scale=20, lw=2, arrowstyle="-", color="r")
    bay = Arrow3D([0, 0], [0, bloch[1]], [0, 0], mutation_scale=20, lw=2, arrowstyle="-", color="g")
    baz = Arrow3D([0, 0], [0, 0], [0, bloch[2]], mutation_scale=20, lw=2, arrowstyle="-", color="b")
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

    plt.title(title)
    plt.show()


def plot_state(rho, method='city'):
    """Plot the cityscape of quantum state."""
    num = int(np.log2(len(rho)))
    # Need updating to check its a matrix
    if method == 'city':
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

        plt.show()
    elif method == "paulivec":
        labels = list(map(lambda x: x.to_label(), pauli_group(num)))
        values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))), pauli_group(num)))
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
        ax.set_xticklabels(labels, fontsize=12, rotation=70)
        ax.set_xlabel('Pauli', fontsize=12)
        ax.set_ylim([-1, 1])
        plt.show()
    elif method == "qsphere":
        """Plot the qsphere."""
        # get the eigenvectors and egivenvalues
        we, stateall = la.eigh(rho)
        for i in range(2**num):
            # start with the max
            probmix = we.max()
            prob_location = we.argmax()
            if probmix > 0.001:
                print("The " + str(i) + "th eigenvalue = " + str(probmix))
                # get the max eigenvalue
                state = stateall[:, prob_location]
                element_location = np.absolute(state).argmax()
                # get the element location closes to lowest bin representation.
                for j in range(2**num):
                    if np.absolute(np.absolute(state[j])-np.absolute(state[element_location]))<0.001:
                        element_location = j
                        break
                # remove the global phase
                angles = (np.angle(state[element_location]) + 2 * np.pi) % (2 * np.pi)
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
                ax.plot_surface(x, y, z, rstride=1, cstride=1, color='k', alpha=0.05,
                                linewidth=0)
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
                    angle = (weight_order) * 2 * np.pi / number_of_divisions
                    xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                    yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)
                    ax.plot([xvalue], [yvalue], [zvalue], markerfacecolor=(.5,.5,.5), markeredgecolor=(.5,.5,.5), marker='o', markersize=10, alpha=1)
                    # get prob and angle - prob will be shade and angle color
                    prob = np.real(np.dot(state[i], state[i].conj()))
                    angles = np.angle(state[i])
                    angleround = int(((angles + 2 * np.pi) % (2 * np.pi))/2/np.pi*12)
                    #print(angleround)
                    if angleround == 4:
                        colorstate = (1, 0, 0)
                    elif angleround == 5:
                        colorstate = (1, 0.5, 0)
                    elif angleround == 6:
                        colorstate = (1, 1, 0)
                    elif angleround == 7:
                        colorstate = (0.5, 1, 0)
                    elif angleround == 8:
                        colorstate = (0, 1, 0)
                    elif angleround == 9:
                        colorstate = (0, 1, 0.5)
                    elif angleround == 10:
                        colorstate = (0, 1, 1)
                    elif angleround == 11:
                        colorstate = (0, 0.5, 1)
                    elif angleround == 0:
                        colorstate = (0, 0, 1)
                    elif angleround == 1:
                        colorstate = (0.5, 0, 1)
                    elif angleround == 2:
                        colorstate = (1, 0, 1)
                    elif angleround == 3:
                        colorstate = (1, 0, 0.5)
                    # print("outcome = " + element + " weight " + str(weight) + " angle " + str(angle) + " amp " + str(linewidth))
                    a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue], mutation_scale=20,
                                 alpha=prob, arrowstyle="-", color=colorstate, lw = 10)
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
                ax.plot([0], [0], [0], markerfacecolor=(.5,.5,.5), markeredgecolor=(.5,.5,.5), marker='o', markersize=10, alpha=1)
                plt.show()
                we[prob_location] = 0
            else:
                break
    elif method == "bloch":
        for i in range(num):
            bloch_state = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))), pauli_singles(i, num)))
            plot_bloch_vector(bloch_state, "qubit " + str(i))
    else:
        print("No method given")


###############################################################
# Constructing tomographic measurement circuits
###############################################################
def build_keys_helper(keys, qubit):
    """
    Returns array of measurement strings ['Xj', 'Yj', 'Zj'] for qubit=j.
    """
    tmp = []
    for k in keys:
        for b in ["X","Y","Z"]:
            tmp.append(k + b + str(qubit))
    return tmp

def build_tomo_keys(circuit, qubit_list):
    """
    For input circuit string returns an array of all measurement circits orded in
    lexicographic order from last qubit to first qubit.
    Example:
    qubit_list = [0]: [circuitX0, circuitY0, circuitZ0].
    qubit_list = [0,1]: [circuitX1X0, circuitX1Y0, circuitX1Z0, circuitY1X0,..., circuitZ1Z0].
    """
    keys = [circuit]
    for j in sorted(qubit_list,reverse=True):
        keys = build_keys_helper(keys, j)
    return keys

# We need to build circuits in QASM lexical order, not standard!

def build_tomo_circuit_helper(Q_program, circuits, qreg: str, creg: str, qubit: int):
    """
    Adds measurements for the qubit=j to the input circuits, so if circuits = [c0, c1,...]
    circuits-> [c0Xj, c0Yj, c0Zj, c1Xj,...]
    """
    for c in circuits:
        circ = Q_program.get_circuit(c)
        for b in ["X","Y","Z"]:
            meas = b+str(qubit)
            tmp = Q_program.create_circuit(meas, [qreg],[creg])
            qr = Q_program.get_quantum_registers(qreg)
            cr = Q_program.get_classical_registers(creg)
            if b == "X":
                tmp.u2(0., np.pi, qr[qubit])
            if b == "Y":
                tmp.u2(0., np.pi / 2., qr[qubit])
            tmp.measure(qr[qubit], cr[qubit])
            Q_program.add_circuit(c+meas, circ + tmp)

def build_tomo_circuits(Q_program, circuit, qreg: str, creg: str, qubit_list):
    """
    Generates circuits in the input QuantumProgram for implementing complete quantum
    state tomography of the state of the qubits in qubit_list prepared by circuit.
    """
    circ = [circuit]
    for j in sorted(qubit_list, reverse=True):
        build_tomo_circuit_helper(Q_program, circ, qreg, creg, j)
        circ = build_keys_helper(circ, j)


###############################################################
# Tomographic Reconstruction functions.
###############################################################

def vectorize(op):
    """
    Flatten an operator to a column-major vector.
    """
    return op.flatten(order='F')

def devectorize(v):
    """
    Devectorize a column-major vectorized square matrix.
    """
    d = int(np.sqrt(v.size));
    return v.reshape(d,d, order='F')

def meas_basis_matrix(meas_basis):
    """
    Returns a matrix of vectorized measurement operators S = sum_j |j><M_j|.
    """
    n = len(meas_basis)
    d = meas_basis[0].size
    S = np.array([vectorize(m).conj() for m in meas_basis])
    return S.reshape(n,d)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def outer(v1, v2=None):
    """
    Returns the matrix |v1><v2| resulting from the outer product of two vectors.
    """
    if v2 is None:
        u = v1.conj()
    else:
        u = v2.conj()
    return np.outer(v1, u)

def wizard(rho, normalize_flag = True, epsilon = 0.):
    """
    Maps an operator to the nearest positive semidefinite operator
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.
    See arXiv:1106.5458 [quant-ph]
    """
    #print("Using wizard method to constrain positivity")
    if normalize_flag:
        rho = rho / np.trace(rho)
    dim = len(rho)
    rho_wizard = np.zeros([dim, dim])
    v, w = np.linalg.eigh(rho) # v eigenvecrors v[0] < v[1] <...
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # redistribute loop
            x = 0.
            for k in range(j+1,dim):
                x += tmp / (dim-(j+1))
                v[k] = v[k] + tmp / (dim -(j+1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:,j])
    return rho_wizard

def fit_state(freqs, meas_basis, weights=None, normalize_flag = True, wizard_flag = False):
    """
    Returns a matrix reconstructed by unconstrained least-squares fitting.
    """
    if weights is None:
        W = np.eye(len(freqs)) # use uniform weights
    else:
        W = np.array(np.diag(weights))
    S = np.dot(W, meas_basis_matrix(meas_basis)) # actually W.S
    v = np.dot(W, freqs) # W|f>
    v = np.array(np.dot(S.T.conj(), v)) # S^*.W^*W.|f>
    inv = np.linalg.pinv(np.dot(S.T.conj(), S)) # pseudo inverse of  S^*.W^*.W.S
    v = np.dot(inv, v) # |rho>
    rho = devectorize(v)
    if normalize_flag:
        rho = rho / np.trace(rho)
    if wizard_flag:
        #rho_wizard = wizard(rho,normalize_flag)
        rho = wizard(rho, normalize_flag=normalize_flag)
    return rho



###############################################################
# Parsing measurement data for reconstruction
###############################################################

def nqubit_basis(n):
    """
    Returns the measurement basis for n-qubits in the correct order for the
    meas_outcome_strings function.
    """
    b1 =  np.array([
            np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]), # Xp, Xm
            np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]), # Yp, Ym
            np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), # Zp, Zm
            ])
    if n == 1:
        return b1
    else:
        bnm1 = nqubit_basis(n-1)
        m = 2**(n-1)
        d = bnm1.size // m**3
        return np.kron( bnm1.reshape(d, m, m, m), b1.reshape(3, 2, 2, 2)).reshape(3*d * 2*m, 2*m, 2*m)

def meas_outcome_strings(nq):
    """
    Returns a list of the measurement outcome strings for nq qubits.
    """
    return [bin(j)[2:].zfill(nq) for j in range(2**nq)]

def tomo_outcome_strings(meas_qubits,nq=None):
    """
    Returns a list of the measurement outcome strings for meas_qubits qubits in an
    nq qubit system.
    """
    if nq is None:
        nq = len(meas_qubits)
    qs = sorted(meas_qubits, reverse=True)
    outs = meas_outcome_strings(len(qs))
    res = [];
    for s in outs:
        label = ""
        for j in range(nq):
            if j in qs:
                label = s[qs.index(j)] + label
            else:
                label = str(0) + label
        res.append(label)
    return res

def none_to_zero(val):
    """
    Returns 0 if the input argument is None, else it returns the input.
    """
    if val is None:
        return 0
    else:
        return val

###############################################################
# Putting it all together
###############################################################
def state_tomography(Q_program, tomo_circuits, shots, nq,
                                    meas_qubits, method='leastsq_wizard'):
    """
    Returns the reconstructed density matrix.
    """
    m = len(meas_qubits)
    counts = np.array([none_to_zero(Q_program.get_counts(c).get(s))
                        for c in tomo_circuits
                        for s in tomo_outcome_strings(meas_qubits, nq)])
    if method == 'leastsq_wizard':
        return fit_state(counts / shots, nqubit_basis(m), normalize_flag=True, wizard_flag=True)
    elif method == 'least_sq':
        return fit_state(counts / shots, nqubit_basis(m), normalize_flag=True, wizard_flag=False)
    else:
        print("error: unknown reconstruction method")

def fidelity(rho, psi):
    """
    Returns the state fidelity F between a density matrix rho
    and a target pure state psi.
    """
    return np.sqrt(np.abs(np.dot(psi, np.dot(rho, psi))))
