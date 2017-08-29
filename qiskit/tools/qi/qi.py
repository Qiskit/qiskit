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
A collection of useful quantum information functions.

Currently this file is very sparse. More functions will be added
over time.
"""

import math
import numpy as np
import scipy.linalg as la
from qiskit.tools.qi.pauli import pauli_group

###############################################################
# circuit manipulation.
###############################################################


# Define methods for making QFT circuits
def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi / float(2**(j - k)), q[j], q[k])
        circ.h(q[j])


###############################################################
# State manipulation.
###############################################################


def partial_trace(state, sys, dims=None, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        state (NxN matrix_like): a matrix
        sys (list(int): a list of subsystems (starting from 0) to trace over
        dims (list(int), optional): a list of the dimensions of the subsystems.
            If this is not set it will assume all subsystems are qubits.
        reverse (bool, optional): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        A matrix with the appropriate subsytems traced over.
    """
    state = np.array(state)  # convert op to density matrix

    if dims is None:  # compute dims if not specified
        n = int(np.log2(len(state)))
        dims = [2 for i in range(n)]
        if len(state) != 2 ** n:
            raise Exception("Input is not a multi-qubit state, \
                specifify input state dims")
    else:
        dims = list(dims)

    if isinstance(sys, int):
        sys = [sys]
    else:  # reverse sort trace sys
        sys = sorted(sys, reverse=True)

    # trace out subsystems
    if state.ndim == 1:
        # optimized partial trace for input state vector
        return __partial_trace_vec(state, sys, dims, reverse)
    else:
        # standard partial trace for input density matrix
        return __partial_trace_mat(state, sys, dims, reverse)


def __partial_trace_vec(vec, sys, dims, reverse=True):
    """
    Partial trace over middle system of tripartite vector.

    Args:
        vec (N) ndarray: a tri-partite complex vector
        dL: dimension of the left subsystem
        dM: dimension of the middle (traced over) subsystem
        dR: dimension of the right subsystem

    Returns:
        A (D) ndarray where D = dL * dR
    """

    # trace sys positions
    if reverse:
        dims = dims[::-1]
        sys = len(dims) - 1 - np.array(sys)

    rho = vec.reshape(dims)
    rho = np.tensordot(rho, rho.conj(), axes=(sys, sys))
    d = int(np.sqrt(np.product(rho.shape)))

    return rho.reshape(d, d)


def __partial_trace_mat(mat, sys, dims, reverse=True):
    """
    Partial trace over middle system of tripartite matrix.

    Args:
        vec (N, N) ndarray: a tri-partite complex matrix
        dL: dimension of the left subsystem
        dM: dimension of the middle (traced over) subsystem
        dR: dimension of the right subsystem

    Returns:
        A (D, D) ndarray where D = dL * dR
    """

    sys = sorted(sys, reverse=True)

    for j in sys:
        dpre = dims[:j]
        dpost = dims[j + 1:]
        if reverse:
            dpre, dpost = (dpost, dpre)
        dL = int(np.prod(dpre))
        dM = int(dims[j])
        dR = int(np.prod(dpost))
        dims = dpre + dpost  # remove middle from dims
        # trace middle subsystem
        mat = mat.reshape([dL, dM, dR, dL, dM, dR])
        mat = mat.trace(axis1=1, axis2=4).reshape(dL * dR, dL * dR)

    return mat


def vectorize(rho, method='col'):
    """Flatten an operator to a vector in a specified basis.

    Args:
        rho (ndarray): a density matrix.
        method (str): the method of vectorization. Allowed values are
            - 'col' (default) flattens to column-major vector.
            - 'row' flattens to row-major vector.
            - 'pauli'flattens in the n-qubit Pauli basis.
            - 'pauli-weights': flattens in the n-qubit Pauli basis ordered by
               weight.

    Returns:
        ndarray: the resulting vector.

    """
    rho = np.array(rho)
    if method == 'col':
        return rho.flatten(order='F')
    elif method == 'row':
        return rho.flatten(order='C')
    elif method in ['pauli', 'pauli_weights']:
        num = int(np.log2(len(rho)))  # number of qubits
        if len(rho) != 2**num:
            raise Exception('Input state must be n-qubit state')
        if method is 'pauli_weights':
            pgroup = pauli_group(num, case=0)
        else:
            pgroup = pauli_group(num, case=1)
        vals = [np.trace(np.dot(p.to_matrix(), rho)) for p in pgroup]
        return np.array(vals)


def devectorize(vec, method='col'):
    """Devectorize a vectorized square matrix.

    Args:
        vec (ndarray): a vectorized density matrix.
        basis (str): the method of devectorizaation. Allowed values are
            - 'col' (default): flattens to column-major vector.
            - 'row': flattens to row-major vector.
            - 'pauli': flattens in the n-qubit Pauli basis.
            - 'pauli-weights': flattens in the n-qubit Pauli basis ordered by
               weight.

    Returns:
        ndarray: the resulting matrix.

    """
    vec = np.array(vec)
    d = int(np.sqrt(vec.size))  # the dimension of the matrix
    if len(vec) != d * d:
        raise Exception('Input is not a vectorized square matrix')

    if method == 'col':
        return vec.reshape(d, d, order='F')
    elif method == 'row':
        return vec.reshape(d, d, order='C')
    elif method in ['pauli', 'pauli_weights']:
        num = int(np.log2(d))  # number of qubits
        if d != 2 ** num:
            raise Exception('Input state must be n-qubit state')
        if method is 'pauli_weights':
            pgroup = pauli_group(num, case=0)
        else:
            pgroup = pauli_group(num, case=1)
        pbasis = np.array([p.to_matrix() for p in pgroup]) / 2 ** num
        return np.tensordot(vec, pbasis, axes=1)


def choi_to_rauli(choi, order=1):
    """
    Convert a Choi-matrix to a Pauli-basis superoperator.

    Note that this function assumes that the Choi-matrix
    is defined in the standard column-stacking converntion
    and is normalized to have trace 1. For a channel E this
    is defined as: choi = (I \otimes E)(bell_state).

    The resulting 'rauli' R acts on input states as
    |rho_out>_p = R.|rho_in>_p
    where |rho> = vectorize(rho, method='pauli') for order=1
    and |rho> = vectorize(rho, method='pauli_weights') for order=0.

    Args:
        Choi (matrix): the input Choi-matrix.
        order (int, optional): ordering of the Pauli group vector.
            order=1 (default) is standard lexicographic ordering.
                Eg: [II, IX, IY, IZ, XI, XX, XY,...]
            order=0 is ordered by weights.
                Eg. [II, IX, IY, IZ, XI, XY, XZ, XX, XY,...]

    Returns:
        A superoperator in the Pauli basis.
    """
    # get number of qubits'
    n = int(np.log2(np.sqrt(len(choi))))
    pgp = pauli_group(n, case=order)
    rauli = []
    for i in pgp:
        for j in pgp:
            pauliop = np.kron(j.to_matrix().T, i.to_matrix())
            rauli += [np.trace(np.dot(choi, pauliop))]
    return np.array(rauli).reshape(4 ** n, 4 ** n)


def chop(op, epsilon=1e-10):
    """
    Truncate small values of a complex array.

    Args:
        op (array_like): array to truncte small values.
        epsilon (float): threshold.

    Returns:
        A new operator with small values set to zero.
    """
    ret = np.array(op)

    if (np.isrealobj(ret)):
        ret[abs(ret) < epsilon] = 0.0
    else:
        ret.real[abs(ret.real) < epsilon] = 0.0
        ret.imag[abs(ret.imag) < epsilon] = 0.0
    return ret


def outer(v1, v2=None):
    """
    Construct the outer product of two vectors.

    The second vector argument is optional, if absent the projector
    of the first vector will be returned.

    Args:
        v1 (ndarray): the first vector.
        v2 (ndarray): the (optional) second vector.

    Returns:
        The matrix |v1><v2|.

    """
    if v2 is None:
        u = np.array(v1).conj()
    else:
        u = np.array(v2).conj()
    return np.outer(v1, u)


###############################################################
# Random Matrices.
###############################################################

def random_unitary(N):
    """
    Return a random unitary ndarray.
    Args:
        N (int): the length of the returned unitary.
    Returns:
        A (N,N): unitary ndarray.
    """
    Q, R = la.qr(__ginibre_matrix(N))
    return Q


def random_density(N, rank=None, method='hs'):
    """
    Generate a random density matrix rho.

    Args:
        N (int): the length of the density matrix.
        rank (int) optional: the rank of the density matrix. The default
        value is full-rank.
        method (string) optional: the method to use.

    Returns:
        rho (N,N) ndarray: a density matrix.

    method:
        'hs': sample rho from the Hilbert-Schmidt metric.
        'bures': sample rho from the Bures metric.
    """
    if method == 'hs':
        return __random_density_hs(N, rank)
    elif method == 'bures':
        return __random_density_bures(N, rank)
    else:
        print('Error: unrecognized method ', method)


def __ginibre_matrix(nrow, ncol=None):
    """
    Return a normaly distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int) optional: number of columns in output matrix.

    Returns:
        A complex rectangular matrix where each real and imaginary
        entry is sampled from the normal distribution.
    """
    if ncol is None:
        ncol = nrow
    G = np.random.normal(size=(nrow, ncol)) + \
        np.random.normal(size=(nrow, ncol)) * 1j
    return G


def __random_density_hs(N, rank=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        N (int): the length of the density matrix.
        rank (int) optional: the rank of the density matrix. The default
        value is full-rank.
    Returns:
        rho (N,N) ndarray: a density matrix.
    """
    G = __ginibre_matrix(N, rank)
    G = G.dot(G.conj().T)
    return G / np.trace(G)


def __random_density_bures(N, rank=None):
    """
    Generate a random density matrix from the Bures metric.

    Args:
        N (int): the length of the density matrix.
        rank (int) optional: the rank of the density matrix. The default
        value is full-rank.
    Returns:
        rho (N,N) ndarray: a density matrix.
    """
    P = np.eye(N) + random_unitary(N)
    G = P.dot(__ginibre_matrix(N, rank))
    G = G.dot(G.conj().T)
    return G / np.trace(G)


###############################################################
# Measures.
###############################################################

def funm_svd(a, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        a : (N, N) array_like
            Matrix at which to evaluate the function.
        func : callable
            Callable object that evaluates a scalar function f.

    Returns:
        funm : (N, N) ndarray
            Value of the matrix function specified by func evaluated at `A`.
    """
    U, s, Vh = la.svd(a, lapack_driver='gesvd')
    S = np.diag(func(s))
    return U.dot(S).dot(Vh)


def state_fidelity(state1, state2):
    """Return the state fidelity between two quantum states.

    Either input may be a state vector, or a density matrix.

    Args:
        state1: a quantum state vector or density matrix.
        state2: a quantum state vector or density matrix.

    Returns:
        The state fidelity F(state1, state2).

    """
    # convert input to numpy arrays
    s1 = np.array(state1)
    s2 = np.array(state2)

    # fidelity of two state vectors
    if s1.ndim == 1 and s2.ndim == 1:
        return np.abs(s2.conj().dot(s1))
    # fidelity of vector and density matrix
    elif s1.ndim == 1:
        # psi = s1, rho = s2
        return np.sqrt(np.abs(s1.conj().dot(s2).dot(s1)))
    elif s2.ndim == 1:
        # psi = s2, rho = s1
        return np.sqrt(np.abs(s2.conj().dot(s1).dot(s2)))
    # fidelity of two density matrices
    else:
        s1sq = funm_svd(s1, np.sqrt)
        s2sq = funm_svd(s2, np.sqrt)
        return np.linalg.norm(s1sq.dot(s2sq), ord='nuc')


def purity(state):
    """Calculate the purity of a quantum state.

    Args:
        state (np.array): a quantum state
    Returns:
        purity.
    """
    rho = np.array(state)
    if rho.ndim == 1:
        rho = outer(rho)
    return np.real(np.trace(rho.dot(rho)))


def concurrence(state):
    """Calculate the concurrence.

    Args:
        state (np.array): a quantum state
    Returns:
        concurrence.
    """
    rho = np.array(state)
    if rho.ndim == 1:
        rho = outer(state)
    if len(state) != 4:
        raise Exception("Concurence is not defined for more than two qubits")

    YY = np.fliplr(np.diag([-1, 1, 1, -1]))
    A = rho.dot(YY).dot(rho.conj()).dot(YY)
    w = la.eigh(A, eigvals_only=True)
    w = np.sqrt(np.maximum(w, 0))
    return max(0.0, w[-1] - np.sum(w[0:-1]))


def shannon_entropy(pvec, base=2):
    """
    Compute the Shannon entropy of a probability vector.

    The shannon entropy of a probability vector pv is defined as
    $H(pv) = - \sum_j pv[j] log_b (pv[j])$ where $0 log_b 0 = 0$.

    Args:
        pvec (array_like): a probability vector.
        base (int): the base of the logarith

    Returns:
        The Shannon entropy H(pvec).
    """
    if base == 2:
        logfn = lambda x: - x * np.log2(x)
    elif base == np.e:
        logfn = lambda x: - x * np.log(x)
    else:
        logfn = lambda x: -x * np.log(x) / np.log(base)

    h = 0.
    for x in pvec:
        if 0 < x < 1:
            h += logfn(x)
    return h


def entropy(state):
    """
    Compute the von-Neumann entropy of a quantum state.

    Args:
        rho (array_like): a density matrix or state vector.

    Returns:
        The von-Neumann entropy S(rho).
    """

    rho = np.array(state)
    if rho.ndim == 1:
        return 0
    else:
        evals = np.maximum(np.linalg.eigvalsh(state), 0.)
        return shannon_entropy(evals, base=np.e)


def mutual_information(state, d0, d1=None):
    """
    Compute the mutual information of a bipartite state.

    Args:
        state: a bipartite state-vector or density-matrix.
        d0 (int): dimension of the first subsystem.
        d1 (int) optional: dimension of the second subsystem.

    Returns:
        The mutual information S(rho_A) + S(rho_B) - S(rho_AB).
    """

    if d1 is None:
        d1 = int(len(state) / d0)
    mi = entropy(partial_trace(state, [0], dims=[d0, d1]))
    mi += entropy(partial_trace(state, [1], dims=[d0, d1]))
    mi -= entropy(state)
    return mi


def entanglement_of_formation(state, d0, d1=None):
    """
    Compute the entanglement of formation of quantum state.

    The input quantum state must be either a bipartite state vector, or a
    2-qubit density matrix.

    Args:
        state (N) array_like or (4,4) array_like: a bipartite quantum state.
        d0 (int) optional: the dimension of the first subsystem.
        d1 (int) optional: the dimension of the second subsystem.

    Returns:
        The entanglement of formation.
    """

    state = np.array(state)

    if d1 is None:
        d1 = int(len(state) / d0)

    if state.ndim == 2 and len(state) == 4 and d0 == 2 and d1 == 2:
        return __eof_qubit(state)
    elif state.ndim == 1:
        # trace out largest dimension
        if d0 < d1:
            tr = [1]
        else:
            tr = [0]
        state = partial_trace(state, tr, dims=[d0, d1])
        return entropy(state)
    else:
        print('Input must be a state-vector or 2-qubit density matrix.')


def __eof_qubit(rho):
    """
    Compute the Entanglement of Formation of a 2-qubit density matrix.

    Args:
        rho (4,4) array_like: input density matrix.

    Returns:
        The entanglement of formation.
    """
    c = concurrence(rho)
    c = 0.5 + 0.5 * np.sqrt(1 - c * c)
    return shannon_entropy([c, 1 - c])


###############################################################
# Other.
###############################################################


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
