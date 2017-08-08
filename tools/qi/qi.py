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
from tools.qi.pauli import pauli_group

###############################################################
# circuit manipulation.
###############################################################


# Define methods for making QFT circuits
def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
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
    # convert op to density matrix
    rho = np.array(state)
    if rho.ndim == 1:
        rho = outer(rho)  # convert state vector to density mat

    # compute dims if not specified
    if dims is None:
        n = int(np.log2(len(rho)))
        dims = [2 for i in range(n)]
        if len(rho) != 2 ** n:
            raise Exception("Input is not a multi-qubit state, \
                specifify input state dims")
    else:
        dims = list(dims)

    # reverse sort trace sys
    if isinstance(sys, int):
        sys = [sys]
    else:
        sys = sorted(sys, reverse=True)

    # trace out subsystems
    for j in sys:
        # get trace dims
        if reverse:
            dpre = dims[j + 1:]
            dpost = dims[:j]
        else:
            dpre = dims[:j]
            dpost = dims[j + 1:]
        dim1 = int(np.prod(dpre))
        dim2 = int(dims[j])
        dim3 = int(np.prod(dpost))
        # dims with sys-j removed
        dims = dpre + dpost
        # do the trace over j
        rho = __trace_middle(rho, dim1, dim2, dim3)
    return rho


def __trace_middle_dims(sys, dims, reverse=True):
    """
    Get system dimensions for __trace_middle.

    Args:
        j (int): system to trace over.
        dims(list[int]): dimensions of all subsystems.
        reverse (bool): if true system-0 is right-most system tensor product.

    Returns:
        Tuple (dim1, dims2, dims3)
    """
    dpre = dims[:sys]
    dpost = dims[sys + 1:]
    if reverse:
        dpre, dpost = (dpost, dpre)
    dim1 = int(np.prod(dpre))
    dim2 = int(dims[sys])
    dim3 = int(np.prod(dpost))
    return (dim1, dim2, dim3)


def __trace_middle(op, dim1=1, dim2=1, dim3=1):
    """
    Partial trace over middle system of tripartite state.

    Args:
        op (NxN matrix_like): a tri-partite matrix
        dim1: dimension of the first subsystem
        dim2: dimension of the second (traced over) subsystem
        dim3: dimension of the third subsystem

    Returns:
        A (D,D) matrix where D = dim1 * dim3
    """

    op = op.reshape(dim1, dim2, dim3, dim1, dim2, dim3)
    d = dim1 * dim3
    return op.trace(axis1=1, axis2=4).reshape(d, d)


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
    if len(vec) != d*d:
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
    op.real[abs(op.real) < epsilon] = 0.0
    op.imag[abs(op.imag) < epsilon] = 0.0
    return op


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
    return max(0.0, w[-1]-np.sum(w[0:-1]))


###############################################################
# Other.
###############################################################


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
