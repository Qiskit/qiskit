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
A collection of useful quantum information functions. 

Author: Christopher J. Wood <cjwood@us.ibm.com>

Currently this file is very sparse. More functions will be added in
the future.
"""


import numpy as np
from scipy.linalg import sqrtm
from tools.pauli import pauli_group


###############################################################
# State manipulation.
###############################################################


def partial_trace(state, sys, dims=None):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        state (NxN matrix_like): a matrix
        sys (list(int): a list of subsystems (starting from 0) to trace over
        dims (list(int), optional): a list of the dimensions of the subsystems.
            If this is not set it will assume all subsystems are qubits.

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
            print("ERRROR!")
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
        dpre = dims[:j]
        dpost = dims[j+1:]
        dim1 = int(np.prod(dpre))
        dim2 = int(dims[j])
        dim3 = int(np.prod(dpost))
        # dims with sys-j removed
        dims = dpre + dpost
        # do the trace over j
        rho = __trace_middle(rho, dim1, dim2, dim3)
    return rho


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


def vectorize(rho, basis='col'):
    """Flatten an operator to a vector in a specified basis.

    Args:
        rho (ndarray): a density matrix.
        basis (str): the basis to vectorize in. Allowed values are
            - 'col' (default) flattens to column-major vector.
            - 'row' flattens to row-major vector.
            - 'pauli'flattens in the n-qubit Pauli basis.

    Returns:
        ndarray: the resulting vector.

    """
    if basis == 'col':
        return rho.flatten(order='F')
    elif basis == 'row':
        return rho.flatten(order='C')
    elif basis == 'pauli':
        num = int(np.log2(len(rho)))  # number of qubits
        if len(rho) != 2**num:
            print('error input state must be n-qubit state')
        vals = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                        pauli_group(num)))
        return np.array(vals)


def devectorize(vec, basis='col'):
    """Devectorize a vectorized square matrix.

    Args:
        vec (ndarray): a vectorized density matrix.
        basis (str): the basis to vectorize in. Allowed values are
            - 'col' (default): flattens to column-major vector.
            - 'row': flattens to row-major vector.
            - 'pauli': flattens in the n-qubit Pauli basis.

    Returns:
        ndarray: the resulting matrix.

    """
    d = int(np.sqrt(vec.size)) # the dimension of the matrix
    if len(vec) != d*d:
        print('error input is not a vectorized square matrix')

    if basis == 'col':
        return vec.reshape(d, d, order='F')
    elif basis == 'row':
        return vec.reshape(d, d, order='C')
    elif basis == 'pauli':
        num = int(np.log2(d))  # number of qubits
        if d != 2 ** num:
            print('error input state must be n-qubit state')
        pbasis = np.array([p.to_matrix() for p in pauli_group(2)]) / num**2
        return np.tensordot(vec, pbasis, axes=1)

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
        return np.abs(np.dot(s2.conj(), s1))
    # fidelity of vector and density matrix
    elif s1.ndim == 1:
        # psi = s1, rho = s2
        return np.sqrt(np.abs(np.dot(s1.conj(), np.dot(s2, s1))))
    elif s2.ndim == 1:
        # psi = s2, rho = s1
        return np.sqrt(np.abs(np.dot(s2.conj(), np.dot(s1, s2))))
    # fidelity of two density matrices
    else:
        (s1sq, err1) = sqrtm(s1, disp=False)
        (s2sq, err2) = sqrtm(s2, disp=False)
        return np.linalg.norm(np.dot(s1sq, s2sq), ord='nuc')


###############################################################
# Other.
###############################################################


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
