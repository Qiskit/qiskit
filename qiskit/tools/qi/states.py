# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string

"""
A collection of useful quantum information functions for states.


"""

import numpy as np
import scipy.linalg as la

from qiskit import QISKitError


###############################################################
# States.
###############################################################


def basis(basis_state, num):
    """
    Return a basis state ndarray.

    Args:
        basis_state (string): a string representing the state.
        num (int): the number of qubits
    Returns:
        ndarray:  state(2**num) a quantum state with basis basis state.
    Raise:
        QISKitError if the dimensions is wrong
    """
    n = int(basis_state, 2)
    if num >= len(basis_state):
        state = np.zeros(1 << num, dtype=complex)
        state[n] = 1
        return state
    else:
        raise QISKitError('size of bitstring is greater than num.')


def random_state(num):
    """
    Return a random quantum state from the uniform (Haar) measure on 
    state space.

    Args:
        num (int): the number of qubits
    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    x = np.random.random(1 << num)+0.00000001
    x = -np.log(x)
    sumx = sum(x)
    phases = np.random.random(1 << num)*2.0*np.pi
    return np.sqrt(x/sumx)*np.exp(1j*phases)


def psi_2_rho(state):
    """
    maps a pure state to a state matrix

    Args:
        state (ndarray): the number of qubits
    Returns:
        ndarray:  state_mat(2**num, 2**num).
    """
    return np.outer(state.conjugate(), state)


###############################################################
# Measures.
###############################################################


def state_fidelity(state1, state2):
    """Return the state fidelity between two quantum states.

    Either input may be a state vector, or a density matrix. The state
    fidelity (F) for two density matrices is defined as:
        F(rho1, rho2) = Tr[sqrt(sqrt(rho1).rho2.sqrt(rho1))] ^ 2
    For a pure state and mixed state the fidelity is given by
        F(|psi1>, rho2) = <psi1|rho2|psi1>
    For two pure states the fidelity is given by
        F(|psi1>, |psi2>) = |<psi1|psi2>|^2

    Args:
        state1 (array_like): a quantum state vector or density matrix.
        state2 (array_like): a quantum state vector or density matrix.

    Returns:
        array_like: The state fidelity F(state1, state2).
    """
    # convert input to numpy arrays
    s1 = np.array(state1)
    s2 = np.array(state2)

    # fidelity of two state vectors
    if s1.ndim == 1 and s2.ndim == 1:
        return np.abs(s2.conj().dot(s1)) ** 2
    # fidelity of vector and density matrix
    elif s1.ndim == 1:
        # psi = s1, rho = s2
        return np.abs(s1.conj().dot(s2).dot(s1))
    elif s2.ndim == 1:
        # psi = s2, rho = s1
        return np.abs(s2.conj().dot(s1).dot(s2))
    # fidelity of two density matrices
    s1sq = _funm_svd(s1, np.sqrt)
    s2sq = _funm_svd(s2, np.sqrt)
    return np.linalg.norm(s1sq.dot(s2sq), ord='nuc') ** 2


def _funm_svd(a, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        a (array_like): (N, N) Matrix at which to evaluate the function.
        func (callable): Callable object that evaluates a scalar function f.

    Returns:
        ndarray: funm (N, N) Value of the matrix function specified by func
        evaluated at `A`.
    """
    U, s, Vh = la.svd(a, lapack_driver='gesvd')
    S = np.diag(func(s))
    return U.dot(S).dot(Vh)
