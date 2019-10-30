# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A collection of useful quantum information functions for states.


"""

import numpy as np
import scipy.linalg as la


def state_fidelity(state1, state2):
    """Return the state fidelity between two quantum states.

    Either input may be a state vector, or a density matrix. The state
    fidelity (F) for two density matrices is defined as::

        F(rho1, rho2) = Tr[sqrt(sqrt(rho1).rho2.sqrt(rho1))] ^ 2

    For a pure state and mixed state the fidelity is given by::

        F(|psi1>, rho2) = <psi1|rho2|psi1>

    For two pure states the fidelity is given by::

        F(|psi1>, |psi2>) = |<psi1|psi2>|^2

    Args:
        state1 (array_like): a quantum state vector or density matrix.
        state2 (array_like): a quantum state vector or density matrix.

    Returns:
        array_like: The state fidelity F(state1, state2).
    """
    # convert input to numpy arrays
    state1 = np.array(state1)
    state2 = np.array(state2)

    # fidelity of two state vectors
    if state1.ndim == 1 and state2.ndim == 1:
        return np.abs(state2.conj().dot(state1)) ** 2
    # fidelity of vector and density matrix
    elif state1.ndim == 1:
        # psi = s1, rho = s2
        return np.abs(state1.conj().dot(state2).dot(state1))
    elif state2.ndim == 1:
        # psi = s2, rho = s1
        return np.abs(state2.conj().dot(state1).dot(state2))
    # fidelity of two density matrices
    s1sq = _funm_svd(state1, np.sqrt)
    s2sq = _funm_svd(state2, np.sqrt)
    return np.linalg.norm(s1sq.dot(s2sq), ord='nuc') ** 2


def _funm_svd(matrix, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        matrix (array_like): (N, N) Matrix at which to evaluate the function.
        func (callable): Callable object that evaluates a scalar function f.

    Returns:
        ndarray: funm (N, N) Value of the matrix function specified by func
        evaluated at `A`.
    """
    unitary1, singular_values, unitary2 = la.svd(matrix, lapack_driver='gesvd')
    diag_func_singular = np.diag(func(singular_values))
    return unitary1.dot(diag_func_singular).dot(unitary2)
