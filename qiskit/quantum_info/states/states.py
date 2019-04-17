# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string

"""
A collection of useful quantum information functions for states.


"""
import logging
import numpy as np
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


def basis_state(str_state, num):
    """
    Return a basis state ndarray.

    Args:
        str_state (string): a string representing the state.
        num (int): the number of qubits
    Returns:
        ndarray:  state(2**num) a quantum state with basis basis state.
     Raises:
        QiskitError: if the dimensions is wrong
    """
    n = int(str_state, 2)
    if num >= len(str_state):
        state = np.zeros(1 << num, dtype=complex)
        state[n] = 1
        return state
    else:
        raise QiskitError('size of bitstring is greater than num.')


def projector(state, flatten=False):
    """
    maps a pure state to a state matrix

    Args:
        state (ndarray): the number of qubits
        flatten (bool): determine if state matrix of column work
    Returns:
        ndarray:  state_mat(2**num, 2**num) if flatten is false
        ndarray:  state_mat(4**num) if flatten is true stacked on by the column
    """
    density_matrix = np.outer(state.conjugate(), state)
    if flatten:
        return density_matrix.flatten(order='F')
    return density_matrix


def purity(state):
    """Calculate the purity of a quantum state.

    Args:
        state (ndarray): a quantum state
    Returns:
        float: purity.
    """
    rho = np.array(state)
    if rho.ndim == 1:
        return 1.0
    return np.real(np.trace(rho.dot(rho)))
