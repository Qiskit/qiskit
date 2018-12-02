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
from qiskit import QiskitError


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


def random_state(num):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        num (int): the number of qubits
    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    # Random array over interval (0, 1]
    x = np.random.random(1 << num)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = np.random.random(1 << num)*2.0*np.pi
    return np.sqrt(x/sumx)*np.exp(1j*phases)


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
