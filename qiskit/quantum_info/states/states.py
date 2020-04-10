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
A collection of DEPRECATED quantum information functions for states.
"""
import warnings
import numpy as np
from qiskit.exceptions import QiskitError


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
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' Please use `quantum_info.Statevector.from_label(label) for '
        ' equivalent functionality.', DeprecationWarning)
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
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' Please use `quantum_info.DensityMatrix(state) for '
        ' equivalent functionality.', DeprecationWarning)
    density_matrix = np.outer(state.conjugate(), state)
    if flatten:
        return density_matrix.flatten(order='F')
    return density_matrix
