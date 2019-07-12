# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy

import numpy as np
from qiskit.quantum_info import Pauli


def measure_pauli_z(data, pauli):
    """
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str: int})
        pauli (Pauli): a Pauli object

    Returns:
        float: Expected value of paulis given data
    """
    observable = 0.0
    num_shots = sum(data.values())
    p_z_or_x = np.logical_or(pauli.z, pauli.x)
    for key, value in data.items():
        bitstr = np.asarray(list(key))[::-1].astype(np.bool)
        # pylint: disable=no-member
        sign = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p_z_or_x)) else 1.0
        observable += sign * value
    observable /= num_shots
    return observable


def covariance(data, pauli_1, pauli_2, avg_1, avg_2):
    """
    Compute the covariance matrix element between two
    Paulis, given the measurement outcome.
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str:int})
        pauli_1 (Pauli): a Pauli class member
        pauli_2 (Pauli): a Pauli class member
        avg_1 (float): expectation value of pauli_1 on `data`
        avg_2 (float): expectation value of pauli_2 on `data`

    Returns:
        float: the element of the covariance matrix between two Paulis
    """
    cov = 0.0
    num_shots = sum(data.values())

    if num_shots == 1:
        return cov

    p1_z_or_x = np.logical_or(pauli_1.z, pauli_1.x)
    p2_z_or_x = np.logical_or(pauli_2.z, pauli_2.x)
    for key, value in data.items():
        bitstr = np.asarray(list(key))[::-1].astype(np.bool)
        # pylint: disable=no-member
        sign_1 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p1_z_or_x)) else 1.0
        sign_2 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p2_z_or_x)) else 1.0
        cov += (sign_1 - avg_1) * (sign_2 - avg_2) * value
    cov /= (num_shots - 1)
    return cov


def row_echelon_F2(matrix_in):
    """
    Computes the row Echelon form of a binary matrix on the binary
    finite field

    Args:
        matrix_in (numpy.ndarray): binary matrix

    Returns:
        numpy.ndarray : matrix_in in Echelon row form
    """
    size = matrix_in.shape

    for i in range(size[0]):
        pivot_index = 0
        for j in range(size[1]):
            if matrix_in[i, j] == 1:
                pivot_index = j
                break
        for k in range(size[0]):
            if k != i and matrix_in[k, pivot_index] == 1:
                matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)

    matrix_out_temp = copy.deepcopy(matrix_in)
    indices = []
    matrix_out = np.zeros(size)

    for i in range(size[0] - 1):
        if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
            indices.append(i)
    for row in np.sort(indices)[::-1]:
        matrix_out_temp = np.delete(matrix_out_temp, (row), axis=0)

    matrix_out[0:size[0] - len(indices), :] = matrix_out_temp
    matrix_out = matrix_out.astype(int)

    return matrix_out


def kernel_F2(matrix_in):
    """
    Computes the kernel of a binary matrix on the binary finite field

    Args:
        matrix_in (numpy.ndarray): binary matrix

    Returns:
        [numpy.ndarray]: the list of kernel vectors
    """
    size = matrix_in.shape
    kernel = []
    matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
    matrix_in_id_ech = (row_echelon_F2(matrix_in_id.transpose())).transpose()

    for col in range(size[1]):
        if (np.array_equal(matrix_in_id_ech[0:size[0], col], np.zeros(size[0])) and not
                np.array_equal(matrix_in_id_ech[size[0]:, col], np.zeros(size[1]))):
            kernel.append(matrix_in_id_ech[size[0]:, col])

    return kernel


def suzuki_expansion_slice_pauli_list(pauli_list, lam_coef, expansion_order):
    """
    Similar to _suzuki_expansion_slice_matrix, with the difference that this method
    computes the list of pauli terms for a single slice of the suzuki expansion,
    which can then be fed to construct_evolution_circuit to build the QuantumCircuit.
    #TODO: polish the docstring
    Args:
        pauli_list (list[list[complex, Pauli]]): the weighted pauli list??
        lam_coef (float): ???
        expansion_order (int): ???
    """
    if expansion_order == 1:
        half = [[lam_coef / 2 * c, p] for c, p in pauli_list]
        return half + list(reversed(half))
    else:
        pk = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
        side_base = suzuki_expansion_slice_pauli_list(
            pauli_list,
            lam_coef * pk,
            expansion_order - 1
        )
        side = side_base * 2
        middle = suzuki_expansion_slice_pauli_list(
            pauli_list,
            lam_coef * (1 - 4 * pk),
            expansion_order - 1
        )
        return side + middle + side


def check_commutativity(op_1, op_2, anti=False):
    """
    Check the commutativity between two operators.

    Args:
        op_1 (WeightedPauliOperator):
        op_2 (WeightedPauliOperator):
        anti (bool): if True, check anti-commutativity, otherwise check commutativity.

    Returns:
        bool: whether or not two operators are commuted or anti-commuted.
    """
    com = op_1 * op_2 - op_2 * op_1 if not anti else op_1 * op_2 + op_2 * op_1
    com.remove_zero_weights()
    return True if com.is_empty() else False
