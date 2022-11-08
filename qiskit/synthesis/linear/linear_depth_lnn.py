# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimize the synthesis of an n-qubit circuit contains only CX gates for
linear nearest neighbor (LNN) connectivity.
The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2

References:
    [1] S. A. Kutin, D. P. Moulton, and L. M. Smithline, “Computation at a distance,” 2007.
"""

from copy import deepcopy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
    calc_inverse_matrix,
    check_invertible_binary_matrix,
)


def _el_op(mat, ctrl, trgt, row_op=True):
    # Perform a ROW or COL operation on a matrix mat
    if row_op:
        mat[trgt] = mat[trgt] ^ mat[ctrl]
    else:
        mat[:, ctrl] = mat[:, trgt] ^ mat[:, ctrl]


def _el_op_all(mat, mat_inv, ctrl, trgt, row_op=True):
    # Perform a ROW or COL operation on a matrix mat, and update the inverted mat
    _el_op(mat, ctrl, trgt, row_op=row_op)
    _el_op(mat_inv, ctrl, trgt, row_op=not row_op)


def _append_gate_lite(cx_instructions, mat, a, b):
    # Add a cx gate to the instructions and update the matrices
    cx_instructions.append((a, b))
    _el_op(mat, a, b)


def _get_lu_transformation(n, mat, mat_inv):
    # Get the instructions for LU decomposition of a matrix mat
    mat = deepcopy(mat)
    mat_t = deepcopy(mat)
    mat_inv_t = deepcopy(mat_inv)

    cx_instructions_rows = []

    # Use the instructions in U, which contains only gates of the form cx(a,b) a>b
    # to transform the matrix to a permuted lower-triangular matrix.
    # The original Matrix is unchanged.
    for i in reversed(range(0, n)):
        found_first = False
        # Find the last "1" in row i, use COL operations to the left in order to
        # zero out all other "1"s in that row.
        for j in reversed(range(0, n)):
            if mat[i, j]:
                if not found_first:
                    found_first = True
                    first_j = j
                else:
                    # cx_instructions_cols (L instructions) are not needed
                    _el_op(mat, j, first_j, row_op=False)
        # Use row operations directed upwards to zero out all "1"s above the remaining "1" in row i
        for k in reversed(range(0, i)):
            if mat[k, first_j]:
                _append_gate_lite(cx_instructions_rows, mat, i, k)

    # Apply only U instructions to get the permuted L
    for inst in cx_instructions_rows:
        _el_op_all(mat_t, mat_inv_t, inst[0], inst[1])
    return mat_t, mat_inv_t


def _get_label_arr(n, mat_t):
    # For each row in mat_t, save the column index of the last "1"
    label_arr = []
    for i in range(n):
        j = 0
        while not mat_t[i, n - 1 - j]:
            j += 1
        label_arr.append(j)
    return label_arr


def _in_linear_combination(label_arr_t, mat_inv_t, row, k):
    # Check if "row" is a linear combination of all rows in mat_inv_t not including the row labeled by k
    indx_k = label_arr_t[k]
    w_needed = np.zeros(len(row), dtype=bool)
    # Find the linear combination of mat_t rows which produces "row"
    for row_l, _ in enumerate(row):
        if row[row_l]:
            # mat_inv_t can be thought of as a set of instructions. Row l in mat_inv_t
            # indicates which rows from mat_t are necessary to produce the elementary vector e_l
            w_needed = w_needed ^ mat_inv_t[row_l]
    # If the linear combination requires the row labeled by k
    if w_needed[indx_k]:
        return False
    return True


def _get_label_arr_t(n, label_arr):
    # Returns label_arr_t = label_arr^(-1)
    label_arr_t = [None] * n
    for i in range(n):
        label_arr_t[label_arr[i]] = i
    return label_arr_t


def _matrix_to_north_west(n, cx_instructions_rows, mat, mat_inv):
    # Transform an arbitrary boolean invertible matrix to a north-west triangular matrix
    # by Proposition 7.3 in [1]

    # The rows of mat_t hold all w_j vectors [1]. mat_inv_t is the inverted matrix of mat_t
    mat_t, mat_inv_t = _get_lu_transformation(n, mat, mat_inv)

    # Get all pi(i) labels
    label_arr = _get_label_arr(n, mat_t)

    # Save the original labels, exchange index <-> value
    label_arr_t = _get_label_arr_t(n, label_arr)

    first_qubit = 0
    empty_layers = 0
    done = False
    while not done:
        # At each iteration the values of i switch between even and odd
        i = first_qubit
        at_least_one_needed = False
        while i + 1 < n:
            # "If j < k, we do nothing" (see [1])
            # "If j > k, we swap the two labels, and we also perform a box" (see [1])
            if label_arr[i] > label_arr[i + 1]:
                at_least_one_needed = True
                # "Let W be the span of all w_l for l!=k" (see [1])
                # " We can perform a box on <i> and <i + 1> that writes a vector in W to wire <i + 1>."
                # (see [1])
                if _in_linear_combination(label_arr_t, mat_inv_t, mat[i + 1], label_arr[i + 1]):
                    pass

                elif _in_linear_combination(
                    label_arr_t, mat_inv_t, mat[i + 1] ^ mat[i], label_arr[i + 1]
                ):
                    _append_gate_lite(cx_instructions_rows, mat, i, i + 1)

                elif _in_linear_combination(label_arr_t, mat_inv_t, mat[i], label_arr[i + 1]):
                    _append_gate_lite(cx_instructions_rows, mat, i + 1, i)
                    _append_gate_lite(cx_instructions_rows, mat, i, i + 1)

                label_arr[i], label_arr[i + 1] = label_arr[i + 1], label_arr[i]

            i = i + 2

        if not at_least_one_needed:
            empty_layers += 1
            if empty_layers > 1:  # if nothing happened twice in a row, then finished.
                done = True
        else:
            empty_layers = 0

        first_qubit = int(not first_qubit)


def _north_west_to_identity(n, cx_instructions_rows, mat):
    # Transform a north-west triangular matrix to identity in depth 3*n by Proposition 7.4 of [1]

    # At start the labels are in reversed order
    label_arr = list(reversed(range(n)))
    first_qubit = 0
    empty_layers = 0
    done = False
    while not done:
        i = first_qubit
        at_least_one_needed = False
        while i + 1 < n:
            # Exchange the labels if needed
            if label_arr[i] > label_arr[i + 1]:
                at_least_one_needed = True

                # If row i has "1" in colum i+1, swap and remove the "1" (in depth 2)
                # otherwise, only do a swap (in depth 3)
                if not mat[i, label_arr[i + 1]]:
                    # Adding this turns the operation to a SWAP
                    _append_gate_lite(cx_instructions_rows, mat, i + 1, i)

                _append_gate_lite(cx_instructions_rows, mat, i, i + 1)
                _append_gate_lite(cx_instructions_rows, mat, i + 1, i)

                label_arr[i], label_arr[i + 1] = label_arr[i + 1], label_arr[i]

            i = i + 2

        if not at_least_one_needed:
            empty_layers += 1
            if empty_layers > 1:  # if nothing happened twice in a row, then finished.
                done = True
        else:
            empty_layers = 0

        first_qubit = int(not first_qubit)


def _optimize_cx_circ_depth_5n_line(mat):
    # Optimize CX circuit in depth bounded by 5n for LNN connectivity.
    # The algorithm [1] has two steps:
    # a) transform the originl matrix to a North-West matrix (m2nw),
    # b) transform the north-west matrix to identity (nw2id).

    # According to [1] the synthesis is done on the inverse matrix
    # so the matrix mat is inverted at this step
    mat_inv = deepcopy(mat)
    mat_cpy = calc_inverse_matrix(mat_inv)

    n = len(mat_cpy)
    cx_instructions_rows_m2nw = []
    cx_instructions_rows_nw2id = []

    # Transform an arbitrary invertible matrix to a north-west triangular matrix
    # by Proposition 7.3 of [1]
    _matrix_to_north_west(n, cx_instructions_rows_m2nw, mat_cpy, mat_inv)

    # Transform a north-west triangular matrix to identity in depth 3*n
    # by Proposition 7.4 of [1]
    _north_west_to_identity(n, cx_instructions_rows_nw2id, mat_cpy)

    return cx_instructions_rows_m2nw, cx_instructions_rows_nw2id


def synth_cnot_depth_line_kms(mat):
    """
    Synthesis algorithm for linear reversible circuits from [1], Chapter 7.
    Synthesizes any linear reversible circuit of n qubits over linear nearest-neighbor
    architecture using CX gates with depth at most 5*n.

    Args:
        mat(np.ndarray]): A boolean invertible matrix.

    Returns:
        QuantumCircuit: the synthesized quantum circuit.

    Raises:
        QiskitError: if mat is not invertible.

    References:
        [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
             Computation at a Distance.
            `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError("The input matrix is not invertible.")

    # Returns the quantum circuit constructed from the instructions
    # that we got in _optimize_cx_circ_depth_5n_line
    num_qubits = len(mat)
    cx_inst = _optimize_cx_circ_depth_5n_line(mat)
    qc = QuantumCircuit(num_qubits)
    for pair in cx_inst[0]:
        qc.cx(pair[0], pair[1])
    for pair in cx_inst[1]:
        qc.cx(pair[0], pair[1])
    return qc
