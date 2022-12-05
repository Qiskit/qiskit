# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for handling linear reversible circuits."""

import copy
from typing import Callable, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix


def transpose_cx_circ(qc: QuantumCircuit):
    """Takes a circuit having only CX gates, and calculates its transpose.
    This is done by recursively replacing CX(i, j) with CX(j, i) in all instructions.

    Args:
        qc: a QuantumCircuit containing only CX gates.

    Returns:
        QuantumCircuit: the transposed circuit.

    Raises:
        CircuitError: if qc has a non-CX gate.
    """
    transposed_circ = QuantumCircuit(qc.qubits, qc.clbits, name=qc.name + "_transpose")
    for instruction in reversed(qc.data):
        if instruction.operation.name != "cx":
            raise CircuitError("The circuit contains non-CX gates.")
        transposed_circ._append(instruction.replace(qubits=reversed(instruction.qubits)))
    return transposed_circ


def optimize_cx_4_options(function: Callable, mat: np.ndarray, optimize_count: bool = True):
    """Get the best implementation of a circuit implementing a binary invertible matrix M,
    by considering all four options: M,M^(-1),M^T,M^(-1)^T.
    Optimizing either the CX count or the depth.

    Args:
        function: the synthesis function.
        mat: a binary invertible matrix.
        optimize_count: True if the number of CX gates in optimize, False if the depth is optimized.

    Returns:
        QuantumCircuit: an optimized QuantumCircuit, has the best depth or CX count of the four options.

    Raises:
        QiskitError: if mat is not an invertible matrix.
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError("The matrix is not invertible.")

    circuits = _cx_circuits_4_options(function, mat)
    best_qc = choose_best_circuit(circuits, optimize_count)
    return best_qc


def _cx_circuits_4_options(function: Callable, mat: np.ndarray) -> List[QuantumCircuit]:
    """Construct different circuits implementing a binary invertible matrix M,
    by considering all four options: M,M^(-1),M^T,M^(-1)^T.

    Args:
        function: the synthesis function.
        mat: a binary invertible matrix.

    Returns:
        List[QuantumCircuit]: constructed circuits.
    """
    circuits = []
    qc = function(mat)
    circuits.append(qc)

    for i in range(1, 4):
        mat_cpy = copy.deepcopy(mat)
        # i=1 inverse, i=2 transpose, i=3 transpose and inverse
        if i == 1:
            mat_cpy = calc_inverse_matrix(mat_cpy)
            qc = function(mat_cpy)
            qc = qc.inverse()
            circuits.append(qc)
        elif i == 2:
            mat_cpy = np.transpose(mat_cpy)
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
            circuits.append(qc)
        elif i == 3:
            mat_cpy = calc_inverse_matrix(np.transpose(mat_cpy))
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
            qc = qc.inverse()
            circuits.append(qc)

    return circuits


def choose_best_circuit(circuits: List[QuantumCircuit], optimize_count: bool = True) -> QuantumCircuit:
    """Returns the best quantum circuit either in terms of gate count or depth.

    Args:
        circuits: a list of quantum circuits
        optimize_count: True if the number of CX gates is optimized, False if the depth is optimized.

    Returns:
        QuantumCircuit: the best quantum circuit out of the given circuits.
    """
    best_qc = circuits[0]
    best_depth = circuits[0].depth()
    best_count = circuits[0].size()
    print(f"In choose_best_circuit: count = {best_count}, depth = {best_depth}")

    for circuit in circuits[1:]:
        new_depth = circuit.depth()
        new_count = circuit.size()
        print(f"In choose_best_circuit: count = {new_count}, depth = {new_depth}")

        # Prioritize count, and if it has the same count, then also consider depth
        better_count = (optimize_count and best_count > new_count) or (
            not optimize_count and best_depth == new_depth and best_count > new_count
        )
        # Prioritize depth, and if it has the same depth, then also consider count
        better_depth = (not optimize_count and best_depth > new_depth) or (
            optimize_count and best_count == new_count and best_depth > new_depth
        )

        if better_count or better_depth:
            best_count = new_count
            best_depth = new_depth
            best_qc = circuit

    return best_qc


def _compare_circuits(qc1: QuantumCircuit, qc2: QuantumCircuit, optimize_count: bool = True) -> bool:
    """Compares two quantum circuits either in terms of gate count or depth.

     Args:
        qc1: the first quantum circuit
        qc2: the second quantum circuit
        optimize_count: True if the number of CX gates is optimized, False if the depth is optimized.

    Returns:
        bool: ``False`` means that the first quantum circuit is "better", ``True`` means the second.
    """
    count1 = qc1.size()
    depth1 = qc1.depth()
    count2 = qc2.size()
    depth2 = qc2.depth()

    # Prioritize count, and if it has the same count, then also consider depth
    count2_is_better = (optimize_count and count1 > count2) or (
        not optimize_count and depth1 == depth2 and count1 > count2
    )
    # Prioritize depth, and if it has the same depth, then also consider count
    depth2_is_better = (not optimize_count and depth1 > depth2) or (
        optimize_count and count1 == count2 and depth1 > depth2
    )

    return count2_is_better or depth2_is_better
