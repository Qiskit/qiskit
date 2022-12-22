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
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear.linear_matrix_utils import (
    calc_inverse_matrix,
    check_invertible_binary_matrix,
)


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


def _optimize_cx_4_options(function: Callable, mat: np.ndarray, optimize_count: bool = True):
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
    best_qc = _choose_best_linear_circuit(circuits, optimize_count)
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


def _choose_best_linear_circuit(
    circuits: List[QuantumCircuit], optimize_count: bool = True
) -> QuantumCircuit:
    """Returns the best quantum circuit either in terms of gate count or depth.

    Args:
        circuits: a list of quantum circuits
        optimize_count: True if the number of CX gates is optimized, False if the depth is optimized.

    Returns:
        QuantumCircuit: the best quantum circuit out of the given circuits.
    """
    best_qc = circuits[0]
    for circuit in circuits[1:]:
        if _compare_linear_circuits(best_qc, circuit, optimize_count=optimize_count):
            best_qc = circuit
    return best_qc


def _linear_circuit_depth(qc: QuantumCircuit):
    """Computes the depth of a linear circuit (that is, a circuit that only contains CX and SWAP
    gates). This is similar to quantum circuit's depth method except that SWAPs are counted as
    depth-3 gates.
    """
    qubit_depths = [0] * qc.num_qubits
    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits)}
    for instruction in qc.data:
        if instruction.operation.name not in ["cx", "swap"]:
            raise CircuitError("The circuit contains non-linear gates.")
        new_depth = max(qubit_depths[bit_indices[q]] for q in instruction.qubits)
        new_depth += 3 if instruction.operation.name == "swap" else 1
        for q in instruction.qubits:
            qubit_depths[bit_indices[q]] = new_depth
    return max(qubit_depths)


def _compare_linear_circuits(
    qc1: QuantumCircuit, qc2: QuantumCircuit, optimize_count: bool = True
) -> bool:
    """Compares two quantum circuits either in terms of gate count or depth.

     Args:
        qc1: the first quantum circuit
        qc2: the second quantum circuit
        optimize_count: True if the number of CX gates is optimized, False if the depth is optimized.

    Returns:
        bool: ``False`` means that the first quantum circuit is "better", ``True`` means the second.
    """
    count1 = qc1.size()
    depth1 = _linear_circuit_depth(qc1)
    count2 = qc2.size()
    depth2 = _linear_circuit_depth(qc2)

    # Prioritize count, and if it has the same count, then also consider depth
    count2_is_better = (optimize_count and count1 > count2) or (
        not optimize_count and depth1 == depth2 and count1 > count2
    )
    # Prioritize depth, and if it has the same depth, then also consider count
    depth2_is_better = (not optimize_count and depth1 > depth2) or (
        optimize_count and count1 == count2 and depth1 > depth2
    )

    return count2_is_better or depth2_is_better


def _linear_circuit_complies_with_coupling_map(qc: QuantumCircuit, coupling_list: list) -> bool:
    """Returns whether a linear quantum circuit (consisting of CX and SWAP gates)
    only has connections from the coupling_list.
    """
    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits)}
    coupling_list_set = set(coupling_list)
    for circuit_instruction in qc.data:
        if len(circuit_instruction.qubits) != 2:
            return False
        q0 = bit_indices[circuit_instruction.qubits[0]]
        q1 = bit_indices[circuit_instruction.qubits[1]]
        if (q0, q1) not in coupling_list_set:
            return False
    return True
