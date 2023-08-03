# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Converts any block of 2 qubit gates into a matrix."""

from numpy import identity, kron
from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError


SWAP_GATE = SwapGate()
SWAP_MATRIX = SWAP_GATE.to_matrix()
IDENTITY = identity(2, dtype=complex)


def _block_to_matrix(block, block_index_map):
    """
    The function converts any sequence of operations between two qubits into a matrix
    that can be utilized to create a gate or a unitary.

    Args:
        block (List(DAGOpNode)): A block of operations on two qubits.
        block_index_map (dict(Qubit, int)): The mapping of the qubit indices in the main circuit.

    Returns:
        NDArray: Matrix representation of the block of operations.
    """
    block_index_length = len(block_index_map)
    if block_index_length != 2:
        raise QiskitError(
            "This function can only operate with blocks of 2 qubits."
            + f"This block had {block_index_length}"
        )
    matrix = identity(2**block_index_length, dtype=complex)
    for node in block:
        try:
            current = node.op.to_matrix()
        except QiskitError:
            current = Operator(node.op).data
        q_list = [block_index_map[qubit] for qubit in node.qargs]
        if len(q_list) > 2:
            raise QiskitError(
                f"The operation {node.op.name} in this block has "
                + f"{len(q_list)} qubits, only 2 max allowed."
            )
        basis_change = False
        if len(q_list) < block_index_length:
            if q_list[0] == 1:
                current = kron(current, IDENTITY)
            else:
                current = kron(IDENTITY, current)
        else:
            if q_list[0] > q_list[1]:
                if node.op != SWAP_GATE:
                    basis_change = True
        if basis_change:
            matrix = (SWAP_MATRIX @ current) @ (SWAP_MATRIX @ matrix)
        else:
            matrix = current @ matrix
    return matrix
