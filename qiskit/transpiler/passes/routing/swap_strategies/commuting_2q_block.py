# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate made of commuting two-qubit gates."""

from typing import Set

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode


class Commuting2QBlocks(Gate):
    """A gate made of commuting two-qubit gates.

    This gate is intended for use with commuting swap strategies to make it convenient
    for the swap strategy router to identify which blocks of operations commute.
    """

    def __init__(self, node_block: Set[DAGOpNode]):
        """Initialize the op.

        Args:
            node_block: A block of nodes that commute.
        """
        qubits, cbits = set(), set()
        for node in node_block:
            qubits.update(node.qargs)
            cbits.update(node.cargs)

        super().__init__("Commuting 2Q gates", num_qubits=len(qubits), params=[])
        self._node_block = node_block
        self.qubits = list(qubits)
