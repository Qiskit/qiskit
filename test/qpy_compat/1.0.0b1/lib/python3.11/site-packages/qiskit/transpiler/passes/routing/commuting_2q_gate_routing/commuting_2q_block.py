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
from __future__ import annotations

from collections.abc import Iterable

from qiskit.exceptions import QiskitError
from qiskit.circuit import Gate, Qubit, Clbit
from qiskit.dagcircuit import DAGOpNode


class Commuting2qBlock(Gate):
    """A gate made of commuting two-qubit gates.

    This gate is intended for use with commuting swap strategies to make it convenient
    for the swap strategy router to identify which blocks of operations commute.
    """

    def __init__(self, node_block: Iterable[DAGOpNode]) -> None:
        """
        Args:
            node_block: A block of nodes that commute.

        Raises:
            QiskitError: If the nodes in the node block do not apply to two-qubits.
        """
        qubits: set[Qubit] = set()
        cbits: set[Clbit] = set()
        for node in node_block:
            if len(node.qargs) != 2:
                raise QiskitError(f"Node {node.name} does not apply to two-qubits.")

            qubits.update(node.qargs)
            cbits.update(node.cargs)

        if cbits:
            raise QiskitError(
                f"{self.__class__.__name__} does not accept nodes with classical bits."
            )

        super().__init__(
            "commuting_2q_block", num_qubits=len(qubits), params=[], label="Commuting 2q gates"
        )
        self.node_block = node_block
        self.qubits = qubits

    def __iter__(self):
        """Iterate through the nodes in the block."""
        return iter(self.node_block)
