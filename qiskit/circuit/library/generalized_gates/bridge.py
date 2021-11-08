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

"""Bridge gate."""

from typing import Optional
from itertools import chain

from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import CXGate


class BridgeGate(Gate):
    r"""The Bridge gate.

    For n = 3, the Bridge gate allows executing a CNOT between two qubits
    that share a common neighbor.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤0        ├
             │         │
        q_1: ┤1 bridge ├
             │         │
        q_2: ┤2        ├
             └─────────┘

        q_0: ──■──              q_0: ──■─────────■───────
               │                     ┌─┴─┐     ┌─┴─┐
        q_1: ──┼──      =       q_1: ┤ X ├──■──┤ X ├──■──
             ┌─┴─┐                   └───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├              q_2: ─────┤ X ├─────┤ X ├
             └───┘                        └───┘     └───┘

    For n > 3, the Bridge gate allows executing a CNOT between the first and
    last qubit assuming they are connected by a line path that passes through
    the intermediate qubits in the quantum device.
    """

    def __init__(self, num_qubits: int, label: Optional[str] = None):
        """Create a new Bridge gate."""
        super().__init__("bridge", num_qubits, [], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister

        n = self.num_qubits
        qr = QuantumRegister(n, name="q")
        qc = QuantumCircuit(qr, name=self.name)

        rules = []
        for _ in range(2):
            rules.extend(
                (CXGate(), [qr[i], qr[i + 1]], ()) for i in chain(range(n - 2), range(n - 2, 0, -1))
            )

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse Bridge gate (itself)."""
        return BridgeGate(self.num_qubits)  # self-inverse
