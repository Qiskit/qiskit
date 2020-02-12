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

# pylint: disable=invalid-name

"""
Global Mølmer–Sørensen gate.

The Mølmer–Sørensen gate is native to ion-trap systems. The global MS can be
applied to multiple ions to entangle multiple qubits simultaneously.

In the two-qubit case, this is equivalent to an XX(theta) interaction,
and is thus reduced to the RXXGate.
"""


from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class MSGate(Gate):
    """Global Molmer-Sorensen gate."""

    def __init__(self, n_qubits, theta):
        """Create new MS gate."""
        super().__init__("ms", n_qubits, [theta])

    def _define(self):
        from qiskit.extensions.standard.rxx import RXXGate
        definition = []
        q = QuantumRegister(self.num_qubits, "q")
        rule = []
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                rule += [(RXXGate(self.params[0]), [q[i], q[j]], [])]

        for inst in rule:
            definition.append(inst)
        self.definition = definition


def ms(self, theta, qubits):
    """Apply MS to q1 and q2."""
    return self.append(MSGate(len(qubits), theta), qubits)


QuantumCircuit.ms = ms
