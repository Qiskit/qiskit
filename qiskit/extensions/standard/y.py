# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Pauli Y (bit-phase-flip) gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class YGate(Gate):
    """Pauli Y (bit-phase-flip) gate."""

    def __init__(self, qubit, circ=None):
        """Create new Y gate."""
        super().__init__("y", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("y %s[%d];" % (qubit[0].openqasm_name, qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circuit):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circuit.y(self.arg[0]))


def y(self, q):
    """Apply Y to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.y((q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(YGate(q, self))


QuantumCircuit.y = y
CompositeGate.y = y
