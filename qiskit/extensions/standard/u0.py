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
Single qubit gate cycle idle.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister


class U0Gate(Gate):
    """Wait gate."""

    def __init__(self, m, qubit, circ=None):
        """Create new u0 gate."""
        super().__init__("u0", [m], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        m = self.param[0]
        return self._qasmif("u0(%f) %s[%d];" % (m,
                                                qubit[0].openqasm_name,
                                                qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u0(self.param[0], self.arg[0]))


def u0(self, m, q):
    """Apply u0 with length m to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u0(m, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U0Gate(m, q, self))


QuantumCircuit.u0 = u0
CompositeGate.u0 = u0
