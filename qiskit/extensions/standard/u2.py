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
One-pulse single-qubit gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import InstructionSet
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class U2Gate(Gate):
    """One-pulse single-qubit gate."""

    def __init__(self, phi, lam, qubit, circ=None):
        """Create new one-pulse single-qubit gate."""
        super().__init__("u2", [phi, lam], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        phi = self.param[0]
        lam = self.param[1]
        return self._qasmif("u2(%s,%s) %s[%d];" % (phi, lam,
                                                   qubit[0].openqasm_name,
                                                   qubit[1]))

    def inverse(self):
        """Invert this gate.

        u2(phi,lamb)^dagger = u2(-lamb-pi,-phi+pi)
        """
        phi = self.param[0]
        self.param[0] = -self.param[1] - pi
        self.param[1] = -phi + pi
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.u2(self.param[0], self.param[1], self.arg[0]))


def u2(self, phi, lam, q):
    """Apply u2 to q."""
    if isinstance(q, QuantumRegister):
        instructions = InstructionSet()
        for j in range(q.size):
            instructions.add(self.u2(phi, lam, (q, j)))
        return instructions

    self._check_qubit(q)
    return self._attach(U2Gate(phi, lam, q, self))


QuantumCircuit.u2 = u2
CompositeGate.u2 = u2
