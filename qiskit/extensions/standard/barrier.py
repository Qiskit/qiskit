# -*- coding: utf-8 -*-

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
Barrier instruction.
"""
from qiskit import Instruction
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import QuantumRegister
from qiskit import InstructionSet
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, arg, circ):
        """Create new barrier instruction."""
        super().__init__("barrier", [], list(arg), circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        string = "barrier "
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                string += "%s" % self.arg[j].openqasm_name
            else:
                string += "%s[%d]" % (self.arg[j][0].openqasm_name, self.arg[j][1])
            if j != len(self.arg) - 1:
                string += ","
        string += ";"
        return string  # no c_if on barrier instructions

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ._attach(Barrier(self.arg, self)))


def barrier(self, q=None):
    """Apply barrier to q. If q is None, applies to all the """
    qregs = self.get_qregs().values() if q is None else []
    qubits = [q] if isinstance(q, tuple) else []

    if isinstance(q, QuantumRegister):
        qregs.append(q)

    for qreg in qregs:
        for j in range(qreg.size):
            qubits.append((qreg,j))

    if len(qubits) == 0:
        self._check_qubit(q)

    self._check_dups(qubits)
    for qubit in qubits:
        self._check_qubit(qubit)
    return self._attach(Barrier(qubits, self))

QuantumCircuit.barrier = barrier
CompositeGate.barrier = barrier
