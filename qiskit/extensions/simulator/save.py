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
Simulator command to save the quantum state.
"""
from qiskit import Instruction
from qiskit import QuantumCircuit
from qiskit import CompositeGate
from qiskit import QuantumRegister
from qiskit.extensions._extensionerror import ExtensionError
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Save(Instruction):
    """Simulator save instruction."""

    def __init__(self, slot, qubits, circ):
        """Create save save instruction."""
        super().__init__("save", [slot], list(qubits), circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        string = "save(%d) " % self.param[0]
        for j in range(len(self.arg)):
            if len(self.arg[j]) == 1:
                string += "%s" % self.arg[j].name
            else:
                string += "%s[%d]" % (self.arg[j][0].name, self.arg[j][1])
            if j != len(self.arg) - 1:
                string += ","
        string += ";"
        return string

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.save(self.param[0], *self.arg))


def save(self, slot):
    """Save the internal simulator representation (statevector, probability,
    density matrix, clifford table).
    Works on all qubits, and prevents reordering (like barrier).

    Args:
        slot (int): a slot to save into

    Returns:
        QuantumCircuit: with attached command

    Raises:
        ExtensionError: malformed command
    """
    tuples = []
    if isinstance(self, QuantumCircuit):
        for register in self.regs.values():
            if isinstance(register, QuantumRegister):
                tuples.append(register)
    if not tuples:
        raise ExtensionError("no qubits for save")
    if slot is None:
        raise ExtensionError("no save slot passed")
    qubits = []
    for tuple_element in tuples:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                self._check_qubit((tuple_element, j))
                qubits.append((tuple_element, j))
        else:
            self._check_qubit(tuple_element)
            qubits.append(tuple_element)
    self._check_dups(qubits)
    return self._attach(Save(slot, qubits, self))


# Add to QuantumCircuit and CompositeGate classes
QuantumCircuit.save = save
CompositeGate.save = save
