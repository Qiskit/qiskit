# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simulator command to load a saved quantum state.
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.extensions._extensionerror import ExtensionError
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Load(Instruction):
    """Simulator load instruction."""

    def __init__(self, slot, qubits, circ):
        """Create new load instruction."""
        super().__init__("load", [slot], list(qubits), [], circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.load(self.param[0]))


def load(self, slot):
    """Load the internal simulator representation (statevector, probability,
    density matrix, clifford table)
    Works on all qubits, and prevents reordering (like barrier).

    Args:
        slot (int): a slot to load from

    Returns:
        QuantumCircuit: with attached command

    Raises:
        ExtensionError: malformed command
    """
    tuples = []
    if isinstance(self, QuantumCircuit):
        for register in self.qregs:
            tuples.append(register)
    if not tuples:
        raise ExtensionError("no qubits for load")
    if slot is None:
        raise ExtensionError("no load slot passed")
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
    return self._attach(Load(slot, qubits, self))


# Add to QuantumCircuit class
QuantumCircuit.load = load
