# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simulator command to snapshot internal simulator representation.
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.extensions.exceptions import ExtensionError
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Snapshot(Instruction):
    """Simulator snapshot instruction."""

    def __init__(self, label, snap_type, qubits, circ):
        """Create new snapshot instruction."""
        super().__init__("snapshot", [label, snap_type], list(qubits), [], circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.snapshot(self.param[0], self.param[1]))


def snapshot(self, label, snap_type='statevector'):
    """Take a snapshot of the internal simulator representation (statevector)
    Works on all qubits, and prevents reordering (like barrier).

    Args:
        label (str): a snapshot label to report the result
        snap_type (str): a snapshot type (only supports statevector)

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
        raise ExtensionError("no qubits for snapshot")
    if label is None:
        raise ExtensionError("no snapshot label passed")
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
    return self._attach(Snapshot(label, snap_type, qubits, self))


# Add to QuantumCircuit class
QuantumCircuit.snapshot = snapshot
