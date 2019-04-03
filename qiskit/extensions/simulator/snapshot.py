# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simulator command to snapshot internal simulator representation.
"""
from qiskit import QuantumCircuit
from qiskit.circuit import CompositeGate
from qiskit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.extensions.exceptions import ExtensionError


class Snapshot(Instruction):
    """Simulator snapshot instruction."""

    def __init__(self, num_qubits, num_clbits, label, snap_type):
        """Create new snapshot instruction."""
        super().__init__("snapshot", num_qubits, num_clbits, [label, snap_type])

    def inverse(self):
        """Special case. Return self."""
        return Snapshot(self.num_qubits, self.num_clbits, self.params[0], self.params[1])


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
                qubits.append((tuple_element, j))
        else:
            qubits.append(tuple_element)
    return self.append(Snapshot(len(qubits), 0, label, snap_type), qubits, [])


# Add to QuantumCircuit and CompositeGate classes
QuantumCircuit.snapshot = snapshot
CompositeGate.snapshot = snapshot
