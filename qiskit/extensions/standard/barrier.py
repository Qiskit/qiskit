# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Barrier instruction.
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, qubits, circ=None):
        """Create new barrier instruction."""
        super().__init__("barrier", [], list(qubits), [], circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.barrier(*self.qargs))


def barrier(self, *qargs):
    """Apply barrier to circuit.
    If qargs is None, applies to all the qbits.
    Args is a list of QuantumRegister or single qubits.
    For QuantumRegister, applies barrier to all the qbits in that register."""
    qubits = []

    if not qargs:  # None
        for qreg in self.qregs:
            for j in range(qreg.size):
                qubits.append((qreg, j))

    for qarg in qargs:
        if isinstance(qarg, QuantumRegister):
            for j in range(qarg.size):
                qubits.append((qarg, j))
        else:
            qubits.append(qarg)

    self._check_dups(qubits)
    for qubit in qubits:
        self._check_qubit(qubit)
    return self._attach(Barrier(qubits, self))


QuantumCircuit.barrier = barrier
