# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Barrier instruction.
"""
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import CompositeGate
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Instruction


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, num_qubits, circ=None):
        """Create new barrier instruction."""
        super().__init__("barrier", num_qubits, 0, [], circ)

    def inverse(self):
        """Special case. Return self."""
        return self


def barrier(self, *qargs):
    """Apply barrier to circuit.
    If qargs is None, applies to all the qbits.
    Args is a list of QuantumRegister or single qubits.
    For QuantumRegister, applies barrier to all the qubits in that register."""
    qubits = []

    if not qargs:  # None
        for qreg in self.qregs:
            for j in range(qreg.size):
                qubits.append((qreg, j))

    for qarg in qargs:
        if isinstance(qarg, (QuantumRegister, list)):
            if isinstance(qarg, QuantumRegister):
                qubits.extend([(qarg, j) for j in range(qarg.size)])
            else:
                qubits.extend(qarg)
        else:
            qubits.append(qarg)

    return self.append(Barrier(len(qubits)), qubits, [])


QuantumCircuit.barrier = barrier
CompositeGate.barrier = barrier
