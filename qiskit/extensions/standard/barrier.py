# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Barrier instruction.
"""
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.exceptions import QiskitError


class Barrier(Instruction):
    """Barrier instruction."""

    def __init__(self, num_qubits):
        """Create new barrier instruction."""
        super().__init__("barrier", num_qubits, 0, [])

    def inverse(self):
        """Special case. Return self."""
        return Barrier(self.num_qubits)

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise QiskitError('Barriers are compiler directives and cannot be conditional.')


def barrier(self, *qargs):
    """Apply barrier to circuit.
    If qargs is None, applies to all the qbits.
    Args is a list of QuantumRegister or single qubits.
    For QuantumRegister, applies barrier to all the qubits in that register."""
    qubits = []

    if not qargs:  # None
        for qreg in self.qregs:
            for j in range(qreg.size):
                qubits.append(qreg[j])

    for qarg in qargs:
        if isinstance(qarg, QuantumRegister):
            qubits.extend([qarg[j] for j in range(qarg.size)])
        elif isinstance(qarg, list):
            qubits.extend(qarg)
        elif isinstance(qarg, range):
            qubits.extend(list(qarg))
        elif isinstance(qarg, slice):
            qubits.extend(self.qubits[qarg])
        else:
            qubits.append(qarg)

    return self.append(Barrier(len(qubits)), qubits, [])


QuantumCircuit.barrier = barrier
