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
Qubit reset to computational zero.
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self):
        """Create new reset instruction."""
        super().__init__("reset", 1, 0, [])

    def broadcast_arguments(self, qargs, cargs):
        for qarg in qargs[0]:
            yield [qarg], []


def reset(self, qubit):
    """Reset q."""
    return self.append(Reset(), [qubit], [])


QuantumCircuit.reset = reset
