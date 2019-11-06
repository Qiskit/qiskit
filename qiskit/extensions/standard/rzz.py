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
two-qubit ZZ-rotation gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class RZZGate(Gate):
    """Two-qubit ZZ-rotation gate."""

    def __init__(self, theta):
        """Create new rzz gate."""
        super().__init__("rzz", 2, [theta])

    def _define(self):
        """
        gate rzz(theta) a, b { cx a, b; u1(theta) b; cx a, b; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0]), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return RZZGate(-self.params[0])


def rzz(self, theta, qubit1, qubit2):
    """Apply RZZ to circuit."""
    return self.append(RZZGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.rzz = rzz
