# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Two-qubit YY-rotation gate.
"""
import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class RYYGate(Gate):
    r"""Two-qubit YY-rotation gate.

    This gate corresponds to the rotation :math:`U(\theta) = e^{-i\theta / 2 Y \otimes Y)`, 
    multiplied by the phase :math:`e^{i \theta / 2}`.
    """

    def __init__(self, theta):
        """Create new ryy gate."""
        super().__init__('ryy', 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        from qiskit.extensions.standard.x import CXGate
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.rx import RXGate
        definition = []
        q = QuantumRegister(2, 'q')
        theta = self.params[0]
        rule = [
            (RXGate(np.pi / 2), [q[0]], []),
            (RXGate(np.pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RXGate(-np.pi / 2), [q[0]], []),
            (RXGate(-np.pi / 2), [q[1]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return RYYGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.arry for the RYY gate."""
        theta = self.params[0]
        return np.exp(0.5j * theta) * np.array([
            [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]
        ], dtype=complex)


def ryy(self, theta, qubit1, qubit2):
    """Apply RYY to circuit."""
    return self.append(RYYGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.ryy = ryy
