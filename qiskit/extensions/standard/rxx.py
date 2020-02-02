# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Two-qubit XX-rotation gate.
"""
import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class RXXGate(Gate):
    """Two-qubit XX-rotation gate.

    This gate corresponds to the rotation U(θ) = exp(-1j * θ * X⊗X / 2)
    """

    def __init__(self, theta):
        """Create new rxx gate."""
        super().__init__("rxx", 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        from qiskit.extensions.standard.x import CnotGate
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.u2 import U2Gate
        from qiskit.extensions.standard.u3 import U3Gate
        from qiskit.extensions.standard.h import HGate
        definition = []
        q = QuantumRegister(2, "q")
        theta = self.params[0]
        rule = [
            (U3Gate(np.pi / 2, theta, 0), [q[0]], []),
            (HGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-theta), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], []),
            (U2Gate(-np.pi, np.pi - theta), [q[0]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return RXXGate(-self.params[0])

    # NOTE: we should use the following as the canonical matrix
    # definition but we don't include it yet since it differs from
    # the circuit decomposition matrix by a global phase
    # def to_matrix(self):
    #   """Return a Numpy.array for the RXX gate."""
    #    theta = float(self.params[0])
    #    return np.array([
    #        [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
    #        [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
    #        [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
    #        [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]], dtype=complex)


def rxx(self, theta, qubit1, qubit2):
    """Apply RXX to circuit."""
    return self.append(RXXGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.rxx = rxx
