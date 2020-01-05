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
Rotation around the x-axis.
"""
import math
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.r import RGate


class RXGate(Gate):
    """rotation around the x-axis."""

    def __init__(self, theta):
        """Create new rx single qubit gate."""
        super().__init__("rx", 1, [theta])

    def _define(self):
        """
        gate rx(theta) a {r(theta, 0) a;}
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (RGate(self.params[0], 0), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        return RXGate(-self.params[0])

    def to_matrix(self):
        """Return a Numpy.array for the RX gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -1j * sin],
                            [-1j * sin, cos]], dtype=complex)


def rx(self, theta, q):  # pylint: disable=invalid-name
    """Apply Rx gate with angle theta to a specified qubit (q).
    The Rx gate corresponds to a rotation of theta radians from |0> about the
    x-axis on the Bloch sphere.

    Example:
    circuit = QuantumCircuit(1)
    theta = numpy.pi/2
    circuit.rx(theta,0)
    circuit.draw()
            ┌──────────┐
    q_0: |0>┤ Rx(pi/2) ├
            └──────────┘
    Resulting Statevector:
    [ 0.707+0j, 0-0.707j ]
    """
    return self.append(RXGate(theta), [q], [])


QuantumCircuit.rx = rx
