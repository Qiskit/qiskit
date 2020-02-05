# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Rotation around an axis in x-y plane.
"""
import math
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class RGate(Gate):
    """Rotation θ around the cos(φ)x + sin(φ)y axis."""

    def __init__(self, theta, phi):
        """Create new r single-qubit gate."""
        super().__init__("r", 1, [theta, phi])

    def _define(self):
        """
        gate r(θ, φ) a {u3(θ, φ - π/2, -φ + π/2) a;}
        """
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, "q")
        theta = self.params[0]
        phi = self.params[1]
        rule = [
            (U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate.

        r(θ, φ)^dagger = r(-θ, φ)
        """
        return RGate(-self.params[0], self.params[1])

    def to_matrix(self):
        """Return a Numpy.array for the R gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        exp_m = numpy.exp(-1j * self.params[1])
        exp_p = numpy.exp(1j * self.params[1])
        return numpy.array([[cos, -1j * exp_m * sin],
                            [-1j * exp_p * sin, cos]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def r(self, theta, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply R to q."""
    return self.append(RGate(theta, phi), [qubit], [])


QuantumCircuit.r = r
