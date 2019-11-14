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
    r"""Rotation θ around the cos(φ)x + sin(φ)y axis.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{R}}(\theta, \phi)
            = \exp\left(-i (\cos(\phi)\sigma_X + \sin(\phi)\sigma_Y) \right)
            = \begin{bmatrix}
                \cos(\theta/2) & -i e^{-i\phi}\sin(\theta/2) \\
                -i e^{i\phi}\sin(\theta/2) & \cos(\theta/2)
            \end{bmatrix}
    """

    def __init__(self, theta, phi, phase=0, label=None):
        """Create new r single qubit gate."""
        super().__init__("r", 1, [theta, phi],
                         phase=phase, label=label)

    def _define(self):
        """
        gate r(θ, φ) a {u3(θ, φ - π/2, -φ + π/2) a;}
        """
        from qiskit.extensions.standard.u3 import U3Gate
        q = QuantumRegister(1, "q")
        theta = self.params[0]
        phi = self.params[1]
        self.definition = [
            (U3Gate(theta, phi - pi / 2, -phi + pi / 2,
                    phase=self.phase), [q[0]], [])
        ]

    def inverse(self):
        """Invert this gate.

        r(θ, φ)^dagger = r(-θ, φ)
        """
        return RGate(-self.params[0], self.params[1],
                     phase=-self.phase)

    def _matrix_definition(self):
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
