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

"""Rotation around an axis in x-y plane."""

import math
import numpy
from qiskit.qasm import pi
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RGate(Gate):
    r"""Rotation θ around the cos(φ)x + sin(φ)y axis.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ R(ϴ) ├
             └──────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        RY(\theta) = e^{-i \th Y} =
            \begin{pmatrix}
                \cos{\th} & -i e^{-i \phi} \sin{\th} \\
                -i e^{i \phi} \sin{\th} & \cos{\th}
            \end{pmatrix}
    """
    def __init__(self, theta, phi):
        """Create new r single-qubit gate."""
        super().__init__('r', 1, [theta, phi])

    def _define(self):
        """
        gate r(θ, φ) a {u3(θ, φ - π/2, -φ + π/2) a;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        theta = self.params[0]
        phi = self.params[1]
        rules = [
            (U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        """Invert this gate.

        r(θ, φ)^dagger = r(-θ, φ)
        """
        return RGate(-self.params[0], self.params[1])

    def to_matrix(self):
        """Return a numpy.array for the R gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        exp_m = numpy.exp(-1j * self.params[1])
        exp_p = numpy.exp(1j * self.params[1])
        return numpy.array([[cos, -1j * exp_m * sin],
                            [-1j * exp_p * sin, cos]], dtype=complex)
