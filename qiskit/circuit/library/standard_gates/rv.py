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


class RVGate(Gate):
    r"""Rotation around arbitrary rotation axis v where |v| is
    angle of rotation in radians.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ R(v) ├
             └──────┘

    **Matrix Representation:**

    .. math::

    \newcommand{\th}{|\vec{v}|}
    \newcommand{\sinc}{\text{sinc}}
        R(\vec{v}) = e^{-i \vec{v}\cdot\vec{\sigma}} =
            \begin{pmatrix}
                \cos{\th} -i v_z \sinc(\th) & -(i v_x + v_y) \sinc(\th) \\
                -(i v_x - v_y) \sinc(\th) & \cos(\th) + i v_z \sinc(\th)
            \end{pmatrix}
    """

    def __init__(self, v_x, v_y, v_z, seq='zyz'):
        """Create new rv single-qubit gate."""
        super().__init__('rv', 1, [v_x, v_y, v_z])
        self.seq = seq.lower()

    def _define(self):
        """
        gate r(v).
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from scipy.spatial.transform import Rotation
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        rot = Rotation.from_rotvec(numpy.asarray(self.params))
        euler = rot.as_euler(self.seq)
        getattr(qc, 'r' + self.seq[0])(euler[0], qr[0])
        getattr(qc, 'r' + self.seq[1])(euler[1], qr[0])
        getattr(qc, 'r' + self.seq[2])(euler[2], qr[0])
        self.definition = qc

    def inverse(self):
        """Invert this gate.
        """
        return RVGate(-self.params[0], -self.params[1], -self.params[2])

    def to_matrix(self):
        """Return a numpy.array for the R gate."""
        v = numpy.asarray(self.params, dtype=float)
        theta = numpy.sqrt(v.dot(v))
        sinc = numpy.sinc(theta)
        cos = numpy.cos(theta)
        # return numpy.array([[cos - v[2] * sinc, (-1j * v[0] + v[1]) * sinc],
        #                     [-(1j * v[0] - v[1]) * sinc, cos + 1j * v[2] * sinc]],
        #                    dtype=complex)
        return numpy.array([[   cos + 1j * v[2] * sinc,   (1j * v[0] + v[1]) * sinc],
                            [1j * (v[0] + v[1]) * sinc,      cos - 1j * v[2] * sinc]],
                           dtype=complex)

