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
from qiskit.circuit.gate import Gate


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

    def __init__(self, v_x, v_y, v_z, basis='U3'):
        """Create new rv single-qubit gate."""
        from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
        super().__init__('rv', 1, [v_x, v_y, v_z])
        self._decomposer = OneQubitEulerDecomposer(basis)

    def _define(self):
        from qiskit.circuit import QuantumRegister, QuantumCircuit
        from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
        from qiskit.circuit.library.standard_gates import U3Gate
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        theta, phi, lam, global_phase = self._decomposer.angles_and_phase(self.to_matrix())
        qc._append(U3Gate(theta, phi, lam), [q[0]], [])
        qc.global_phase = global_phase
        self.definition = qc


    def inverse(self):
        """Invert this gate."""
        vx, vy, vz = self.params
        return RVGate(-vx, -vy, -vz)

    def to_matrix(self):
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        angle = numpy.sqrt(v.dot(v))
        nx, ny, nz = v / angle
        sin = numpy.sin(angle / 2)
        cos = numpy.cos(angle / 2)
        return numpy.array([[cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
                            [(ny - 1j * nx) * sin, cos + 1j * nz * sin]])
