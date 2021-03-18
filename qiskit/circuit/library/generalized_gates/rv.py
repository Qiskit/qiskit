# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around an arbitrary axis on the Bloch sphere."""

import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError


class RVGate(Gate):
    r"""Single-qubit rotation around arbitrary axis v, with angle ||v||.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────────┐
        q_0: ┤   RV(vx,vy,vz)  ├
             └─────────────────┘

    **Matrix Representation:**

    .. math::

    \newcommand{\th}{|\vec{v}|}
        R(\vec{v}) = e^{-i \vec{v}\cdot\vec{\sigma}} =
            \begin{pmatrix}
                \cos{\th} -i v_z \sin(\th) & -(i v_x + v_y) \sin(\th) \\
                -(i v_x - v_y) \sin(\th) & \cos(\th) + i v_z \sin(\th)
            \end{pmatrix}
    """

    def __init__(self, vx, vy, vz):
        """Create new rv single-qubit gate.

        Args:
            vx (float): x-component
            vy (float): y-component
            vz (float): z-component
        """
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
        super().__init__('rv', 1, [vx, vy, vz])
        self._decomposer = OneQubitEulerDecomposer(basis='U')

    def _define(self):
        try:
            self.definition = self._decomposer(self.to_matrix())
        except TypeError as ex:
            raise CircuitError(f'The {self.name} gate cannot be decomposed '
                               'with unbound parameters') from ex

    def inverse(self):
        """Invert this gate."""
        vx, vy, vz = self.params
        return RVGate(-vx, -vy, -vz)

    def to_matrix(self):
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        magnitude = numpy.sqrt(v.dot(v))
        angle = numpy.pi * magnitude
        if angle == 0:
            return numpy.array([[1, 0], [0, 1]])
        nx, ny, nz = v / magnitude
        sin = numpy.sin(angle / 2)
        cos = numpy.cos(angle / 2)
        return numpy.array([[cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
                            [(ny - 1j * nx) * sin, cos + 1j * nz * sin]])
