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

import cmath
import math
import numpy
from qiskit.circuit import QuantumCircuit, ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import StandardGate


class RVGate(Gate):
    r"""Rotation around arbitrary rotation axis :math:`\vec{v}` where :math:`\|\vec{v}\|_2` is
    angle of rotation in radians.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rv` method.

    Circuit symbol:

    .. code-block:: text

             ┌─────────────────┐
        q_0: ┤ RV(v_x,v_y,v_z) ├
             └─────────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\|\vec{v}\|_2}{2}}
            R(\vec{v}) = e^{-i \vec{v}\cdot\vec{\sigma} / 2} =
                \begin{pmatrix}
                    \cos\left(\rotationangle\right)
                    -i \frac{v_z}{\|\vec{v}\|_2} \sin\left(\rotationangle\right)
                    & -(i \frac{v_x}{\|\vec{v}\|_2}
                    + \frac{v_y}{\|\vec{v}\|_2}) \sin\left(\rotationangle\right) \\
                    -(i \frac{v_x}{\|\vec{v}\|_2}
                    - \frac{v_y}{\|\vec{v}\|_2}) \sin\left(\rotationangle\right)
                    & \cos\left(\rotationangle\right)
                    + i \frac{v_z}{\|\vec{v}\|_2} \sin\left(\rotationangle\right)
                \end{pmatrix}
    """

    _standard_gate = StandardGate.RV

    def __init__(self, v_x: float, v_y: float, v_z: float, basis: str = "U"):
        """
        Args:
            v_x: x-component
            v_y: y-component
            v_z: z-component
            basis: basis (see
                :class:`~qiskit.synthesis.one_qubit.one_qubit_decompose.OneQubitEulerDecomposer`)
        """
        # pylint: disable=cyclic-import
        from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer

        super().__init__("rv", 1, [v_x, v_y, v_z])
        self._decomposer = None
        if basis != "U":
            self._decomposer = OneQubitEulerDecomposer(basis=basis)

    def _define(self):
        if self._decomposer is not None:
            try:
                self.definition = self._decomposer(self.to_matrix())
            except TypeError as ex:
                raise CircuitError(
                    f"The {self.name} gate cannot be decomposed with unbound parameters"
                ) from ex
        else:
            if all(x == 0 for x in self.params):
                self.definition = QuantumCircuit(1)
                return
            elif any(isinstance(x, ParameterExpression) for x in self.params):
                v_x = self.params[0]
                v_y = self.params[1]
                v_z = self.params[2]
                angle = ((v_x * v_x) + (v_y * v_y) + (v_z * v_z)) ** 0.5
                nx = v_x / angle
                ny = v_y / angle
                nz = v_z / angle
                sin = (angle / 2).sin()
                cos = (angle / 2).cos()
                u00_re = cos
                u00_im = -nz * sin
                u00 = u00_re + 1j * u00_im
                u01_re = -ny * sin
                u01_im = -nx * sin
                u01 = u01_re + 1j * u01_im
                u10_re = ny * sin
                u10_im = -nx * sin
                u10 = u10_re + 1j * u10_im
                u11_re = cos
                u11_im = nz * sin
                u11 = u11_re + 1j * u11_im
                real_det = ((u00_re * u11_re) - (u00_im * u11_im)) - (
                    (u01_re * u10_re) - (u01_im * u10_im)
                )
                im_det = ((u00_re * u11_im) + (u00_im * u11_re)) - (
                    (u01_re * u10_im) + (u01_im * u10_im)
                )
                det_arg = (real_det / im_det).arctan()
                theta = 2 * (u00.abs() / u10.abs()).arctan()
                ang1 = (u11_re / u11_im).arctan()
                ang2 = (u10_re / u10_im).arctan()
                phi = ang1 + ang2 - det_arg
                lam = ang1 - ang2
                phase = (0.5 * det_arg) - 0.5 * (phi + lam)
            else:
                u00, u01, u10, u11 = self._matrix_components()
                det = u00 * u11 - u01 * u10
                det_arg = cmath.phase(det)
                theta = 2 * math.atan2(abs(u10), abs(u00))
                ang1 = cmath.phase(u11)
                ang2 = cmath.phase(u10)
                phi = ang1 + ang2 - det_arg
                lam = ang1 - ang2
                phase = (0.5 * det_arg) - 0.5 * (phi + lam)
            qc = QuantumCircuit(1, global_phase=phase)
            qc.u(theta, phi, lam, 0)
            self.definition = qc

    def inverse(self, annotated: bool = False):
        """Invert this gate."""
        vx, vy, vz = self.params
        return RVGate(-vx, -vy, -vz)

    def _matrix_components(self):
        v_x = self.params[0]
        v_y = self.params[1]
        v_z = self.params[2]
        angle = math.sqrt((v_x * v_x) + (v_y * v_y) + (v_z * v_z))
        nx = v_x / angle
        ny = v_y / angle
        nz = v_z / angle
        sin = math.sin(angle / 2)
        cos = math.cos(angle / 2)
        u00 = cos - 1j * nz * sin
        u01 = (-ny - 1j * nx) * sin
        u10 = (ny - 1j * nx) * sin
        u11 = cos + 1j * nz * sin
        return u00, u01, u10, u11

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the R(v) gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        if all(x == 0 for x in self.params):
            return numpy.array([[1, 0], [0, 1]], dtype=dtype)
        else:
            u00, u01, u10, u11 = self._matrix_components()
            return numpy.array(
                [
                    [u00, u01],
                    [u10, u11],
                ],
                dtype=dtype,
            )
