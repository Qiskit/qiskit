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
from qiskit.circuit.parameterexpression import ParameterExpression
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
        from scipy.spatial.transform import Rotation
        self.theta=None
        self.phi=None
        self.lam=None
        self.L = None

    def _define(self):
        """
        gate r(v).
        """
        from numpy import cos, sin, tan, arctan2, arccos, sqrt, arcsin, exp
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, ParameterExpression
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)

        if any([isinstance(param, ParameterExpression) for param in self.params]):
            if self.seq != 'zyz':
                raise NotImplementedError('the decomposition of parameter '
                                          'expressions only implmented for '
                                          'zxz decomposition')

            vx, vy, vz = self.params
            L = sqrt(vx**2 + vy**2 + vz**2)
            tanL = tan(L/2)

            self.theta = θ = 2 * arctan2(vy*vz*tan(L/2) - L*vx - sqrt(vx**4 + 2*(vx*vy)**2 + vy**4 +
                                                                      ((vx*vz)**2 + (vy*vz)**2)/cos(L/2)**2),
                                         vx*vz*tan(L/2) + L*vy)
            self.lam = λ = 2 * arctan2(vx * tan(θ/2) - vy, vx + vy * tan(θ/2))
            self.phi = φ = 2 * arcsin((-1j*vx + vy) * exp(1j*(λ-θ)/2) * sin(L/2))

            # self.theta = θ = 2 * arctan2(vx*vz * tan(L/2) + L*vy + sqrt(
            #     vz**4 + vy**4 + ((vx*vz)**2 + (vy*vz)**2)/cos(L/2)**2), L*vx - vy*vz*tan(L/2))
            # self.lam = λ = 2 * arctan2(vx + vy*tan(θ/2), vy - vx*tan(θ/2))
            #self.phi = φ = 2 * arccos(cos(L/2) / cos((λ + θ)/2))
            #self.phi = φ = 2 * arccos(vz*sin(L/2) / (L*sin((λ + θ)/2)))
            # self.phi =  φ = 2 * arcsin((-1j * vx + vy) * exp(1j * (λ - θ)/2) * sin(L/2) / L)
            #print('phi', self.phi, '\n phi2', self.phi2)
            #import ipdb;ipdb.set_trace()
            # θ = 2 * numpy.arctan2(vx*vz*tanL + vy + numpy.sqrt((vx**2 + vy**2) * vz**2 * tanL**2 + 1),
            #                       vx - vy*vz*tanL)
            # λ = 2 * numpy.arctan2((vx/vy) + numpy.arctan2(θ/2), 1 - (vx/vy) * numpy.arctan(θ/2))
            # φ = 2 * numpy.arccos(numpy.cos(vmag/2) / numpy.cos((λ+θ)/2))
            qc.L = L
            qc.rz(λ, 0)
            qc.ry(φ, 0)
            qc.rz(θ, 0)


            # v0, v1, v2 = self.params
            # vmag = numpy.sqrt(v0**2 + v1**2 + v2**2)
            # tanL = numpy.tan(vmag/2)

            # a = vmag**2 * v0 + numpy.tan(vmag/2) * v1 * v2
            # b = vmag**2 * v0 + numpy.tan(vmag/2) * v1 * v2 * vmag
            # c = 2 * vmag**2 * v1 - v0 * v2 * numpy.tan(vmag/2) - v0 * v2 * vmag * numpy.tan(vmag/2)

            # λ = 2 * numpy.arctan2(c + numpy.sqrt(c + 4*a*b), 2 * a)
            # θ = 2 * numpy.arctan2(v0 + v1 * numpy.tan(λ/2), v1 - v0 * numpy.tan(λ/2))
            # φ = 2 * numpy.arccos(numpy.cos(vmag/2) / numpy.cos((θ + λ) / 2))
            # qc.rz(λ, 0)
            # qc.ry(φ, 0)
            # qc.rz(θ, 0)

            # λ = 2 * numpy.arctan2(-v0 + tanL*v1*v2 + numpy.sqrt(v0**2 + v1**2 + v0**2 * v2**2 * tanL**2 + v1**2 * v2**2 * tanL**2),
            #                       v1 + v0*v2*tanL)
            # θ = 2 * numpy.arctan2(v0 * numpy.tan(λ/2) - v1, v0 + v1 * numpy.tan(λ/2))
            # φ = 2 * numpy.arcsin(v0 * numpy.sin(vmag/2) / numpy.cos((θ - λ) / 2))
            # qc.rz(λ, 0)
            # qc.rx(φ, 0)
            # qc.rz(θ, 0)

            # q0 = qi = v0 / norm * numpy.sin(norm / 2)
            # q1 = qj = v1 / norm * numpy.sin(norm / 2)
            # q2 = qk = v2 / norm * numpy.sin(norm / 2)
            # q3 = qr = numpy.cos(norm / 2)
            # self._quat = [q0, q1, q2, q3]
            # θ = numpy.arctan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1**2 + q2**2))
            # φ = numpy.arcsin(2 * (q0*q2 - q3*q1))
            # ψ = numpy.arctan2(2 * (q0*q3 + q1*q2), 1 - 2 * (q2**2 + q3*2))
            
            # this is from https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
            # φ = numpy.arctan2(qi*qk + qj*qr, -(qi*qk - qi*qr))
            # θ = numpy.arccos(-qi**2 - qj**2 + qk**2 + qr**2)
            # ψ = numpy.arctan2(qi*qk - qj*qr, qi*qk + qi*qr)
            # qc.rz(θ, 0)
            # qc.rx(φ, 0)
            # qc.rz(ψ, 0)
        else:
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_rotvec(self.params)
            euler = rot.as_euler(self.seq)
            for iaxis in range(3):
                if euler[iaxis]:
                    getattr(qc, 'r' + self.seq[iaxis])(euler[iaxis], qr[0])
        self.definition = qc

    def inverse(self):
        """Invert this gate."""
        v0, v1, v2 = self.params
        return RVGate(-v0, -v1, -v2, seq=self.seq)

    def to_matrix(self):
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        θ = numpy.sqrt(v.dot(v))
        nx, ny, nz = v / θ
        sin = numpy.sin(θ/2)
        cos = numpy.cos(θ/2)
        return numpy.array([[cos -1j * nz * sin, (-ny - 1j*nx) * sin],
                            [(ny - 1j*nx) * sin, cos + 1j * nz * sin]])


    def to_matrix2(self):
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        θ = numpy.sqrt(v.dot(v))
        nx, ny, nz = v / θ
        sinc = numpy.sinh(θ)
        cos = numpy.cos(θ)
        return numpy.array([[cos - 1j*v[2]*sinc, -(1j*v[0] + v[1]) * sinc],
                            [-(1j*v[0] - v[1]) * sinc, cos + 1j*v[2]*sinc]])
