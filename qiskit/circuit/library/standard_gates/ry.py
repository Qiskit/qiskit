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

"""Rotation around the Y axis."""

import math
import numpy
from qiskit.qasm import pi
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RYGate(Gate):
    r"""Single-qubit rotation about the Y axis.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        RY(\theta) = exp(-i \th Y) =
            \begin{pmatrix}
                \cos{\th} & -\sin{\th} \\
                \sin{\th} & \cos{\th}
            \end{pmatrix}
    """

    def __init__(self, theta, label=None):
        """Create new RY gate."""
        super().__init__('ry', 1, [theta], label=label)

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .r import RGate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RGate(self.params[0], pi / 2), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-RY gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CRYGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted RY gate.

        :math:`RY(\lambda){\dagger} = RY(-\lambda)`
        """
        return RYGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the RY gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin],
                            [sin, cos]], dtype=complex)


class CRYMeta(type):
    """A metaclass to ensure that CryGate and CRYGate are of the same type.

    Can be removed when CryGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CRYGate, CryGate}  # pylint: disable=unidiomatic-typecheck


class CRYGate(ControlledGate, metaclass=CRYMeta):
    r"""Controlled-RY gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        CRY(\theta)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RY(\theta) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0         & 0 & 0 \\
                0 & \cos{\th} & 0 & -\sin{\th} \\
                0 & 0         & 1 & 0 \\
                0 & \sin{\th} & 0 & \cos{\th}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Ry(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            CRY(\theta)\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RY(\theta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos{\th} & -\sin{\th} \\
                    0 & 0 & \sin{\th} & \cos{\th}
                \end{pmatrix}
    """

    def __init__(self, theta, label=None, ctrl_state=None):
        """Create new CRY gate."""
        super().__init__('cry', 2, [theta], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
        self.base_gate = RYGate(theta)

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        """Return inverse RY gate (i.e. with the negative rotation angle)."""
        return CRYGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the CRY gate."""
        half_theta = self.params[0] / 2
        cos = numpy.cos(half_theta)
        sin = numpy.sin(half_theta)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, cos, 0, -sin],
                                [0, 0, 1, 0],
                                [0, sin, 0, cos]],
                               dtype=complex)
        else:
            return numpy.array([[cos, 0, -sin, 0],
                                [0, 1, 0, 0],
                                [sin, 0, cos, 0],
                                [0, 0, 0, 1]],
                               dtype=complex)


class CryGate(CRYGate, metaclass=CRYMeta):
    """The deprecated CRYGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CryGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRYGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)
