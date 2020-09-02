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

"""Rotation around the X axis."""

import math
import numpy
from qiskit.qasm import pi
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RXGate(Gate):
    r"""Single-qubit rotation about the X axis.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Rx(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        RX(\theta) = exp(-i \th X) =
            \begin{pmatrix}
                \cos{\th}   & -i\sin{\th} \\
                -i\sin{\th} & \cos{\th}
            \end{pmatrix}
    """

    def __init__(self, theta, label=None):
        """Create new RX gate."""
        super().__init__('rx', 1, [theta], label=label)

    def _define(self):
        """
        gate rx(theta) a {r(theta, 0) a;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .r import RGate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RGate(self.params[0], 0), [q[0]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-RX gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CRXGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted RX gate.

        :math:`RX(\lambda)^{\dagger} = RX(-\lambda)`
        """
        return RXGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the RX gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -1j * sin],
                            [-1j * sin, cos]], dtype=complex)


class CRXMeta(type):
    """A metaclass to ensure that CrxGate and CRXGate are of the same type.

    Can be removed when CrxGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CRXGate, CrxGate}  # pylint: disable=unidiomatic-typecheck


class CRXGate(ControlledGate, metaclass=CRXMeta):
    r"""Controlled-RX gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rx(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        CRX(\lambda)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RX(\theta) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos{\th} & 0 & -i\sin{\th} \\
                0 & 0 & 1 & 0 \\
                0 & -i\sin{\th} & 0 & \cos{\th}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Rx(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            CRX(\theta)\ q_1, q_0 =
            |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes RX(\theta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos{\th}   & -i\sin{\th} \\
                    0 & 0 & -i\sin{\th} & \cos{\th}
                \end{pmatrix}
    """

    def __init__(self, theta, label=None, ctrl_state=None):
        """Create new CRX gate."""
        super().__init__('crx', 2, [theta], num_ctrl_qubits=1,
                         label=label, ctrl_state=ctrl_state)
        self.base_gate = RXGate(theta)

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1(pi/2) t;
          cx c,t;
          u3(-theta/2,0,0) t;
          cx c,t;
          u3(theta/2,-pi/2,0) t;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .u3 import U3Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U1Gate(pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, -pi / 2, 0), [q[1]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse CRX gate (i.e. with the negative rotation angle)."""
        return CRXGate(-self.params[0], ctrl_state=self.ctrl_state)

    def to_matrix(self):
        """Return a numpy.array for the CRX gate."""
        half_theta = self.params[0] / 2
        cos = numpy.cos(half_theta)
        isin = 1j * numpy.sin(half_theta)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, cos, 0, -isin],
                                [0, 0, 1, 0],
                                [0, -isin, 0, cos]],
                               dtype=complex)
        else:
            return numpy.array([[cos, 0, -isin, 0],
                                [0, 1, 0, 0],
                                [-isin, 0, cos, 0],
                                [0, 0, 0, 1]],
                               dtype=complex)


class CrxGate(CRXGate, metaclass=CRXMeta):
    """The deprecated CRXGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CrxGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)
