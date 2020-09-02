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

"""Hadamard gate."""

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qasm import pi
from .t import TGate, TdgGate
from .s import SGate, SdgGate


class HGate(Gate):
    r"""Single-qubit Hadamard gate.

    This gate is a \pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\rangle,|1\rangle` to
    :math:`|+\rangle,|-\rangle` and vice-versa.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}
    """

    def __init__(self, label=None):
        """Create new H gate."""
        super().__init__('h', 1, [], label=label)

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u2 import U2Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U2Gate(0, pi), [q[0]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (multi-)controlled-H gate.

        One control qubit returns a CH gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CHGate(label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted H gate (itself)."""
        return HGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the H gate."""
        return numpy.array([[1, 1],
                            [1, -1]], dtype=complex) / numpy.sqrt(2)


class CHGate(ControlledGate):
    r"""Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\rangle` state.

    **Circuit symbol:**

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        CH\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + H \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
                0 & 0 & 1 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ H ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CH\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes H =
                \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 1 \\
                    0 & 0 & 1 & -1
                \end{pmatrix}
    """
    # Define class constants. This saves future allocation time.
    _sqrt2o2 = 1 / numpy.sqrt(2)
    _matrix1 = numpy.array([[1, 0, 0, 0],
                            [0, _sqrt2o2, 0, _sqrt2o2],
                            [0, 0, 1, 0],
                            [0, _sqrt2o2, 0, -_sqrt2o2]],
                           dtype=complex)
    _matrix0 = numpy.array([[_sqrt2o2, 0, _sqrt2o2, 0],
                            [0, 1, 0, 0],
                            [_sqrt2o2, 0, -_sqrt2o2, 0],
                            [0, 0, 0, 1]],
                           dtype=complex)

    def __init__(self, label=None, ctrl_state=None):
        """Create new CH gate."""
        super().__init__('ch', 2, [], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
        self.base_gate = HGate()

    def _define(self):
        """
        gate ch a,b {
            s b;
            h b;
            t b;
            cx a, b;
            tdg b;
            h b;
            sdg b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate  # pylint: disable=cyclic-import
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (SGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (TGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (TdgGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (SdgGate(), [q[1]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverted CH gate (itself)."""
        return CHGate(ctrl_state=self.ctrl_state)  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CH gate."""
        if self.ctrl_state:
            return self._matrix1
        else:
            return self._matrix0
