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

"""
Hadamard gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.extensions.standard.t import TGate, TdgGate
from qiskit.extensions.standard.s import SGate, SdgGate
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
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
                1 & -2
            \end{pmatrix}
    """

    def __init__(self, label=None):
        """Create new H gate."""
        super().__init__('h', 1, [], label=label)

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        from qiskit.extensions.standard.u2 import U2Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U2Gate(0, pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

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
        if ctrl_state is None:
            if num_ctrl_qubits == 1:
                return CHGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted H gate (itself)."""
        return HGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the H gate."""
        return numpy.array([[1, 1],
                            [1, -1]], dtype=complex) / numpy.sqrt(2)


@deprecate_arguments({'q': 'qubit'})
def h(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply :class:`~qiskit.extensions.standard.HGate`."""
    return self.append(HGate(), [qubit], [])


QuantumCircuit.h = h


class CHGate(ControlledGate):
    r"""Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\rangle` state.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ H ├
             └─┬─┘
        q_1: ──■──

    **Matrix Representation:**

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

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which is how we present the gate above as well, resulting in textbook
        matrices. Instead, if we use q_0 as control, the matrix will be:

        .. math::

            CH\ q_0, q_1 =
                I \otimes |0\rangle\langle 0| + H \otimes |1\rangle\langle 1| =
                \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 1 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & -1
                \end{pmatrix}
    """

    def __init__(self):
        """Create new CH gate."""
        super().__init__('ch', 2, [], num_ctrl_qubits=1)
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
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (SGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (TGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (TdgGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (SdgGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverted CH gate (itself)."""
        return CHGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CH gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2)],
                            [0, 0, 1, 0],
                            [0, 1 / numpy.sqrt(2), 0, -1 / numpy.sqrt(2)]],
                           dtype=complex)


@deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
def ch(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.CHGate`."""
    return self.append(CHGate(), [control_qubit, target_qubit], [])


QuantumCircuit.ch = ch
