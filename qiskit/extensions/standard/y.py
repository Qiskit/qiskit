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
Y and CY gates.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class YGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`).

    **Matrix Representation:**

    .. math::

        Y = \begin{pmatrix}
                0 & -i \\
                i & 0
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ Y ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the Y axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RY(\pi)` and :math:`Y`.

        .. math::

            RY(\pi) = \begin{pmatrix}
                        0 & -1 \\
                        1 & 0
                      \end{pmatrix}
                    = -i.Y

    The gate is equivalent to a bit and phase flip.

    .. math::

        |0\rangle \rightarrow i|1\rangle \\
        |1\rangle \rightarrow -i|0\rangle
    """

    def __init__(self, label=None):
        """Create new Y gate."""
        super().__init__('y', 1, [], label=label)

    def _define(self):
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U3Gate(pi, pi / 2, pi / 2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-Y gate.

        One control returns a CY gate.

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
                return CYGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted Y gate (:math:`Y{\dagger} = Y`)"""
        return YGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Y gate."""
        return numpy.array([[0, -1j],
                            [1j, 0]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def y(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.YGate`."""
    return self.append(YGate(), [qubit], [])


QuantumCircuit.y = y


class CYMeta(type):
    """A metaclass to ensure that CyGate and CYGate are of the same type.

    Can be removed when CyGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CYGate, CyGate}  # pylint: disable=unidiomatic-typecheck


class CYGate(ControlledGate, metaclass=CYMeta):
    r"""Controlled-Y gate.

    **Circuit symbol:**

    .. parsed-literal::

                ┌───┐
        q_0: |0>┤ Y ├
                └─┬─┘
        q_1: |0>──■──


    **Matrix representation:**

    .. math::

        CY\ q_1, q_0 =
            |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes Y =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which is how we present the gate above as well, resulting in textbook
        matrices. Instead, if we use q_0 as control, the matrix will be:

        .. math::

            CY\ q_0, q_1 =
            I \otimes |0\rangle\langle0| + Y \otimes |1\rangle\langle1|  =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & -i \\
                    0 & 0 & 1 & 0 \\
                    0 & i & 0 & 0
                \end{pmatrix}

    """
    def __init__(self):
        """Create new CY gate."""
        super().__init__('cy', 2, [], num_ctrl_qubits=1)
        self.base_gate = YGate()

    def _define(self):
        """
        gate cy a,b { sdg b; cx a,b; s b; }
        """
        from qiskit.extensions.standard.s import SGate
        from qiskit.extensions.standard.s import SdgGate
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (SdgGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (SGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverted CY gate (:math:`CY^{\dagger} = CY`)"""
        return CYGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CY gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, -1j],
                            [0, 0, 1, 0],
                            [0, 1j, 0, 0]], dtype=complex)


class CyGate(CYGate, metaclass=CYMeta):
    """A deprecated CYGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class CyGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CYGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cy(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.CYGate`."""
    return self.append(CYGate(), [control_qubit, target_qubit], [])


QuantumCircuit.cy = cy
