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

"""Z and CZ gates."""

import numpy
from qiskit.qasm import pi
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class ZGate(Gate):
    r"""The single-qubit Pauli-Z gate (:math:`\sigma_z`).

    **Matrix Representation:**

    .. math::

        Z = \begin{pmatrix}
                1 & 0 \\
                0 & -1
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ Z ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the Z axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RZ(\pi)` and :math:`Z`.

        .. math::

            RZ(\pi) = \begin{pmatrix}
                        -1 & 0 \\
                        0 & 1
                      \end{pmatrix}
                    = -Z

    The gate is equivalent to a phase flip.

    .. math::

        |0\rangle \rightarrow |0\rangle \\
        |1\rangle \rightarrow -|1\rangle
    """

    def __init__(self, label=None):
        """Create new Z gate."""
        super().__init__('z', 1, [], label=label)

    def _define(self):
        from .u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-Z gate.

        One control returns a CZ gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CZGate(label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        """Return inverted Z gate (itself)."""
        return ZGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Z gate."""
        return numpy.array([[1, 0],
                            [0, -1]], dtype=complex)


class CZMeta(type):
    """A metaclass to ensure that CzGate and CZGate are of the same type.

    Can be removed when CzGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CZGate, CzGate}  # pylint: disable=unidiomatic-typecheck


class CZGate(ControlledGate, metaclass=CZMeta):
    r"""Controlled-Z gate.

    This is a Clifford and symmetric gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─■─

    **Matrix representation:**

    .. math::

        CZ\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes Z =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubit is in the :math:`|1\rangle` state.
    """

    def __init__(self, label=None, ctrl_state=None):
        """Create new CZ gate."""
        super().__init__('cz', 2, [], label=label, num_ctrl_qubits=1,
                         ctrl_state=ctrl_state)
        self.base_gate = ZGate()

    def _define(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        from .h import HGate
        from .x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverted CZ gate (itself)."""
        return CZGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CZ gate."""
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, -1]], dtype=complex)
        else:
            return numpy.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]], dtype=complex)


class CzGate(CZGate, metaclass=CZMeta):
    """The deprecated CZGate class."""

    def __init__(self, label=None, ctrl_state=None):
        import warnings
        warnings.warn('The class CzGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CZGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(label=label, ctrl_state=ctrl_state)
