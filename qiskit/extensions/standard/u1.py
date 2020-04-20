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
U1 Gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
class U1Gate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ U1(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        U1(\lambda) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\lambda}
            \end{pmatrix}

    **Examples:**

        .. math::

            U1(\lambda = \pi) = Z

        .. math::

            U1(\lambda = \pi/2) = S

        .. math::

            U1(\lambda = \pi/4) = T

    .. seealso::

        :class:`~qiskit.extensions.standard.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                U1(\lambda) = e^{i{\lambda}/2} RZ(\lambda)

        :class:`~qiskit.extensions.standard.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, theta, label=None):
        """Create new U1 gate."""
        super().__init__('u1', 1, [theta], label=label)

    def _define(self):
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U3Gate(0, 0, self.params[0]), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-U1 gate.

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
                return CU1Gate(self.params[0])
            return MCU1Gate(self.params[0], num_ctrl_qubits)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted U1 gate (:math:`U1(\lambda){\dagger} = U1(-\lambda)`)"""
        return U1Gate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the U1 gate."""
        lam = self.params[0]
        lam = float(lam)
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def u1(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply :class:`~qiskit.extensions.standard.U1Gate`."""
    return self.append(U1Gate(theta), [qubit], [])


QuantumCircuit.u1 = u1


class CU1Meta(type):
    """A metaclass to ensure that Cu1Gate and CU1Gate are of the same type.

    Can be removed when Cu1Gate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CU1Gate, Cu1Gate}  # pylint: disable=unidiomatic-typecheck


class CU1Gate(ControlledGate, metaclass=CU1Meta):
    r"""Controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CU1 =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U1 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\lambda}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.extensions.standard.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta):
        """Create new CU1 gate."""
        super().__init__('cu1', 2, [theta], num_ctrl_qubits=1)
        self.base_gate = U1Gate(theta)

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        definition = []
        q = QuantumRegister(2, 'q')
        from qiskit.extensions.standard.x import CXGate
        rule = [
            (U1Gate(self.params[0] / 2), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0] / 2), [q[1]], [])
        ]

        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            return MCU1Gate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted CU1 gate (:math:`CU1(\lambda){\dagger} = CU1(-\lambda)`)"""
        return CU1Gate(-self.params[0])


class Cu1Gate(CU1Gate, metaclass=CU1Meta):
    """The deprecated CU1Gate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class Cu1Gate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CU1Gate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cu1(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.CU1Gate`."""
    return self.append(CU1Gate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cu1 = cu1


class MCU1Gate(ControlledGate):
    r"""Multi-controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ────■────
                     │
                     .
                     │
        q_(n-1): ────■────
                 ┌───┴───┐
            q_n: ┤ U1(λ) ├
                 └───────┘

    .. seealso::

        :class:`~qiskit.extensions.standard.CU1Gate`:
        The singly-controlled-version of this gate.
    """

    def __init__(self, lam, num_ctrl_qubits):
        """Create new MCU1 gate."""
        super().__init__('mcu1', num_ctrl_qubits + 1, [lam], num_ctrl_qubits=num_ctrl_qubits)
        self.base_gate = U1Gate(lam)

    def _define(self):
        q = QuantumRegister(self.num_qubits, 'q')

        if self.num_ctrl_qubits == 0:
            definition = U1Gate(self.params[0]).definition
        if self.num_ctrl_qubits == 1:
            definition = CU1Gate(self.params[0]).definition
        else:
            from qiskit.extensions.standard.u3 import _gray_code_chain
            scaled_lam = self.params[0] / (2 ** (self.num_ctrl_qubits - 1))
            bottom_gate = CU1Gate(scaled_lam)
            definition = _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate)

        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            return MCU1Gate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted MCU1 gate (:math:`MCU1(\lambda){\dagger} = MCU1(-\lambda)`)"""
        return MCU1Gate(-self.params[0], self.num_ctrl_qubits)


def mcu1(self, lam, control_qubits, target_qubit):
    """Apply :class:`~qiskit.extensions.standard.CU1Gate`."""
    num_ctrl_qubits = len(control_qubits)
    return self.append(MCU1Gate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])


QuantumCircuit.mcu1 = mcu1
