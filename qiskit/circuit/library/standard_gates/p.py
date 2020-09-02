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

"""Phase Gate."""

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class PhaseGate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ P(λ) ├
             └──────┘

    **Matrix Representation:**

    .. math::

        P(\lambda) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\lambda}
            \end{pmatrix}

    **Examples:**

        .. math::

            P(\lambda = \pi) = Z

        .. math::

            P(\lambda = \pi/2) = S

        .. math::

            P(\lambda = \pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                P(\lambda) = e^{i{\lambda}/2} RZ(\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, theta, label=None):
        """Create new Phase gate."""
        super().__init__('p', 1, [theta], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.u1(self.params[0], 0)
        self.definition = qc

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-Phase gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CPhaseGate(self.params[0], label=label, ctrl_state=ctrl_state)
        elif ctrl_state is None and num_ctrl_qubits > 1:
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits, label=label)
        else:
            return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                                   ctrl_state=ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        r"""Return inverted Phase gate (:math:`Phase(\lambda){\dagger} = Phase(-\lambda)`)"""
        return PhaseGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the Phase gate."""
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)


class CPhaseGate(ControlledGate):
    r"""Controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CPhase =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes P =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\lambda}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of Phase and RZ, CPhase and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta, label=None, ctrl_state=None):
        """Create new CPhase gate."""
        super().__init__('cp', 2, [theta], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
        self.base_gate = PhaseGate(theta)

    def _define(self):
        """
        gate cphase(lambda) a,b
        { phase(lambda/2) a; cx a,b;
          phase(-lambda/2) b; cx a,b;
          phase(lambda/2) b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[0] / 2, 0)
        qc.cx(0, 1)
        qc.p(-self.params[0] / 2, 1)
        qc.cx(0, 1)
        qc.p(self.params[0] / 2, 1)
        self.definition = qc

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
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1, label=label)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted CPhase gate (:math:`CPhase(\lambda){\dagger} = CPhase(-\lambda)`)"""
        return CPhaseGate(-self.params[0], ctrl_state=self.ctrl_state)

    def to_matrix(self):
        """Return a numpy.array for the CPhase gate."""
        eith = numpy.exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, eith]],
                               dtype=complex)
        return numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, eith, 0],
                            [0, 0, 0, 1]],
                           dtype=complex)


class MCPhaseGate(ControlledGate):
    r"""Multi-controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ───■────
                    │
                    .
                    │
        q_(n-1): ───■────
                 ┌──┴───┐
            q_n: ┤ P(λ) ├
                 └──────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CPhaseGate`:
        The singly-controlled-version of this gate.
    """

    def __init__(self, lam, num_ctrl_qubits, label=None):
        """Create new MCPhase gate."""
        super().__init__('mcphase', num_ctrl_qubits + 1, [lam], num_ctrl_qubits=num_ctrl_qubits,
                         label=label)
        self.base_gate = PhaseGate(lam)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)

        if self.num_ctrl_qubits == 0:
            qc.p(self.params[0], 0)
        if self.num_ctrl_qubits == 1:
            qc.cp(self.params[0], 0, 1)
        else:
            from .u3 import _gray_code_chain
            scaled_lam = self.params[0] / (2 ** (self.num_ctrl_qubits - 1))
            bottom_gate = CPhaseGate(scaled_lam)
            definition = _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate)
            qc.data = definition
        self.definition = qc

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
            gate = MCPhaseGate(self.params[0],
                               num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits,
                               label=label)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted MCU1 gate (:math:`MCU1(\lambda){\dagger} = MCU1(-\lambda)`)"""
        return MCPhaseGate(-self.params[0], self.num_ctrl_qubits)
