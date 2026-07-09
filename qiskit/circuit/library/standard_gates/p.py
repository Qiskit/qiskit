# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Phase Gate."""

from __future__ import annotations

import math
from cmath import exp

import numpy

from qiskit._accelerate.circuit import StandardGate
from qiskit.circuit._utils import _ctrl_state_to_int
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


class PhaseGate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.p` method.

    Circuit symbol:

    .. code-block:: text

             ┌──────┐
        q_0: ┤ P(θ) ├
             └──────┘

    Matrix representation:

    .. math::

        P(\theta) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{pmatrix}

    Examples:

        .. math::

            P(\theta = \pi) = Z

        .. math::

            P(\theta = \pi/2) = S

        .. math::

            P(\theta = \pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                P(\theta) = e^{i{\theta}/2} RZ(\theta)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    _standard_gate = StandardGate.Phase

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        """
        Args:
            theta: The rotation angle.
            label: An optional label for the gate.
        """
        super().__init__("p", 1, [theta], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌──────────┐
        # q: ┤ U(0,0,θ) ├
        #    └──────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Phase._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the Phase gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CPhaseGate`.
        For more than one control qubits, the controlled gate is implemented as :class:`.MCPhaseGate`.
        In each case, the value of ``annotated`` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CPhaseGate(
                self.params[0], label=label, ctrl_state=ctrl_state, _base_label=self.label
            )
        else:
            gate = MCPhaseGate(
                self.params[0],
                num_ctrl_qubits=num_ctrl_qubits,
                ctrl_state=ctrl_state,
                label=label,
                _base_label=self.label,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted Phase gate (:math:`Phase(\lambda)^{\dagger} = Phase(-\lambda)`)

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always another :class:`.PGate` with an inverse parameter value.

        Returns:
            PGate: inverse gate.
        """
        return PhaseGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the Phase gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, exp(1j * lam)]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return PhaseGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, PhaseGate):
            return self._compare_parameters(other)
        return False


class CPhaseGate(ControlledGate):
    r"""Controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cp` method.

    Circuit symbol:

    .. code-block:: text


        q_0: ─■──
              │θ
        q_1: ─■──


    Matrix representation:

    .. math::

        CPhase =
            I \otimes |0\rangle\langle 0| + P \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\theta}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of Phase and RZ, CPhase and CRZ are different gates with a relative
        phase difference.
    """

    _standard_gate = StandardGate.CPhase

    def __init__(
        self,
        theta: ParameterValueType,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        *,
        _base_label=None,
    ):
        """Create new CPhase gate."""
        super().__init__(
            "cp",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=PhaseGate(theta, label=_base_label),
        )

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌────────┐
        # q_0: ┤ P(θ/2) ├──■───────────────■────────────
        #      └────────┘┌─┴─┐┌─────────┐┌─┴─┐┌────────┐
        # q_1: ──────────┤ X ├┤ P(-θ/2) ├┤ X ├┤ P(θ/2) ├
        #                └───┘└─────────┘└───┘└────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CPhase._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the CPhase gate.

        The controlled gate is implemented as :class:`.MCPhaseGate`, regardless of
        the value of ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        gate = MCPhaseGate(
            self.params[0],
            num_ctrl_qubits=num_ctrl_qubits + 1,
            ctrl_state=new_ctrl_state,
            label=label,
            _base_label=self.label,
        )

        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted CPhase gate (:math:`CPhase(\lambda)^{\dagger} = CPhase(-\lambda)`)"""
        return CPhaseGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CPhase gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype
            )
        return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return CPhaseGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, CPhaseGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False


class MCPhaseGate(ControlledGate):
    r"""Multi-controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcp` method.

    The implementation of this gate is based on the method described in [1, 2],
    which requires :math:`O(n)` depth and :math:`O(n^2)` elementary gates

    Circuit symbol:

    .. code-block:: text

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

    References:
        [1] A. J. da Silva and D. K. Park,
        Linear-depth quantum circuits for multiqubit controlled gates,
        `Phys. Rev. A 106, 042602
        <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602>`__.

        [2] https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py
    """

    def __init__(
        self,
        lam: ParameterValueType,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        *,
        _base_label=None,
    ):
        """Create new MCPhase gate."""
        super().__init__(
            "mcphase",
            num_ctrl_qubits + 1,
            [lam],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=PhaseGate(lam, label=_base_label),
        )

    def _define(self):
        from qiskit.circuit import QuantumCircuit, QuantumRegister

        qr = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(qr)

        if self.num_ctrl_qubits == 0:
            qc.p(self.params[0], 0)
        elif self.num_ctrl_qubits == 1:
            qc.cp(self.params[0], 0, 1)
        else:
            phi = self.params[0]

            MCPhaseGate._apply_controlled_gates(qc, phi, self.num_qubits, step=1)
            MCPhaseGate._apply_controlled_gates(qc, phi, self.num_qubits, step=2)
            MCPhaseGate._apply_controlled_gates(qc, phi, self.num_qubits - 1, step=3)
            MCPhaseGate._apply_controlled_gates(qc, phi, self.num_qubits - 1, step=4)

        self.definition = qc

    @staticmethod
    def _apply_controlled_gates(circuit, phi, n_qubits, step):
        if step in [1, 3]:
            start = 0
            reverse = True
        else:
            start = 1
            reverse = False

        qubit_pairs = [
            (control, target) for target in range(n_qubits) for control in range(start, target)
        ]

        qubit_pairs.sort(key=lambda e: e[0] + e[1], reverse=reverse)

        for control, target in qubit_pairs:
            exponent = target - control
            if control == 0:
                exponent = exponent - 1
            param = 2**exponent

            if target == n_qubits - 1 and step in [1, 2]:
                sign = 1 if step == 1 else -1
                circuit.cp(sign * phi / param, control, target)
            else:
                if step == 1:
                    sign = 1
                elif step == 2:
                    sign = -1
                elif step == 3:
                    sign = -1 if control == 0 else 1
                else:
                    sign = 1 if control == 0 else -1
                circuit.crx(sign * math.pi / param, control, target)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the MCPhaseGate gate.


        The controlled gate is implemented as :class:`.MCPhaseGate`, regardless of
        the value of ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        gate = MCPhaseGate(
            self.params[0],
            num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits,
            ctrl_state=new_ctrl_state,
            label=label,
            _base_label=self.label,
        )

        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted MCPhase gate (:math:`MCPhase(\lambda)^{\dagger} = MCPhase(-\lambda)`)"""
        return MCPhaseGate(
            -self.params[0], num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )

    def __eq__(self, other):
        return (
            isinstance(other, MCPhaseGate)
            and self.num_ctrl_qubits == other.num_ctrl_qubits
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )
