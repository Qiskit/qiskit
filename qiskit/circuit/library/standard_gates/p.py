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
import cmath
from cmath import exp
import numpy
from qiskit.circuit._utils import _ctrl_state_to_int
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


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
            lam = self.params[0]

            q_controls = list(range(self.num_ctrl_qubits))
            q_target = self.num_ctrl_qubits
            new_target = q_target
            for k in range(self.num_ctrl_qubits):
                # Note: it's better *not* to run transpile recursively
                qc.mcrz(lam / (2**k), q_controls, new_target, use_basis_gates=False)
                new_target = q_controls.pop()
            qc.p(lam / (2**self.num_ctrl_qubits), new_target)

        self.definition = qc

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


class ACPhaseGate(Gate):
    r"""Anti-controlled Phase gate.

    Applies a phase gate on the target qubit if the control is
    in the :math:`|0\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.acp` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐   ┌───┐
        q_0: ┤ X ├─■─┤ X ├
             └───┘ │ └───┘
                   │θ
        q_1: ──────■──────

    This is equivalent to a controlled-Phase gate with the control state
    set to :math:`|0\rangle`.

    Matrix representation:

    .. math::

        ACPhase(\theta)\ q_0, q_1 =
            P(\theta) \otimes |0\rangle\langle 0| + I \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{i\theta} & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

            q_0: ──────■──────
                       │θ
                 ┌───┐ │ ┌───┐
            q_1: ┤ X ├─■─┤ X ├
                 └───┘   └───┘

        .. math::

            ACPhase(\theta)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes P(\theta) +
                |1\rangle\langle 1| \otimes I =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & e^{i\theta} & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        """Create new ACPhase gate.

        Args:
            theta: The phase angle of the gate.
            label: An optional label for the gate.
        """
        super().__init__("acp", 2, [theta], label=label)

    def _define(self):
        """Decomposition: X on control, CPhase, X on control."""
        from qiskit.circuit import QuantumCircuit

        q = QuantumCircuit(2, name=self.name)
        q.x(0)
        q.cp(self.params[0], 0, 1)
        q.x(0)
        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted ACPhase gate
        (:math:`ACPhase(\theta)^{\dagger} = ACPhase(-\theta)`).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.ACPhaseGate` with an inverted parameter value.

        Returns:
            ACPhaseGate: inverse gate.
        """
        return ACPhaseGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the ACPhase gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        eith = cmath.exp(1j * float(self.params[0]))
        return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, ACPhaseGate):
            return self._compare_parameters(other)
        return False


class AMCPhaseGate(Gate):
    r"""Anti-controlled/controlled multi-qubit Phase gate.

    A multi-controlled Phase gate where some control qubits are anti-controls
    (activate on :math:`|0\rangle`) and others are regular controls
    (activate on :math:`|1\rangle`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.amcp` method.

    Circuit symbol (example with 2 anti-controls and 2 controls):

    .. code-block:: text

             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             ├───┤  │  ├───┤
        q_1: ┤ X ├──■──┤ X ├
             └───┘  │  └───┘
        q_2: ───────■───────
                    │
        q_3: ───────■───────
                 ┌──┴──┐
        q_4: ───┤ P(λ) ├────
                 └─────┘

    The decomposition applies X gates on the anti-control qubits, then an MCPhase
    gate on all control qubits (both anti-controls and regular controls),
    and finally X gates on the anti-control qubits again.

    Args:
        lam: The phase angle of the gate.
        num_anti_ctrl_qubits: The number of anti-control qubits.
        num_ctrl_qubits: The number of regular control qubits.
        label: An optional label for the gate.

    Raises:
        ValueError: If the total number of control qubits is less than 1.
    """

    def __init__(
        self,
        lam: ParameterValueType,
        num_anti_ctrl_qubits: int,
        num_ctrl_qubits: int,
        label: str | None = None,
    ):
        """Create new AMCPhase gate.

        Args:
            lam: The phase angle of the gate.
            num_anti_ctrl_qubits: The number of anti-control qubits.
            num_ctrl_qubits: The number of regular control qubits.
            label: An optional label for the gate.

        Raises:
            ValueError: If the total number of control qubits is less than 1.
        """
        total_ctrl = num_anti_ctrl_qubits + num_ctrl_qubits
        if total_ctrl < 1:
            raise ValueError(
                "AMCPhaseGate requires at least 1 control qubit "
                f"(got {num_anti_ctrl_qubits} anti-controls + {num_ctrl_qubits} controls = 0)."
            )
        if num_anti_ctrl_qubits < 0 or num_ctrl_qubits < 0:
            raise ValueError(
                "Number of anti-control and control qubits must be non-negative, "
                f"got num_anti_ctrl_qubits={num_anti_ctrl_qubits}, "
                f"num_ctrl_qubits={num_ctrl_qubits}."
            )
        self._num_anti_ctrl_qubits = num_anti_ctrl_qubits
        self._num_ctrl_qubits = num_ctrl_qubits
        num_qubits = total_ctrl + 1  # controls + target
        super().__init__("amcp", num_qubits, [lam], label=label)

    @property
    def num_anti_ctrl_qubits(self):
        """Return the number of anti-control qubits."""
        return self._num_anti_ctrl_qubits

    @property
    def num_ctrl_qubits(self):
        """Return the number of regular control qubits."""
        return self._num_ctrl_qubits

    def _define(self):
        """Decomposition: X on anti-controls, MCPhase on all controls, X on anti-controls."""
        from qiskit.circuit import QuantumCircuit

        n = self.num_qubits
        q = QuantumCircuit(n, name=self.name)

        # Apply X on all anti-control qubits
        for i in range(self._num_anti_ctrl_qubits):
            q.x(i)

        # Apply MCPhase with all control qubits controlling the target
        total_ctrl = self._num_anti_ctrl_qubits + self._num_ctrl_qubits
        all_qubits = list(range(n))
        q.append(MCPhaseGate(self.params[0], total_ctrl), all_qubits)

        # Undo X on all anti-control qubits
        for i in range(self._num_anti_ctrl_qubits):
            q.x(i)

        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted AMCPhase gate.

        :math:`AMCPhase(\lambda)^{\dagger} = AMCPhase(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.AMCPhaseGate` with an inverted parameter value.

        Returns:
            AMCPhaseGate: inverse gate.
        """
        return AMCPhaseGate(-self.params[0], self._num_anti_ctrl_qubits, self._num_ctrl_qubits)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the AMCPhase gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        n = self.num_qubits
        dim = 2**n
        mat = numpy.eye(dim, dtype=dtype or complex)

        na = self._num_anti_ctrl_qubits
        nc = self._num_ctrl_qubits
        target_bit = na + nc

        ctrl_pattern = 0
        for i in range(na, na + nc):
            ctrl_pattern |= 1 << i

        state = ctrl_pattern | (1 << target_bit)

        lam = float(self.params[0])
        mat[state, state] = cmath.exp(1j * lam)

        return mat

    def __eq__(self, other):
        if isinstance(other, AMCPhaseGate):
            return (
                self._num_anti_ctrl_qubits == other._num_anti_ctrl_qubits
                and self._num_ctrl_qubits == other._num_ctrl_qubits
                and self._compare_parameters(other)
            )
        return False
