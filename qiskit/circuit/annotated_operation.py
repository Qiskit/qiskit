# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Annotated Operations."""

import dataclasses
from typing import Union, List

from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError


class Modifier:
    """The base class that all modifiers of :class:`~.AnnotatedOperation` should
    inherit from."""

    pass


@dataclasses.dataclass
class InverseModifier(Modifier):
    """Inverse modifier: specifies that the operation is inverted."""

    pass


@dataclasses.dataclass
class ControlModifier(Modifier):
    """Control modifier: specifies that the operation is controlled by ``num_ctrl_qubits``
    and has control state ``ctrl_state``."""

    num_ctrl_qubits: int = 0
    ctrl_state: Union[int, str, None] = None

    def __init__(self, num_ctrl_qubits: int = 0, ctrl_state: Union[int, str, None] = None):
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)


@dataclasses.dataclass
class PowerModifier(Modifier):
    """Power modifier: specifies that the operation is raised to the power ``power``."""

    power: float


class AnnotatedOperation(Operation):
    """Annotated operation."""

    def __init__(self, base_op: Operation, modifiers: Union[Modifier, List[Modifier]]):
        """
        Create a new AnnotatedOperation.

        An "annotated operation" allows to add a list of modifiers to the
        "base" operation. For now, the only supported modifiers are of
        types :class:`~.InverseModifier`, :class:`~.ControlModifier` and
        :class:`~.PowerModifier`.

        An annotated operation can be viewed as an extension of
        :class:`~.ControlledGate` (which also allows adding control to the
        base operation). However, an important difference is that the
        circuit definition of an annotated operation is not constructed when
        the operation is declared, and instead happens during transpilation,
        specifically during the :class:`~.HighLevelSynthesis` transpiler pass.

        An annotated operation can be also viewed as a "higher-level"
        or "more abstract" object that can be added to a quantum circuit.
        This enables writing transpiler optimization passes that make use of
        this higher-level representation, for instance removing a gate
        that is immediately followed by its inverse.

        Args:
            base_op: base operation being modified
            modifiers: ordered list of modifiers. Supported modifiers include
                ``InverseModifier``, ``ControlModifier`` and ``PowerModifier``.

        Examples::

            op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2)])

            op2_inner = AnnotatedGate(SGate(), InverseModifier())
            op2 = AnnotatedGate(op2_inner, ControlModifier(2))

        Both op1 and op2 are semantically equivalent to an ``SGate()`` which is first
        inverted and then controlled by 2 qubits.
        """
        self.base_op = base_op
        self.modifiers = modifiers if isinstance(modifiers, List) else [modifiers]

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return "annotated"

    @property
    def num_qubits(self):
        """Number of qubits."""
        num_ctrl_qubits = 0
        for modifier in self.modifiers:
            if isinstance(modifier, ControlModifier):
                num_ctrl_qubits += modifier.num_ctrl_qubits

        return num_ctrl_qubits + self.base_op.num_qubits

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return self.base_op.num_clbits

    def __eq__(self, other) -> bool:
        """Checks if two AnnotatedOperations are equal."""
        return (
            isinstance(other, AnnotatedOperation)
            and self.modifiers == other.modifiers
            and self.base_op == other.base_op
        )

    def copy(self) -> "AnnotatedOperation":
        """Return a copy of the :class:`~.AnnotatedOperation`."""
        return AnnotatedOperation(base_op=self.base_op, modifiers=self.modifiers.copy())

    def to_matrix(self):
        """Return a matrix representation (allowing to construct Operator)."""
        from qiskit.quantum_info.operators import Operator  # pylint: disable=cyclic-import

        operator = Operator(self.base_op)

        for modifier in self.modifiers:
            if isinstance(modifier, InverseModifier):
                operator = operator.power(-1)
            elif isinstance(modifier, ControlModifier):
                operator = Operator(
                    _compute_control_matrix(
                        operator.data, modifier.num_ctrl_qubits, modifier.ctrl_state
                    )
                )
            elif isinstance(modifier, PowerModifier):
                operator = operator.power(modifier.power)
            else:
                raise CircuitError(f"Unknown modifier {modifier}.")
        return operator


def _canonicalize_modifiers(modifiers):
    """
    Returns the canonical representative of the modifier list. This is possible
    since all the modifiers commute; also note that InverseModifier is a special
    case of PowerModifier. The current solution is to compute the total number
    of control qubits / control state and the total power. The InverseModifier
    will be present if total power is negative, whereas the power modifier will
    be present only with positive powers different from 1.
    """
    power = 1
    num_ctrl_qubits = 0
    ctrl_state = 0

    for modifier in modifiers:
        if isinstance(modifier, InverseModifier):
            power *= -1
        elif isinstance(modifier, ControlModifier):
            num_ctrl_qubits += modifier.num_ctrl_qubits
            ctrl_state = (ctrl_state << modifier.num_ctrl_qubits) | modifier.ctrl_state
        elif isinstance(modifier, PowerModifier):
            power *= modifier.power
        else:
            raise CircuitError(f"Unknown modifier {modifier}.")

    canonical_modifiers = []
    if power < 0:
        canonical_modifiers.append(InverseModifier())
        power *= -1

    if power != 1:
        canonical_modifiers.append(PowerModifier(power))
    if num_ctrl_qubits > 0:
        canonical_modifiers.append(ControlModifier(num_ctrl_qubits, ctrl_state))

    return canonical_modifiers
