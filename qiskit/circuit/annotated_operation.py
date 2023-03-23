import dataclasses
from typing import Optional, Union, List

from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError


class Modifier:
    pass


@dataclasses.dataclass
class InverseModifier(Modifier):
    pass


@dataclasses.dataclass
class ControlModifier(Modifier):
    num_ctrl_qubits: int
    ctrl_state: Union[int, str, None] = None

    def __init__(self, num_ctrl_qubits: int = 0, ctrl_state: Union[int, str, None] = None):
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)


@dataclasses.dataclass
class PowerModifier(Modifier):
    power: float


class AnnotatedOperation(Operation):
    """Gate and modifiers inside."""

    def __init__(
        self,
        base_op: Operation,
        modifiers: Union[Modifier, List[Modifier]]
    ):
        self.base_op = base_op
        self.modifiers = modifiers if isinstance(modifiers, List) else [modifiers]

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return "lazy"

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

    def lazy_inverse(self):
        """Returns lazy inverse
        """

        # ToDo: Should we copy base_op? modifiers?
        modifiers = self.modifiers.copy()
        modifiers.append(InverseModifier())
        return AnnotatedOperation(self.base_op, modifiers)

    def inverse(self):
        return self.lazy_inverse()

    def lazy_control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Maybe does not belong here"""
        modifiers = self.modifiers.copy()
        modifiers.append(ControlModifier(num_ctrl_qubits, ctrl_state))
        return AnnotatedOperation(self.base_op, modifiers)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        return self.lazy_control(num_ctrl_qubits, label, ctrl_state)

    def lazy_power(self, power: float) -> "AnnotatedOperation":
        modifiers = self.modifiers.copy()
        modifiers.append(PowerModifier(power))
        return AnnotatedOperation(self.base_op, modifiers)

    def power(self, power: float):
        return self.lazy_power(power)

    def __eq__(self, other) -> bool:
        """Checks if two AnnotatedOperations are equal."""
        return (
            isinstance(other, AnnotatedOperation)
            and self.modifiers == other.modifiers
            and self.base_op == other.base_op
        )

    def print_rec(self, offset=0, depth=100, header=""):
        """Temporary debug function."""
        line = " " * offset + header + " LazyGate " + self.name
        for modifier in self.modifiers:
            if isinstance(modifier, InverseModifier):
                line += "[inv] "
            elif isinstance(modifier, ControlModifier):
                line += "[ctrl=" + str(modifier.num_ctrl_qubits) + ", state=" + str(modifier.ctrl_state) + "] "
            elif isinstance(modifier, PowerModifier):
                line += "[power=" + str(modifier.power) + "] "
            else:
                raise CircuitError(f"Unknown modifier {modifier}.")

        print(line)
        if depth >= 0:
            self.base_op.print_rec(offset + 2, depth - 1, header="base gate")

    def copy(self) -> "AnnotatedOperation":
        """Return a copy of the :class:`AnnotatedOperation`."""
        return AnnotatedOperation(
            base_op=self.base_op,
            modifiers=self.modifiers.copy()
        )

    def to_matrix(self):
        """Return a matrix representation (allowing to construct Operator)."""
        from qiskit.quantum_info import Operator

        operator = Operator(self.base_op)

        for modifier in self.modifiers:
            if isinstance(modifier, InverseModifier):
                operator = operator.power(-1)
            elif isinstance(modifier, ControlModifier):
                operator = Operator(_compute_control_matrix(operator.data, modifier.num_ctrl_qubits, modifier.ctrl_state))
            elif isinstance(modifier, PowerModifier):
                operator = operator.power(modifier.power)
            else:
                raise CircuitError(f"Unknown modifier {modifier}.")
        return operator


def _canonicalize_modifiers(modifiers):
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
