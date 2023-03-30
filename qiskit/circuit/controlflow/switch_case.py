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

"""Circuit operation representing an ``switch/case`` statement."""

__all__ = ("SwitchCaseOp", "CASE_DEFAULT")

import sys
from typing import Union, Iterable, Any, Tuple, Optional, List

from qiskit.circuit import ClassicalRegister, Clbit, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from .control_flow import ControlFlowOp

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class _DefaultCaseType:
    """The type of the default-case singleton.  This is used instead of just having
    ``CASE_DEFAULT = object()`` so we can set the pretty-printing properties, which are class-level
    only."""

    def __repr__(self):
        return "<default case>"


CASE_DEFAULT = _DefaultCaseType()
"""A special object that represents the "default" case of a switch statement.  If you use this
as a case target, it must be the last case, and will match anything that wasn't already matched.
When using the builder interface of :meth:`.QuantumCircuit.switch`, this can also be accessed as the
``DEFAULT`` attribute of the bound case-builder object.
"""


class SwitchCaseOp(ControlFlowOp):
    """A circuit operation that executes one particular circuit block based on matching a given
    ``target`` against an ordered list of ``values``.  The special value :data:`.CASE_DEFAULT` can
    be used to represent a default condition.

    This is the low-level interface for creating a switch-case statement; in general, the circuit
    method :meth:`.QuantumCircuit.switch_case` should be used as a context manager to access the
    builder interface.  At the low level, you must ensure that all the circuit blocks contain equal
    numbers of qubits and clbits, and that the order the virtual bits of the containing circuit
    should be bound is the same for all blocks.  This will likely mean that each circuit block is
    wider than its natural width, as each block must span the union of all the spaces covered by
    _any_ of the blocks.

    Args:
        target: the runtime value to switch on.
        cases: an ordered iterable of the corresponding value of the ``target`` and the circuit
            block that should be executed if this is matched.  There is no fall-through between
            blocks, and the order matters.
    """

    def __init__(
        self,
        target: Union[Clbit, ClassicalRegister],
        cases: Iterable[Tuple[Any, QuantumCircuit]],
        *,
        label: Optional[str] = None,
    ):
        if not isinstance(target, (Clbit, ClassicalRegister)):
            raise CircuitError("the switch target must be a classical bit or register")

        target_bits = 1 if isinstance(target, Clbit) else len(target)
        target_max = (1 << target_bits) - 1

        case_ids = set()
        num_qubits, num_clbits = None, None
        self.target = target
        self._case_map = {}
        """Mapping of individual jump values to block indices.  This level of indirection is to let
        us more easily track the case of multiple labels pointing to the same circuit object, so
        it's easier for things like `assign_parameters`, which need to touch each circuit object
        exactly once, to function."""
        self._label_spec: List[Tuple[Union[int, Literal[CASE_DEFAULT]], ...]] = []
        """List of the normalised jump value specifiers.  This is a list of tuples, where each tuple
        contains the values, and the indexing is the same as the values of `_case_map` and
        `_params`."""
        self._params = []
        """List of the circuit bodies used.  This form makes it simpler for things like
        :meth:`.replace_blocks` and :class:`.QuantumCircuit.assign_parameters` to do their jobs
        without accidentally mutating the same circuit instance more than once."""
        for i, (value_spec, case_) in enumerate(cases):
            values = tuple(value_spec) if isinstance(value_spec, (tuple, list)) else (value_spec,)
            for value in values:
                if value in self._case_map:
                    raise CircuitError(f"duplicate case value {value}")
                if CASE_DEFAULT in self._case_map:
                    raise CircuitError("cases after the default are unreachable")
                if value is not CASE_DEFAULT:
                    if not isinstance(value, int) or value < 0:
                        raise CircuitError("case values must be Booleans or non-negative integers")
                    if value > target_max:
                        raise CircuitError(
                            f"switch target '{target}' has {target_bits} bit(s) of precision,"
                            f" but case {value} is larger than the maximum of {target_max}."
                        )
                self._case_map[value] = i
            self._label_spec.append(values)
            if not isinstance(case_, QuantumCircuit):
                raise CircuitError("case blocks must be QuantumCircuit instances")
            if id(case_) in case_ids:
                raise CircuitError("ungrouped cases cannot point to the same block")
            case_ids.add(id(case_))
            if num_qubits is None:
                num_qubits, num_clbits = case_.num_qubits, case_.num_clbits
            if case_.num_qubits != num_qubits or case_.num_clbits != num_clbits:
                raise CircuitError("incompatible bits between cases")
            self._params.append(case_)
        if not self._params:
            # This condition also implies that `num_qubits` and `num_clbits` must be non-None.
            raise CircuitError("must have at least one case to run")

        super().__init__("switch_case", num_qubits, num_clbits, self._params, label=label)

    def __eq__(self, other):
        # The general __eq__ will compare the blocks in the right order, so we just need to ensure
        # that all the labels point the right way as well.
        return super().__eq__(other) and all(
            set(labels_self) == set(labels_other)
            for labels_self, labels_other in zip(self._label_spec, other._label_spec)
        )

    def cases_specifier(self) -> Iterable[Tuple[Tuple, QuantumCircuit]]:
        """Return an iterable where each element is a 2-tuple whose first element is a tuple of
        jump values, and whose second is the single circuit block that is associated with those
        values.

        This is an abstract specification of the jump table suitable for creating new
        :class:`.SwitchCaseOp` instances.

        .. seealso::
            :meth:`.SwitchCaseOp.cases`
                Create a lookup table that you can use for your own purposes to jump from values to
                the circuit that would be executed."""
        return zip(self._label_spec, self._params)

    def cases(self):
        """Return a lookup table from case labels to the circuit that would be executed in that
        case.  This object is not generally suitable for creating a new :class:`.SwitchCaseOp`
        because any keys that point to the same object will not be grouped.

        .. seealso::
            :meth:`.SwitchCaseOp.cases_specifier`
                An alternate method that produces its output in a suitable format for creating new
                :class:`.SwitchCaseOp` instances.
        """
        return {key: self._params[index] for key, index in self._case_map.items()}

    @property
    def blocks(self):
        return tuple(self._params)

    def replace_blocks(self, blocks: Iterable[QuantumCircuit]) -> "SwitchCaseOp":
        blocks = tuple(blocks)
        if len(blocks) != len(self._params):
            raise CircuitError(f"needed {len(self._case_map)} blocks but received {len(blocks)}")
        return SwitchCaseOp(self.target, zip(self._label_spec, blocks))

    def c_if(self, classical, val):
        raise NotImplementedError(
            "SwitchCaseOp cannot be classically controlled through Instruction.c_if. "
            "Please nest it in an IfElseOp instead."
        )
