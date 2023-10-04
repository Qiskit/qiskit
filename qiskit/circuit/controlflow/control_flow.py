# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"Container to encapsulate all control flow operations."

from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.store import Store
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.classical import expr

if typing.TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@dataclasses.dataclass
class VarUsage:
    """Used in the return value of :meth:`.ControlFlowOp.captured_var_usage` to store information
    about how variables are used.

    This is an attribute-based dataclass to allow backwards compatibility should the returned
    information need to expand in the future."""

    written: bool = dataclasses.field(default=False)
    """Whether the variable is written to in any path through the control-flow operation."""


class ControlFlowOp(Instruction, ABC):
    """Abstract class to encapsulate all control flow operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            if block.num_input_vars:
                raise CircuitError("control-flow blocks cannot contain input variables")

    @property
    @abstractmethod
    def blocks(self) -> tuple[QuantumCircuit, ...]:
        """Tuple of QuantumCircuits which may be executed as part of the
        execution of this ControlFlowOp. May be parameterized by a loop
        parameter to be resolved at run time.
        """

    @abstractmethod
    def replace_blocks(self, blocks: typing.Iterable[QuantumCircuit]) -> ControlFlowOp:
        """Replace blocks and return new instruction.
        Args:
            blocks: Tuple of QuantumCircuits to replace in instruction.

        Returns:
            New ControlFlowOp with replaced blocks.
        """

    def captured_var_usage(self) -> dict[expr.Var, VarUsage]:
        """Get information about the variables captured in the blocks of this operation."""

        # This is very inefficient to do on the tree structure, but until our graph-based circuits
        # represent control flow better, we need a way to query the operations for this information.
        # A better graph-based structure will let us chase def-use chains through to find
        # redefinitions of the variables, rather than needing an iteration through every operation.
        #
        # Using this while constructing separate DAGCircuit blocks for each part of the control-flow
        # op is also quadratic in the depth of the nested control-flow, since it examines nested
        # structure, but the construction would itself need to recurse into those blocks.  This is
        # again indicative in deficiencies in our graph-based IR in the presence of control flow.

        return _captured_var_usage_recurse(
            self, {var: VarUsage() for block in self.blocks for var in block.iter_captured_vars()}
        )


def _captured_var_usage_recurse(operation, usages):
    for block in operation.blocks:
        to_check = {
            var: usage
            for var in block.iter_captured_vars()
            # No need to do expensive checks of this block if we know there's a write.
            if not (usage := usages[var]).written
        }
        if not to_check:
            continue
        for instruction in block.data:
            if isinstance(instruction.operation, ControlFlowOp):
                usages = _captured_var_usage_recurse(instruction.operation, usages)
            elif isinstance(instruction.operation, Store):
                memory_location = instruction.operation.lvalue
                if (usage := to_check.get(memory_location)) is None:
                    continue
                usage.written = True
                to_check.pop(memory_location)
                if not to_check:
                    break
    return usages
