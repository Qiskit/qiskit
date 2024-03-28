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

"Circuit operation representing a ``break`` from a loop."

from typing import Optional

from qiskit.circuit.instruction import Instruction
from .builder import InstructionPlaceholder, InstructionResources


class BreakLoopOp(Instruction):
    """A circuit operation which, when encountered, jumps to the end of the nearest enclosing loop.
    Can only be used inside loops.
    """

    def __init__(self, num_qubits: int, num_clbits: int, label: Optional[str] = None):
        """
        Args:
            num_qubits: the number of qubits this affects.
            num_clbits: the number of qubits this affects.
            label: an optional string label for the instruction.
        """
        super().__init__("break_loop", num_qubits, num_clbits, [], label=label)


class BreakLoopPlaceholder(InstructionPlaceholder):
    """A placeholder instruction for use in control-flow context managers, when the number of qubits
    and clbits is not yet known.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, *, label: Optional[str] = None):
        super().__init__("break_loop", 0, 0, [], label=label)

    def concrete_instruction(self, qubits, clbits):
        return (
            self._copy_mutable_properties(BreakLoopOp(len(qubits), len(clbits), label=self.label)),
            InstructionResources(qubits=tuple(qubits), clbits=tuple(clbits)),
        )

    def placeholder_resources(self):
        return InstructionResources()
