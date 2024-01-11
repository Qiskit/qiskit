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
Instruction collection.
"""

from __future__ import annotations
from typing import Callable

from qiskit.circuit.exceptions import CircuitError
from .classicalregister import Clbit, ClassicalRegister
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction


class InstructionSet:
    """Instruction collection, and their contexts."""

    __slots__ = ("_instructions", "_requester")

    def __init__(  # pylint: disable=bad-docstring-quotes
        self,
        *,
        resource_requester: Callable[..., ClassicalRegister | Clbit] | None = None,
    ):
        """New collection of instructions.

        The context (``qargs`` and ``cargs`` that each instruction is attached to) is also stored
        separately for each instruction.

        Args:
            resource_requester: A callable that takes in the classical resource used in the
                condition, verifies that it is present in the attached circuit, resolves any indices
                into concrete :obj:`.Clbit` instances, and returns the concrete resource.  If this
                is not given, specifying a condition with an index is forbidden, and all concrete
                :obj:`.Clbit` and :obj:`.ClassicalRegister` resources will be assumed to be valid.

                .. note::

                    The callback ``resource_requester`` is called once for each call to
                    :meth:`.c_if`, and assumes that a call implies that the resource will now be
                    used.  It may throw an error if the resource is not valid for usage.

        """
        self._instructions: list[CircuitInstruction] = []
        self._requester = resource_requester

    def __len__(self):
        """Return number of instructions in set"""
        return len(self._instructions)

    def __getitem__(self, i):
        """Return instruction at index"""
        return self._instructions[i]

    def add(self, instruction, qargs=None, cargs=None):
        """Add an instruction and its context (where it is attached)."""
        if not isinstance(instruction, CircuitInstruction):
            if not isinstance(instruction, Operation):
                raise CircuitError("attempt to add non-Operation to InstructionSet")
            if qargs is None or cargs is None:
                raise CircuitError("missing qargs or cargs in old-style InstructionSet.add")
            instruction = CircuitInstruction(instruction, tuple(qargs), tuple(cargs))
        self._instructions.append(instruction)

    def inverse(self):
        """Invert all instructions."""
        for i, instruction in enumerate(self._instructions):
            self._instructions[i] = instruction.replace(operation=instruction.operation.inverse())
        return self

    def c_if(self, classical: Clbit | ClassicalRegister | int, val: int) -> "InstructionSet":
        """Set a classical equality condition on all the instructions in this set between the
        :obj:`.ClassicalRegister` or :obj:`.Clbit` ``classical`` and value ``val``.

        .. note::

            This is a setter method, not an additive one.  Calling this multiple times will silently
            override any previously set condition on any of the contained instructions; it does not
            stack.

        Args:
            classical: the classical resource the equality condition should be on.  If this is given
                as an integer, it will be resolved into a :obj:`.Clbit` using the same conventions
                as the circuit these instructions are attached to.
            val: the value the classical resource should be equal to.

        Returns:
            This same instance of :obj:`.InstructionSet`, but now mutated to have the given equality
            condition.

        Raises:
            CircuitError: if the passed classical resource is invalid, or otherwise not resolvable
                to a concrete resource that these instructions are permitted to access.

        Example:
            .. plot::
               :include-source:

               from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

               qr = QuantumRegister(2)
               cr = ClassicalRegister(2)
               qc = QuantumCircuit(qr, cr)
               qc.h(range(2))
               qc.measure(range(2), range(2))

               # apply x gate if the classical register has the value 2 (10 in binary)
               qc.x(0).c_if(cr, 2)

               # apply y gate if bit 0 is set to 1
               qc.y(1).c_if(0, 1)

               qc.draw('mpl')

        """
        if self._requester is None and not isinstance(classical, (Clbit, ClassicalRegister)):
            raise CircuitError(
                "Cannot pass an index as a condition variable without specifying a requester"
                " when creating this InstructionSet."
            )
        if self._requester is not None:
            classical = self._requester(classical)
        for instruction in self._instructions:
            instruction.operation.c_if(classical, val)
        return self

    # Legacy support for properties.  Added in Terra 0.21 to support the internal switch in
    # `QuantumCircuit.data` from the 3-tuple to `CircuitInstruction`.

    @property
    def instructions(self):
        """Legacy getter for the instruction components of an instruction set.  This does not
        support mutation."""
        return [instruction.operation for instruction in self._instructions]

    @property
    def qargs(self):
        """Legacy getter for the qargs components of an instruction set.  This does not support
        mutation."""
        return [list(instruction.qubits) for instruction in self._instructions]

    @property
    def cargs(self):
        """Legacy getter for the cargs components of an instruction set.  This does not support
        mutation."""
        return [list(instruction.clbits) for instruction in self._instructions]
