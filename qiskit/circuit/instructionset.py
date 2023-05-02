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
import functools
from typing import Callable

from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_arg
from .classicalregister import Clbit, ClassicalRegister
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction


# ClassicalRegister is hashable, and generally the registers in a circuit are completely fixed after
# its creation, so caching this allows us to only pay the register-unrolling penalty once.  The
# cache does not need to be large, because in general only one circuit is constructed at once.
@functools.lru_cache(4)
def _requester_from_cregs(
    cregs: tuple[ClassicalRegister],
) -> Callable[[Clbit | ClassicalRegister | int], ClassicalRegister | Clbit]:
    """Get a classical resource requester from an iterable of classical registers.

    This implements the deprecated functionality of constructing an :obj:`.InstructionSet` with a
    sequence of :obj:`.ClassicalRegister` instances.  This is the old method of resolving integer
    indices to a :obj:`.Clbit`, which is now replaced by using a requester from the
    :obj:`.QuantumCircuit` instance for consistency.

    .. note::

        This function has "incorrect" behaviour if any classical bit is in more than one register.
        This is to maintain compatibility with the legacy usage of :obj:`.InstructionSet`, and
        should not be used for any internal Qiskit code.  Instead, use the proper requester methods
        within :obj:`.QuantumCircuit`.

    This function can be removed when the deprecation of the ``circuit_cregs`` argument in
    :obj:`.InstructionSet` expires.

    Args:
        cregs: A tuple (needs to be immutable for the caching) of the classical registers to produce
            a requester over.

    Returns:
        A requester function that checks that a passed condition variable is valid, resolves
        integers into concrete :obj:`.Clbit` instances, and returns a valid :obj:`.Clbit` or
        :obj:`.ClassicalRegister` condition resource.
    """

    clbit_flat = tuple(clbit for creg in cregs for clbit in creg)
    clbit_set = frozenset(clbit_flat)
    creg_set = frozenset(cregs)

    def requester(classical: Clbit | ClassicalRegister | int) -> ClassicalRegister | Clbit:
        if isinstance(classical, Clbit):
            if classical not in clbit_set:
                raise CircuitError(
                    f"Condition bit {classical} is not in the registers known here: {creg_set}"
                )
            return classical
        if isinstance(classical, ClassicalRegister):
            if classical not in creg_set:
                raise CircuitError(
                    f"Condition register {classical} is not one of the registers known here:"
                    f" {creg_set}"
                )
            return classical
        if isinstance(classical, int):
            try:
                return clbit_flat[classical]
            except IndexError:
                raise CircuitError(f"Bit index {classical} is out-of-range.") from None
        raise CircuitError(
            "Invalid classical condition. Must be an int, Clbit or ClassicalRegister, but received"
            f" '{classical}'."
        )

    return requester


class InstructionSet:
    """Instruction collection, and their contexts."""

    __slots__ = ("_instructions", "_requester")

    @deprecate_arg(
        "circuit_cregs",
        since="0.19.0",
        additional_msg=(
            "Instead, pass a complete resource requester with the 'resource_requester' argument. "
            "The classical registers are insufficient to access all classical resources in a "
            "circuit, as there may be loose classical bits as well. It can also cause integer "
            "indices to be resolved incorrectly if any registers overlap."
        ),
    )
    def __init__(  # pylint: disable=bad-docstring-quotes
        self,
        circuit_cregs: list[ClassicalRegister] | None = None,
        *,
        resource_requester: Callable[..., ClassicalRegister | Clbit] | None = None,
    ):
        """New collection of instructions.

        The context (``qargs`` and ``cargs`` that each instruction is attached to) is also stored
        separately for each instruction.

        Args:
            circuit_cregs (list[ClassicalRegister]): Optional. List of ``cregs`` of the
                circuit to which the instruction is added. Default: `None`.
            resource_requester: A callable that takes in the classical resource used in the
                condition, verifies that it is present in the attached circuit, resolves any indices
                into concrete :obj:`.Clbit` instances, and returns the concrete resource.  If this
                is not given, specifying a condition with an index is forbidden, and all concrete
                :obj:`.Clbit` and :obj:`.ClassicalRegister` resources will be assumed to be valid.

                .. note::

                    The callback ``resource_requester`` is called once for each call to
                    :meth:`.c_if`, and assumes that a call implies that the resource will now be
                    used.  It may throw an error if the resource is not valid for usage.

        Raises:
            CircuitError: if both ``resource_requester`` and ``circuit_cregs`` are passed.  Only one
                of these may be passed, and it should be ``resource_requester``.
        """
        self._instructions: list[CircuitInstruction] = []
        if circuit_cregs is not None:
            if resource_requester is not None:
                raise CircuitError("Cannot pass both 'circuit_cregs' and 'resource_requester'.")
            self._requester: Callable[..., ClassicalRegister | Clbit] = _requester_from_cregs(
                tuple(circuit_cregs)
            )
        else:
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
