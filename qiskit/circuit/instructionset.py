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

import functools
import warnings
from typing import Callable, Optional, Tuple, Union

from qiskit.circuit.exceptions import CircuitError
from .instruction import Instruction
from .classicalregister import Clbit, ClassicalRegister


# ClassicalRegister is hashable, and generally the registers in a circuit are completely fixed after
# its creation, so caching this allows us to only pay the register-unrolling penalty once.  The
# cache does not need to be large, because in general only one circuit is constructed at once.
@functools.lru_cache(4)
def _requester_from_cregs(cregs: Tuple[ClassicalRegister]) -> Callable:
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

    def requester(classical):
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

    __slots__ = ("instructions", "qargs", "cargs", "_requester")

    def __init__(self, circuit_cregs=None, *, resource_requester: Optional[Callable] = None):
        """New collection of instructions.

        The context (qargs and cargs that each instruction is attached to) is also stored separately
        for each instruction.

        Args:
            circuit_cregs (list[ClassicalRegister]): Optional. List of cregs of the
                circuit to which the instruction is added. Default: `None`.

                .. deprecated:: qiskit-terra 0.19
                    The classical registers are insufficient to access all classical resources in a
                    circuit, as there may be loose classical bits as well.  It can also cause
                    integer indices to be resolved incorrectly if any registers overlap.  Instead,
                    pass a complete requester to the ``resource_requester`` argument.

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
        self.instructions = []
        self.qargs = []
        self.cargs = []
        if circuit_cregs is not None:
            if resource_requester is not None:
                raise CircuitError("Cannot pass both 'circuit_cregs' and 'resource_requester'.")
            warnings.warn(
                "The 'circuit_cregs' argument to 'InstructionSet' is deprecated as of"
                " qiskit-terra 0.19, and will be removed no sooner than 3 months after its release."
                " Pass a complete resource requester as the 'resource_requester' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._requester: Optional[Callable] = _requester_from_cregs(tuple(circuit_cregs))
        else:
            self._requester = resource_requester

    def __len__(self):
        """Return number of instructions in set"""
        return len(self.instructions)

    def __getitem__(self, i):
        """Return instruction at index"""
        return self.instructions[i]

    def add(self, gate, qargs, cargs):
        """Add an instruction and its context (where it is attached)."""
        if not isinstance(gate, Instruction):
            raise CircuitError("attempt to add non-Instruction" + " to InstructionSet")
        self.instructions.append(gate)
        self.qargs.append(qargs)
        self.cargs.append(cargs)

    def inverse(self):
        """Invert all instructions."""
        for index, instruction in enumerate(self.instructions):
            self.instructions[index] = instruction.inverse()
        return self

    def c_if(self, classical: Union[Clbit, ClassicalRegister, int], val: int) -> "InstructionSet":
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
        """
        if self._requester is None and not isinstance(classical, (Clbit, ClassicalRegister)):
            raise CircuitError(
                "Cannot pass an index as a condition variable without specifying a requester"
                " when creating this InstructionSet."
            )
        if self._requester is not None:
            classical = self._requester(classical)
        for gate in self.instructions:
            gate.c_if(classical, val)
        return self
