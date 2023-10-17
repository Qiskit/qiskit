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

"""Builder types for the basic control-flow constructs."""

# This file is in circuit.controlflow rather than the root of circuit because the constructs here
# are only intended to be localised to constructing the control flow instructions.  We anticipate
# having a far more complete builder of all circuits, with more classical control and creation, in
# the future.


import abc
import itertools
import typing
from typing import Callable, Collection, Iterable, List, FrozenSet, Tuple, Union, Optional

from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.circuit.register import Register

from ._builder_utils import condition_resources, node_resources

if typing.TYPE_CHECKING:
    import qiskit  # pylint: disable=cyclic-import


class InstructionResources(typing.NamedTuple):
    """The quantum and classical resources used within a particular instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.

    Attributes:
        qubits: A collection of qubits that will be used by the instruction.
        clbits: A collection of clbits that will be used by the instruction.
        qregs: A collection of quantum registers that are used by the instruction.
        cregs: A collection of classical registers that are used by the instruction.
    """

    qubits: Collection[Qubit] = ()
    clbits: Collection[Clbit] = ()
    qregs: Collection[QuantumRegister] = ()
    cregs: Collection[ClassicalRegister] = ()


class InstructionPlaceholder(Instruction, abc.ABC):
    """A fake instruction that lies about its number of qubits and clbits.

    These instances are used to temporarily represent control-flow instructions during the builder
    process, when their lengths cannot be known until the end of the block.  This is necessary to
    allow constructs like::

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, 0)

    since ``qc.break_loop()`` needs to return a (mostly) functional
    :obj:`~qiskit.circuit.Instruction` in order for :meth:`.InstructionSet.c_if` to work correctly.

    When appending a placeholder instruction into a circuit scope, you should create the
    placeholder, and then ask it what resources it should be considered as using from the start by
    calling :meth:`.InstructionPlaceholder.placeholder_instructions`.  This set will be a subset of
    the final resources it asks for, but it is used for initialising resources that *must* be
    supplied, such as the bits used in the conditions of placeholder ``if`` statements.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    _directive = True

    @abc.abstractmethod
    def concrete_instruction(
        self, qubits: FrozenSet[Qubit], clbits: FrozenSet[Clbit]
    ) -> Tuple[Instruction, InstructionResources]:
        """Get a concrete, complete instruction that is valid to act over all the given resources.

        The returned resources may not be the full width of the given resources, but will certainly
        be a subset of them; this can occur if (for example) a placeholder ``if`` statement is
        present, but does not itself contain any placeholder instructions.  For resource efficiency,
        the returned :class:`.ControlFlowOp` will not unnecessarily span all resources, but only the
        ones that it needs.

        .. note::

            The caller of this function is responsible for ensuring that the inputs to this function
            are non-strict supersets of the bits returned by :meth:`placeholder_resources`.

        Any condition added in by a call to :obj:`.Instruction.c_if` will be propagated through, but
        set properties like ``duration`` will not; it doesn't make sense for control-flow operations
        to have pulse scheduling on them.

        Args:
            qubits: The qubits the created instruction should be defined across.
            clbits: The clbits the created instruction should be defined across.

        Returns:
            A full version of the relevant control-flow instruction, and the resources that it uses.
            This is a "proper" instruction instance, as if it had been defined with the correct
            number of qubits and clbits from the beginning.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def placeholder_resources(self) -> InstructionResources:
        """Get the qubit and clbit resources that this placeholder instruction should be considered
        as using before construction.

        This will likely not include *all* resources after the block has been built, but using the
        output of this method ensures that all resources will pass through a
        :meth:`.QuantumCircuit.append` call, even if they come from a placeholder, and consequently
        will be tracked by the scope managers.

        Returns:
            A collection of the quantum and classical resources this placeholder instruction will
            certainly use.
        """
        raise NotImplementedError

    def _copy_mutable_properties(self, instruction: Instruction) -> Instruction:
        """Copy mutable properties from ourselves onto a non-placeholder instruction.

        The mutable properties are expected to be things like ``condition``, added onto a
        placeholder by the :meth:`c_if` method.  This mutates ``instruction``, and returns the same
        instance that was passed.  This is mostly intended to make writing concrete versions of
        :meth:`.concrete_instruction` easy.

        The complete list of mutations is:

        * ``condition``, added by :meth:`c_if`.

        Args:
            instruction: the concrete instruction instance to be mutated.

        Returns:
            The same instruction instance that was passed, but mutated to propagate the tracked
            changes to this class.
        """
        instruction.condition = self.condition
        return instruction

    # Provide some better error messages, just in case something goes wrong during development and
    # the placeholder type leaks out to somewhere visible.

    def assemble(self):
        raise CircuitError("Cannot assemble a placeholder instruction.")

    def qasm(self):
        raise CircuitError("Cannot convert a placeholder instruction to OpenQASM 2")

    def repeat(self, n):
        raise CircuitError("Cannot repeat a placeholder instruction.")


class ControlFlowBuilderBlock:
    """A lightweight scoped block for holding instructions within a control-flow builder context.

    This class is designed only to be used by :obj:`.QuantumCircuit` as an internal context for
    control-flow builder instructions, and in general should never be instantiated by any code other
    than that.

    Note that the instructions that are added to this scope may not be valid yet, so this elides
    some of the type-checking of :obj:`.QuantumCircuit` until those things are known.

    The general principle of the resource tracking through these builder blocks is that every
    necessary resource should pass through an :meth:`.append` call, so that at the point that
    :meth:`.build` is called, the scope knows all the concrete resources that it requires.  However,
    the scope can also contain "placeholder" instructions, which may need extra resources filling in
    from outer scopes (such as a ``break`` needing to know the width of its containing ``for``
    loop).  This means that :meth:`.build` takes all the *containing* scope's resources as well.
    This does not break the "all resources pass through an append" rule, because the containing
    scope will only begin to build its instructions once it has received them all.

    In short, :meth:`.append` adds resources, and :meth:`.build` may use only a subset of the extra
    ones passed.  This ensures that all instructions know about all the resources they need, even in
    the case of ``break``, but do not block any resources that they do *not* need.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    __slots__ = (
        "instructions",
        "qubits",
        "clbits",
        "registers",
        "global_phase",
        "_allow_jumps",
        "_resource_requester",
        "_built",
        "_forbidden_message",
    )

    def __init__(
        self,
        qubits: Iterable[Qubit],
        clbits: Iterable[Clbit],
        *,
        registers: Iterable[Register] = (),
        resource_requester: Callable,
        allow_jumps: bool = True,
        forbidden_message: Optional[str] = None,
    ):
        """
        Args:
            qubits: Any qubits this scope should consider itself as using from the beginning.
            clbits: Any clbits this scope should consider itself as using from the beginning.  Along
                with ``qubits``, this is useful for things such as ``if`` and ``while`` loop
                builders, where the classical condition has associated resources, and is known when
                this scope is created.
            registers: Any registers this scope should consider itself as using from the
                beginning.  This is useful for :obj:`.IfElseOp` and :obj:`.WhileLoopOp` instances
                which use a classical register as their condition.
            allow_jumps: Whether this builder scope should allow ``break`` and ``continue``
                statements within it.  This is intended to help give sensible error messages when
                dangerous behaviour is encountered, such as using ``break`` inside an ``if`` context
                manager that is not within a ``for`` manager.  This can only be safe if the user is
                going to place the resulting :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that
                uses *exactly* the same set of resources.  We cannot verify this from within the
                builder interface (and it is too expensive to do when the ``for`` op is made), so we
                fail safe, and require the user to use the more verbose, internal form.
            resource_requester: A callback function that takes in some classical resource specifier,
                and returns a concrete classical resource, if this scope is allowed to access that
                resource.  In almost all cases, this should be a resolver from the
                :obj:`.QuantumCircuit` that this scope is contained in.  See
                :meth:`.QuantumCircuit._resolve_classical_resource` for the normal expected input
                here, and the documentation of :obj:`.InstructionSet`, which uses this same
                callback.
            forbidden_message: If a string is given here, a :exc:`.CircuitError` will be raised on
                any attempts to append instructions to the scope with this message.  This is used by
                pseudo scopes where the state machine of the builder scopes has changed into a
                position where no instructions should be accepted, such as when inside a ``switch``
                but outside any cases.
        """
        self.instructions: List[CircuitInstruction] = []
        self.qubits = set(qubits)
        self.clbits = set(clbits)
        self.registers = set(registers)
        self.global_phase = 0.0
        self._allow_jumps = allow_jumps
        self._resource_requester = resource_requester
        self._built = False
        self._forbidden_message = forbidden_message

    @property
    def allow_jumps(self):
        """Whether this builder scope should allow ``break`` and ``continue`` statements within it.

        This is intended to help give sensible error messages when dangerous behaviour is
        encountered, such as using ``break`` inside an ``if`` context manager that is not within a
        ``for`` manager.  This can only be safe if the user is going to place the resulting
        :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that uses *exactly* the same set of
        resources.  We cannot verify this from within the builder interface (and it is too expensive
        to do when the ``for`` op is made), so we fail safe, and require the user to use the more
        verbose, internal form.
        """
        return self._allow_jumps

    def append(self, instruction: CircuitInstruction) -> CircuitInstruction:
        """Add an instruction into the scope, keeping track of the qubits and clbits that have been
        used in total."""
        if self._forbidden_message is not None:
            raise CircuitError(self._forbidden_message)

        if not self._allow_jumps:
            # pylint: disable=cyclic-import
            from .break_loop import BreakLoopOp, BreakLoopPlaceholder
            from .continue_loop import ContinueLoopOp, ContinueLoopPlaceholder

            forbidden = (BreakLoopOp, BreakLoopPlaceholder, ContinueLoopOp, ContinueLoopPlaceholder)
            if isinstance(instruction.operation, forbidden):
                raise CircuitError(
                    f"The current builder scope cannot take a '{instruction.operation.name}'"
                    " because it is not in a loop."
                )

        self.instructions.append(instruction)
        self.qubits.update(instruction.qubits)
        self.clbits.update(instruction.clbits)
        return instruction

    def request_classical_resource(self, specifier):
        """Resolve a single classical resource specifier into a concrete resource, raising an error
        if the specifier is invalid, and track it as now being used in scope.

        Args:
            specifier (Union[Clbit, ClassicalRegister, int]): a specifier of a classical resource
                present in this circuit.  An ``int`` will be resolved into a :obj:`.Clbit` using the
                same conventions that measurement operations on this circuit use.

        Returns:
            Union[Clbit, ClassicalRegister]: the requested resource, resolved into a concrete
            instance of :obj:`.Clbit` or :obj:`.ClassicalRegister`.

        Raises:
            CircuitError: if the resource is not present in this circuit, or if the integer index
                passed is out-of-bounds.
        """
        if self._built:
            raise CircuitError("Cannot add resources after the scope has been built.")
        # Allow the inner resolve to propagate exceptions.
        resource = self._resource_requester(specifier)
        if isinstance(resource, Clbit):
            self.add_bits((resource,))
        else:
            self.add_register(resource)
        return resource

    def peek(self) -> CircuitInstruction:
        """Get the value of the most recent instruction tuple in this scope."""
        if not self.instructions:
            raise CircuitError("This scope contains no instructions.")
        return self.instructions[-1]

    def pop(self) -> CircuitInstruction:
        """Get the value of the most recent instruction in this scope, and remove it from this
        object."""
        if not self.instructions:
            raise CircuitError("This scope contains no instructions.")
        return self.instructions.pop()

    def add_bits(self, bits: Iterable[Union[Qubit, Clbit]]):
        """Add extra bits to this scope that are not associated with any concrete instruction yet.

        This is useful for expanding a scope's resource width when it may contain ``break`` or
        ``continue`` statements, or when its width needs to be expanded to match another scope's
        width (as in the case of :obj:`.IfElseOp`).

        Args:
            bits: The qubits and clbits that should be added to a scope.  It is not an error if
                there are duplicates, either within the iterable or with the bits currently in
                scope.

        Raises:
            TypeError: if the provided bit is of an incorrect type.
        """
        for bit in bits:
            if isinstance(bit, Qubit):
                self.qubits.add(bit)
            elif isinstance(bit, Clbit):
                self.clbits.add(bit)
            else:
                raise TypeError(f"Can only add qubits or classical bits, but received '{bit}'.")

    def add_register(self, register: Register):
        """Add a :obj:`.Register` to the set of resources used by this block, ensuring that
        all bits contained within are also accounted for.

        Args:
            register: the register to add to the block.
        """
        if register in self.registers:
            # Fast return to avoid iterating through the bits.
            return
        self.registers.add(register)
        self.add_bits(register)

    def build(
        self, all_qubits: FrozenSet[Qubit], all_clbits: FrozenSet[Clbit]
    ) -> "qiskit.circuit.QuantumCircuit":
        """Build this scoped block into a complete :obj:`.QuantumCircuit` instance.

        This will build a circuit which contains all of the necessary qubits and clbits and no
        others.

        The ``qubits`` and ``clbits`` arguments should be sets that contains all the resources in
        the outer scope; these will be passed down to inner placeholder instructions, so they can
        apply themselves across the whole scope should they need to.  The resulting
        :obj:`.QuantumCircuit` will be defined over a (nonstrict) subset of these resources.  This
        is used to let ``break`` and ``continue`` span all resources, even if they are nested within
        several :obj:`.IfElsePlaceholder` objects, without requiring :obj:`.IfElsePlaceholder`
        objects *without* any ``break`` or ``continue`` statements to be full-width.

        Args:
            all_qubits: all the qubits in the containing scope of this block.  The block may expand
                to use some or all of these qubits, but will never gain qubits that are not in this
                set.
            all_clbits: all the clbits in the containing scope of this block.  The block may expand
                to use some or all of these clbits, but will never gain clbits that are not in this
                set.

        Returns:
            A circuit containing concrete versions of all the instructions that were in the scope,
            and using the minimal set of resources necessary to support them, within the enclosing
            scope.
        """
        from qiskit.circuit import QuantumCircuit, SwitchCaseOp

        # There's actually no real problem with building a scope more than once.  This flag is more
        # so _other_ operations, which aren't safe can be forbidden, such as mutating instructions
        # that may have been built into other objects.
        self._built = True

        if self._forbidden_message is not None:
            # Reaching this implies a logic error in the builder interface.
            raise RuntimeError("Cannot build a forbidden scope. Please report this as a bug.")

        potential_qubits = all_qubits - self.qubits
        potential_clbits = all_clbits - self.clbits

        # We start off by only giving the QuantumCircuit the qubits we _know_ it will need, and add
        # more later as needed.
        out = QuantumCircuit(
            list(self.qubits), list(self.clbits), *self.registers, global_phase=self.global_phase
        )

        for instruction in self.instructions:
            if isinstance(instruction.operation, InstructionPlaceholder):
                operation, resources = instruction.operation.concrete_instruction(
                    all_qubits, all_clbits
                )
                qubits = tuple(resources.qubits)
                clbits = tuple(resources.clbits)
                instruction = CircuitInstruction(operation, qubits, clbits)
                # We want to avoid iterating over the tuples unnecessarily if there's no chance
                # we'll need to add bits to the circuit.
                if potential_qubits and qubits:
                    add_qubits = potential_qubits.intersection(qubits)
                    if add_qubits:
                        potential_qubits -= add_qubits
                        out.add_bits(add_qubits)
                if potential_clbits and clbits:
                    add_clbits = potential_clbits.intersection(clbits)
                    if add_clbits:
                        potential_clbits -= add_clbits
                        out.add_bits(add_clbits)
                for register in itertools.chain(resources.qregs, resources.cregs):
                    if register not in self.registers:
                        # As of 2021-12-09, QuantumCircuit doesn't have an efficient way to check if
                        # a register is already present, so we use our own tracking.
                        self.add_register(register)
                        out.add_register(register)
            if getattr(instruction.operation, "condition", None) is not None:
                for register in condition_resources(instruction.operation.condition).cregs:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
            elif isinstance(instruction.operation, SwitchCaseOp):
                target = instruction.operation.target
                if isinstance(target, Clbit):
                    target_registers = ()
                elif isinstance(target, ClassicalRegister):
                    target_registers = (target,)
                else:
                    target_registers = node_resources(target).cregs
                for register in target_registers:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
            # We already did the broadcasting and checking when the first call to
            # QuantumCircuit.append happened (which the user wrote), and added the instruction into
            # this scope.  We just need to finish the job now.
            out._append(instruction)
        return out

    def copy(self) -> "ControlFlowBuilderBlock":
        """Return a semi-shallow copy of this builder block.

        The instruction lists and sets of qubits and clbits will be new instances (so mutations will
        not propagate), but any :obj:`.Instruction` instances within them will not be copied.

        Returns:
            a semi-shallow copy of this object.
        """
        out = type(self).__new__(type(self))
        out.instructions = self.instructions.copy()
        out.qubits = self.qubits.copy()
        out.clbits = self.clbits.copy()
        out.registers = self.registers.copy()
        out.global_phase = self.global_phase
        out._allow_jumps = self._allow_jumps
        out._forbidden_message = self._forbidden_message
        return out
