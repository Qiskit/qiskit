# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit operation representing a ``for`` loop."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from collections.abc import Iterable, Mapping

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import ControlFlowType
from qiskit.circuit.classical.expr import Expr, Range, Var
from .control_flow import ControlFlowOp


if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def _validate_for_loop_params(indexset, loop_parameter, body) -> None:
    """Enforce the compile-time vs runtime boundary for ``loop_parameter`` vs ``indexset``."""
    if isinstance(indexset, Range):
        if isinstance(loop_parameter, Parameter):
            raise CircuitError(
                "Cannot use a compile-time Parameter as a loop variable with an expr.Range "
                "indexset. Use an expr.Var instead, or use a Python range/integer list for "
                "gate-parameter unrolling."
            )
    elif isinstance(loop_parameter, Var):
        raise CircuitError(
            "Cannot use an expr.Var as a loop variable with a Python range or integer list "
            "indexset. Use a Parameter instead, or use an expr.Range for runtime execution."
        )


class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    The ``indexset`` selects the loop semantics:

    * A Python :class:`range` or an iterable of integers is a compile-time index
      set.  The ``loop_parameter`` must be a :class:`~.Parameter` (or ``None``),
      and the loop can be unrolled by :class:`~qiskit.transpiler.passes.UnrollForLoops`
      via ``body.assign_parameters``.
    * A classical :class:`~.expr.Range` (constant or dynamic) is a real-time index
      set.  The ``loop_parameter`` must be an :class:`~.expr.Var` (or ``None``).
      A constant :class:`~.expr.Range` paired with an :class:`~.expr.Var` is still
      unrollable: :class:`~qiskit.transpiler.passes.UnrollForLoops` materializes
      each iteration value and substitutes the :class:`~.expr.Var` in every
      classical expression in the body.

    Mixing a :class:`~.Parameter` with an :class:`~.expr.Range`, or an
    :class:`~.expr.Var` with a Python :class:`range`/integer list, raises
    :class:`~.CircuitError`.
    """

    _control_flow_type = ControlFlowType.ForLoop

    def __init__(
        self,
        indexset: Iterable[int] | Range,
        loop_parameter: Parameter | Var | None,
        body: QuantumCircuit,
        label: str | None = None,
    ):
        """
        Args:
            indexset: A collection of integers to loop over, as a Python
                :class:`range`, an iterable of integers, or a classical
                :class:`~.expr.Range`.
            loop_parameter: The placeholder bound to each ``indexset`` value
                inside ``body``. For a Python ``range``/integer list, must be a
                :class:`~.Parameter` or ``None``. For an :class:`~.expr.Range`,
                must be an :class:`~.expr.Var` or ``None``. ``None`` simply
                repeats the body without binding any variable.
            body: The loop body to be repeatedly executed.
            label: An optional label for identifying the instruction.

        Raises:
            CircuitError: if ``loop_parameter``'s type is incompatible with the
                ``indexset`` type.
        """
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits
        super().__init__(
            "for_loop", num_qubits, num_clbits, [indexset, loop_parameter, body], label=label
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):

        from qiskit.circuit import QuantumCircuit

        indexset, loop_parameter, body = parameters

        if not isinstance(loop_parameter, (Parameter, Var, type(None))):
            raise CircuitError(
                "ForLoopOp expects a loop_parameter parameter to "
                "be either of type Parameter, expr.Var, or None, but received "
                f"{type(loop_parameter)}."
            )

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "ForLoopOp expects a body parameter to be of type "
                f"QuantumCircuit, but received {type(body)}."
            )

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the ForLoopOp. "
                f"ForLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        if (
            isinstance(loop_parameter, Parameter)
            and loop_parameter not in body.parameters
            and loop_parameter.name in (p.name for p in body.parameters)
        ):
            warnings.warn(
                "The Parameter provided as a loop_parameter was not found "
                "on the loop body and so no binding of the indexset to loop "
                "parameter will occur. A different Parameter of the same name "
                f"({loop_parameter.name}) was found. If you intended to loop "
                "over that Parameter, please use that Parameter instance as "
                "the loop_parameter.",
                stacklevel=2,
            )

        _validate_for_loop_params(indexset, loop_parameter, body)

        # Consume indexset into a tuple unless it was provided as a range.
        # Preserve ranges so that they can be exported as OpenQASM 3 ranges.
        indexset = indexset if isinstance(indexset, (range, Range)) else tuple(indexset)

        self._params = [indexset, loop_parameter, body]

    @property
    def blocks(self):
        return (self._params[2],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return ForLoopOp(self.params[0], self.params[1], body, label=self.label)

    def substitute(self, substitutions: Mapping[Var, Expr]) -> ForLoopOp:
        """Return a new :class:`ForLoopOp` with classical :class:`~.expr.Var` nodes replaced.

        The substitution is applied to the ``indexset`` when it is an :class:`~.expr.Range`
        (a Python :class:`range`/integer-list indexset has no variables and is left unchanged)
        and recursively to the loop body via :meth:`.QuantumCircuit.substitute_vars`.  The
        ``loop_parameter`` itself is left untouched: it is bound by the loop rather than
        substituted away.

        Args:
            substitutions: mapping from :class:`~.expr.Var` to the replacement
                :class:`~.expr.Expr`.

        Returns:
            A new :class:`ForLoopOp` with the substitutions applied.
        """
        indexset, loop_parameter, body = self.params
        if isinstance(indexset, Range):
            indexset = indexset.substitute(substitutions)
        return ForLoopOp(
            indexset,
            loop_parameter,
            body.substitute_vars(substitutions, strict=False),
            label=self.label,
        )


class ForLoopContext:
    """A context manager for building up ``for`` loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example, and a loop variable will be allocated for
    you, if you do not supply one yourself.  The type of the auto-generated loop variable follows
    the type of the ``indexset``:

    * Python :class:`range` / integer list → compile-time :class:`~.Parameter`.
    * Classical :class:`~.expr.Range` → real-time :class:`~.expr.Var` of the Range's type.

    In both cases, an auto-generated loop variable that the body never references is dropped
    (the resulting :class:`ForLoopOp` will have ``loop_parameter=None``).

    Mixing a :class:`~.Parameter` with an :class:`~.expr.Range`, or an :class:`~.expr.Var` with a
    Python :class:`range`/integer list, raises :class:`~.CircuitError`; see :class:`ForLoopOp`.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.for_loop` in its context-manager form, i.e. by not supplying a ``body`` or
    sets of qubits and clbits.

    Example usage::

        import math
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)) as i:
            qc.rx(i * math.pi/4, 0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()

    This context should almost invariably be created by a :meth:`.QuantumCircuit.for_loop` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.ForLoopOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    # Class-level variable keep track of the number of auto-generated loop variables, so we don't
    # get naming clashes.
    _generated_loop_parameters = 0

    __slots__ = (
        "_circuit",
        "_generate_loop_parameter",
        "_indexset",
        "_label",
        "_loop_parameter",
        "_used",
    )

    def __init__(
        self,
        circuit: QuantumCircuit,
        indexset: Iterable[int] | Range,
        loop_parameter: Parameter | Var | None = None,
        *,
        label: str | None = None,
    ):
        self._circuit = circuit
        self._generate_loop_parameter = loop_parameter is None
        self._loop_parameter = loop_parameter
        # We can pass through `range` instances because OpenQASM 3 has native support for this type
        # of iterator set.
        self._indexset = indexset if isinstance(indexset, (range, Range)) else tuple(indexset)
        self._label = label
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("A for-loop context manager cannot be re-entered.")
        self._used = True
        self._circuit._push_scope()
        scope = self._circuit._current_scope()
        if self._generate_loop_parameter:
            # Auto-generate a loop variable whose type follows the indexset; an
            # auto-generated variable that the body never references is dropped in __exit__.
            self._loop_parameter = self._make_loop_variable()
            type(self)._generated_loop_parameters += 1
            self._register_loop_variable_in_scope(scope, lazy=True)
        else:
            # Explicit user-supplied loop variable: register eagerly so it is always kept,
            # mirroring how an explicit Parameter is kept even if unused.
            self._register_loop_variable_in_scope(scope, lazy=False)
        return self._loop_parameter

    def _make_loop_variable(self):
        """Construct an auto-generated loop variable matched to the indexset type."""
        name = f"_loop_i_{self._generated_loop_parameters}"
        if isinstance(self._indexset, Range):
            return Var.new(name, self._indexset.type)
        return Parameter(name)

    def _register_loop_variable_in_scope(self, scope, *, lazy: bool) -> None:
        """Make ``self._loop_parameter`` resolvable inside the body scope, if it needs to be.

        A :class:`~.Parameter` is automatically tracked through gate parameter expressions and
        needs no scope registration. A real-time :class:`~.expr.Var` is registered in the
        classical-variable scope so :meth:`._BuildScope.use_var` resolves it inside the body:

        * ``lazy=True``: defer until first use; an auto-generated Var that the user never
          references is dropped, matching the existing :class:`~.Parameter` behavior.
        * ``lazy=False``: register eagerly as a declared local, matching an explicit user-supplied
          Var (which is kept even if unused, again matching the explicit-Parameter case).
        """
        if isinstance(self._loop_parameter, Var):
            if lazy:
                scope.add_pending_loop_var(self._loop_parameter)
            else:
                scope.add_uninitialized_var(self._loop_parameter)

    def _is_loop_variable_referenced(self, body) -> bool:
        """Return whether ``self._loop_parameter`` is referenced inside ``body``.

        A single boundary helper between the two kinds of loop variable Qiskit supports:
        a compile-time :class:`~.Parameter` appears in :attr:`.QuantumCircuit.parameters`,
        a real-time :class:`~.expr.Var` appears as a declared/input variable of the body.
        The rest of the for-loop machinery treats ``loop_parameter`` uniformly.
        """
        if self._loop_parameter is None:
            return False
        if isinstance(self._loop_parameter, Var):
            return body.has_var(self._loop_parameter)
        return self._loop_parameter in body.parameters

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        # Loops do not need to pass any further resources in, because this scope itself defines the
        # extent of ``break`` and ``continue`` statements.
        body = scope.build(scope.qubits(), scope.clbits())
        # We always bind the loop variable if the user supplied it explicitly, even if unused.
        # For an auto-generated loop variable, we only keep it if the user actually referenced it
        # inside the body, to avoid leaving stray dangling resources.
        if self._generate_loop_parameter and not self._is_loop_variable_referenced(body):
            loop_parameter = None
        else:
            loop_parameter = self._loop_parameter
        self._circuit.append(
            ForLoopOp(self._indexset, loop_parameter, body, label=self._label),
            tuple(body.qubits),
            tuple(body.clbits),
        )
        return False
