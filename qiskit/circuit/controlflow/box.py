# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simple box basic block."""

from __future__ import annotations

import typing

from qiskit.circuit.exceptions import CircuitError
from .control_flow import ControlFlowOp

if typing.TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class BoxOp(ControlFlowOp):
    """A scoped "box" of operations on a circuit that are treated atomically in the greater context.

    A "box" is a control-flow construct that is entered unconditionally.  The contents of the box
    behave somewhat as if the start and end of the box were barriers, except it is permissible to
    commute operations "all the way" through the box.  The box is also an explicit scope for the
    purposes of variables, stretches and compiler passes.

    Typically you create this by using the builder-interface form of :meth:`.QuantumCircuit.box`.
    """

    def __init__(
        self,
        body: QuantumCircuit,
        duration: None = None,
        unit: typing.Literal["dt", "s", "ms", "us", "ns", "ps"] = "dt",
        label: str | None = None,
    ):
        """
        Default constructor of :class:`BoxOp`.

        Args:
            body: the circuit to use as the body of the box.  This should explicit close over any
                :class:`.expr.Var` variables that must be incident from the outer circuit.  The
                expected number of qubit and clbits for the resulting instruction are inferred from
                the number in the circuit, even if they are idle.
            duration: an optional duration for the box as a whole.
            unit: the unit of the ``duration``.
            label: an optional string label for the instruction.
        """
        super().__init__("box", body.num_qubits, body.num_clbits, [body], label=label)
        self.duration = duration
        self.unit = unit

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        (body,) = parameters

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "BoxOp expects a body parameter of type "
                f"QuantumCircuit, but received {type(body)}."
            )

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the BoxOp. "
                f"BoxOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        self._params = [body]

    @property
    def body(self):
        """The ``body`` :class:`.QuantumCircuit` of the operation.

        This is the same as object returned as the sole entry in :meth:`params` and :meth:`blocks`.
        """
        # Not settable via this property; the only meaningful way to replace a body is via
        # larger `QuantumCircuit` methods, or using `replace_blocks`.
        return self.params[0]

    @property
    def blocks(self):
        return (self._params[0],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return BoxOp(body, duration=self.duration, unit=self.unit, label=self.label)

    def __eq__(self, other):
        return (
            isinstance(other, BoxOp)
            and self.duration == other.duration
            and self.unit == other.unit
            and super().__eq__(other)
        )


class BoxContext:
    """Context-manager that powers :meth:`.QuantumCircuit.box`.

    This is not part of the public interface, and should not be instantiated by users.
    """

    __slots__ = ("_circuit", "_duration", "_unit", "_label")

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        duration: None = None,
        unit: typing.Literal["dt", "s", "ms", "us", "ns", "ps"] = "dt",
        label: str | None = None,
    ):
        """
        Args:
            circuit: the outermost scope of the circuit under construction.
            duration: the final duration of the box.
            unit: the unit of ``duration``.
            label: an optional label for the box.
        """
        self._circuit = circuit
        self._duration = duration
        self._unit = unit
        self._label = label

    def __enter__(self):
        # For a box to have the semantics of internal qubit alignment with a resolvable duration, we
        # can't allow conditional jumps to exit it.  Technically an unconditional `break` or
        # `continue` could work, but we're not getting into that.
        self._circuit._push_scope(allow_jumps=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        # Boxes do not need to pass any further resources in, because there's no jumps out of a
        # `box` permitted.
        body = scope.build(scope.qubits(), scope.clbits())
        self._circuit.append(
            BoxOp(body, duration=self._duration, unit=self._unit, label=self._label),
            body.qubits,
            body.clbits,
        )
        return False
