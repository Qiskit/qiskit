# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Writing a quantum program as ordinary Python arithmetic.

Declaring an input returns a :class:`Tracer`, and every operation on a tracer returns another one.
An operation builds a Python expression, so no program exists while one is being written. Each
operation asks Rust for the types it produces from the types it is given, which is what reports a
dtype or shape mistake at the line that made it and makes every intermediate's type available at
once.

:func:`build` turns the expressions reachable from the declared outputs into a program. It walks
them memoised by identity, so a value used twice becomes one instruction and a value that reaches no
output is never built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit._accelerate.quantum_program import (
    DataTree,
    FunctionBuilder,
)

from . import ops

__all__ = [
    "Tracer",
    "add",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "build",
    "cast",
    "constant",
    "divide",
    "mean",
    "multiply",
    "parity",
    "power",
    "qp_input",
    "remainder",
    "std",
    "subtract",
    "var",
]


class _Instruction:
    """One instruction of a program being written: the values it consumes, the type it produces."""

    __slots__ = ("operands", "type_")

    def __init__(self, operands: tuple[Tracer, ...], type_: TensorType) -> None:
        self.operands = operands
        self.type_ = type_


class _Input(_Instruction):
    """A declared program input, named for the keyword argument that will supply it."""

    __slots__ = ("name",)

    def __init__(self, name: str, type_: TensorType) -> None:
        super().__init__((), type_)
        self.name = name

    def __repr__(self) -> str:
        return f"input {self.name!r}"


class _Op(_Instruction):
    """An operation applied to operands, typed as it is created."""

    __slots__ = ("op",)

    def __init__(self, op: ops.ProgramOp, operands: tuple[Tracer, ...]) -> None:
        (type_,) = op.output_types([operand.type for operand in operands])
        super().__init__(operands, type_)
        self.op = op

    def __repr__(self) -> str:
        return self.op.full_name


class Tracer:
    """One value of a quantum program being written.

    :func:`qp_input` declares an input and returns a tracer for it, and every operation on a tracer
    returns another one. Nothing is built while an expression is written; :func:`build` does that.
    Tracers are not constructed directly.

    The arithmetic and bitwise operators each build an operation, and each has a function of the
    same name, so ``x + y`` and :func:`add(x, y) <add>` are the same thing. An operand that is not a
    tracer becomes a :func:`constant`. Operations promote dtypes and broadcast shapes themselves, so
    a program holds no conversions beyond the ones its author wrote.

    A tracer compares equal only to itself: the catalogue has no comparison operation, and
    :func:`build` relies on identity to collapse a value used twice into one instruction.
    """

    __slots__ = ("_instruction",)

    # NumPy defers to the reflected operators here rather than broadcasting a tracer into an array
    # of objects.
    __array_ufunc__ = None

    def __init__(self, instruction: _Instruction) -> None:
        self._instruction = instruction

    @property
    def type(self) -> TensorType:
        """The type of this value, inferred where it was written."""
        return self._instruction.type_

    @property
    def dtype(self) -> DType:
        """The dtype of this value."""
        return self._instruction.type_.dtype

    @property
    def shape(self) -> tuple[int | bounded, ...]:
        """The shape of this value."""
        return self._instruction.type_.shape

    def __add__(self, other: Operand) -> Tracer:
        return add(self, other)

    def __radd__(self, other: Operand) -> Tracer:
        return add(other, self)

    def __sub__(self, other: Operand) -> Tracer:
        return subtract(self, other)

    def __rsub__(self, other: Operand) -> Tracer:
        return subtract(other, self)

    def __mul__(self, other: Operand) -> Tracer:
        return multiply(self, other)

    def __rmul__(self, other: Operand) -> Tracer:
        return multiply(other, self)

    def __truediv__(self, other: Operand) -> Tracer:
        return divide(self, other)

    def __rtruediv__(self, other: Operand) -> Tracer:
        return divide(other, self)

    def __mod__(self, other: Operand) -> Tracer:
        return remainder(self, other)

    def __rmod__(self, other: Operand) -> Tracer:
        return remainder(other, self)

    def __pow__(self, other: Operand) -> Tracer:
        return power(self, other)

    def __rpow__(self, other: Operand) -> Tracer:
        return power(other, self)

    def __and__(self, other: Operand) -> Tracer:
        return bitwise_and(self, other)

    def __rand__(self, other: Operand) -> Tracer:
        return bitwise_and(other, self)

    def __or__(self, other: Operand) -> Tracer:
        return bitwise_or(self, other)

    def __ror__(self, other: Operand) -> Tracer:
        return bitwise_or(other, self)

    def __xor__(self, other: Operand) -> Tracer:
        return bitwise_xor(self, other)

    def __rxor__(self, other: Operand) -> Tracer:
        return bitwise_xor(other, self)

    def __invert__(self) -> Tracer:
        return bitwise_not(self)

    def mean(self, axis: int) -> Tracer:
        """Average along ``axis``, removing it. Equivalent to :func:`mean(self, axis) <mean>`.

        Args:
            axis: The axis to average along.

        Returns:
            The average, with ``axis`` removed.
        """
        return mean(self, axis)

    def var(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Compute the variance along ``axis``, removing it.

        Equivalent to :func:`var(self, axis, ddof) <var>`.

        Args:
            axis: The axis to take the variance along.
            ddof: The delta degrees of freedom, subtracted from the divisor.

        Returns:
            The variance, with ``axis`` removed.
        """
        return var(self, axis, ddof)

    def std(self, axis: int, ddof: float = 0.0) -> Tracer:
        """Compute the standard deviation along ``axis``, removing it.

        Equivalent to :func:`std(self, axis, ddof) <std>`.

        Args:
            axis: The axis to take the standard deviation along.
            ddof: The delta degrees of freedom, subtracted from the divisor.

        Returns:
            The standard deviation, with ``axis`` removed.
        """
        return std(self, axis, ddof)

    def parity(self, axis: int) -> Tracer:
        """Compute the parity along ``axis``, removing it.

        Equivalent to :func:`parity(self, axis) <parity>`.

        Args:
            axis: The axis to XOR-reduce along.

        Returns:
            The parity of the bits along ``axis``, with that axis removed.
        """
        return parity(self, axis)

    def __repr__(self) -> str:
        return f"Tracer({self._instruction!r}, {self.type})"


def qp_input(name: str, type_: TensorType) -> Tracer:
    """Declare an input of type ``type_``, supplied by the keyword argument ``name``.

    Args:
        name: The keyword argument that will supply this input.
        type_: The dtype and shape the program demands of it.

    Returns:
        A tracer for the input. Using it in several expressions still declares one input.
    """
    return Tracer(_Input(name, type_))


def constant(value: ArrayLike) -> Tracer:
    """Return a value the program holds, rather than one supplied at call time.

    Args:
        value: The value to hold, read with :func:`numpy.asarray`.

    Returns:
        A tracer whose dtype and shape are those of the resulting array.
    """
    return Tracer(_Op(ops.Constant(value), ()))


def add(x: Operand, y: Operand) -> Tracer:
    """Add ``x`` and ``y`` elementwise. Equivalent to ``x + y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise sum.
    """
    return _op(ops.Add(), x, y)


def subtract(x: Operand, y: Operand) -> Tracer:
    """Subtract ``y`` from ``x`` elementwise. Equivalent to ``x - y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise difference.
    """
    return _op(ops.Subtract(), x, y)


def multiply(x: Operand, y: Operand) -> Tracer:
    """Multiply ``x`` and ``y`` elementwise. Equivalent to ``x * y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise product.
    """
    return _op(ops.Multiply(), x, y)


def divide(x: Operand, y: Operand) -> Tracer:
    """Divide ``x`` by ``y`` elementwise. Equivalent to ``x / y``.

    A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.

    Args:
        x: The dividend.
        y: The divisor.

    Returns:
        The elementwise quotient.
    """
    return _op(ops.Divide(), x, y)


def remainder(x: Operand, y: Operand) -> Tracer:
    """Take the remainder of ``x`` divided by ``y``, elementwise. Equivalent to ``x % y``.

    A zero divisor gives a non-finite value in a float dtype, and zero in an integer one.

    Args:
        x: The dividend.
        y: The divisor.

    Returns:
        The elementwise remainder.
    """
    return _op(ops.Remainder(), x, y)


def power(x: Operand, y: Operand) -> Tracer:
    """Raise ``x`` to the power ``y`` elementwise. Equivalent to ``x ** y``.

    Args:
        x: The base.
        y: The exponent.

    Returns:
        The elementwise power.
    """
    return _op(ops.Power(), x, y)


def bitwise_and(x: Operand, y: Operand) -> Tracer:
    """Bitwise AND of ``x`` and ``y``, elementwise. Equivalent to ``x & y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise AND.
    """
    return _op(ops.BitwiseAnd(), x, y)


def bitwise_or(x: Operand, y: Operand) -> Tracer:
    """Bitwise OR of ``x`` and ``y``, elementwise. Equivalent to ``x | y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise OR.
    """
    return _op(ops.BitwiseOr(), x, y)


def bitwise_xor(x: Operand, y: Operand) -> Tracer:
    """Bitwise XOR of ``x`` and ``y``, elementwise. Equivalent to ``x ^ y``.

    Args:
        x: The left operand.
        y: The right operand.

    Returns:
        The elementwise exclusive OR.
    """
    return _op(ops.BitwiseXor(), x, y)


def bitwise_not(x: Operand) -> Tracer:
    """Bitwise NOT of ``x``, elementwise. Equivalent to ``~x``.

    Args:
        x: The bits to invert.

    Returns:
        The elementwise NOT.
    """
    return _op(ops.BitwiseNot(), x)


def mean(x: Operand, axis: int) -> Tracer:
    """Average ``x`` along ``axis``, removing that axis.

    Args:
        x: The value to reduce.
        axis: The axis to average along.

    Returns:
        The average, with ``axis`` removed. Averaging bits gives a float.
    """
    return _op(ops.Mean(axis), x)


def var(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Compute the variance of ``x`` along ``axis``, removing that axis.

    The sum of squared deviations is divided by ``n - ddof``, where ``n`` is the length of ``axis``.

    Args:
        x: The value to reduce.
        axis: The axis to take the variance along.
        ddof: The delta degrees of freedom, subtracted from the divisor.

    Returns:
        The variance, with ``axis`` removed.
    """
    return _op(ops.Variance(axis, ddof), x)


def std(x: Operand, axis: int, ddof: float = 0.0) -> Tracer:
    """Compute the standard deviation of ``x`` along ``axis``, removing that axis.

    This is the square root of :func:`var`, and ``ddof`` means the same thing.

    Args:
        x: The value to reduce.
        axis: The axis to take the standard deviation along.
        ddof: The delta degrees of freedom, subtracted from the divisor.

    Returns:
        The standard deviation, with ``axis`` removed.
    """
    return _op(ops.Std(axis, ddof), x)


def parity(x: Operand, axis: int) -> Tracer:
    """XOR-reduce the bits of ``x`` along ``axis``, removing that axis.

    Args:
        x: The bits to reduce.
        axis: The axis to XOR-reduce along.

    Returns:
        The parity of the bits along ``axis``, with that axis removed.
    """
    return _op(ops.Parity(axis), x)


def cast(x: Operand, dtype: DType) -> Tracer:
    """Convert x to dtype, keeping its shape.

    Args:
        x: The value to cast.
        dtype: The dtype to cast to. A complex value cannot be cast to a real dtype.

    Returns:
        The value in ``dtype``.
    """
    return _op(ops.Cast(dtype), x)


def broadcast_to(x: Operand, shape: Sequence[int | bounded]) -> Tracer:
    """Broadcast ``x`` to ``shape``, aligning its axes with the trailing axes of ``shape``.

    Args:
        x: The value to broadcast.
        shape: The shape to reach. Each axis of ``x`` must either match the axis it aligns with
            or have size one.

    Returns:
        A value of shape ``shape``.
    """
    return _op(ops.BroadcastTo(shape), x)


def build(outputs: Any) -> QuantumProgram:
    """Build the program producing ``outputs``.

    outputs can be a tracer, or any nesting of sequences and mappings of them,
    and that nesting becomes the program's output structure.

    For example::

        x = qp_input("x", f64[3])
        program = build({"mean": x.mean(0), "shifted": x - 1.0})

        program(x=[1.0, 2.0, 3.0])
        # DataTree([mean: array(2.), shifted: array([0., 1., 2.])])

    Args:
        outputs: One tracer, or any nesting of sequences and mappings of them.

    Returns:
        The program, declaring the inputs and outputs the walk found.

    Raises:
        TypeError: If an output is not a tracer.
        ValueError: If two inputs are declared under one name, or a name cannot address a value.
    """
    outputs = DataTree(outputs)
    function = FunctionBuilder()
    # Keyed by `id`, which identifies an instruction only while it is alive. Every instruction
    # reachable from `outputs` is, since each one holds the tracers it consumes.
    values: dict[int, Value] = {}
    inputs: dict[str, _Input] = {}

    def materialize(instruction: _Instruction) -> Value:
        """The value ``instruction`` produces, adding it and all it consumes to the function."""
        stack = [instruction]
        while stack:
            next_instruction = stack[-1]
            if id(next_instruction) in values:
                stack.pop()
                continue
            unbuilt = [
                operand._instruction
                for operand in next_instruction.operands
                if id(operand._instruction) not in values
            ]
            if unbuilt:
                # Reversed, so that the stack pops the operands left to right and a program
                # declares its inputs in the order they were written.
                stack.extend(reversed(unbuilt))
                continue
            stack.pop()
            if isinstance(next_instruction, _Input):
                if (
                    inputs.setdefault(next_instruction.name, next_instruction)
                    is not next_instruction
                ):
                    raise ValueError(f"input {next_instruction.name!r} is declared twice")
                values[id(next_instruction)] = function.add_parameter(next_instruction.type_)
            else:
                operands = [
                    values[id(operand._instruction)] for operand in next_instruction.operands
                ]
                (values[id(next_instruction)],) = function.add_op(next_instruction.op, operands)
        return values[id(instruction)]

    for path, leaf in _leaves(outputs):
        if not isinstance(leaf, Tracer):
            where = f"output {path!r}" if path else "the output"
            raise TypeError(f"{where} is a {type(leaf).__name__}, not a Tracer")
        function.add_result(materialize(leaf._instruction))
    return function.seal(DataTree(dict.fromkeys(inputs)), outputs)


def _op(op: ops.ProgramOp, *operands: Operand) -> Tracer:
    """Apply ``op`` to ``operands``, wrapping any that are not tracers as constants."""
    return Tracer(_Op(op, tuple(_as_tracer(operand) for operand in operands)))


def _as_tracer(value: Operand) -> Tracer:
    """``value`` itself if it is a tracer, and a constant holding it otherwise."""
    return value if isinstance(value, Tracer) else constant(value)


def _leaves(tree: DataTree, path: str = "") -> Iterator[tuple[str, Any]]:
    """Return every leaf of ``tree`` in order, each with the dotted path that addresses it."""
    if tree.is_leaf:
        yield path, tree.leaf
        return
    children = tree.items() if tree.is_mapping else enumerate(tree)
    for key, child in children:
        below = f"{path}.{key}" if path else str(key)
        if isinstance(child, DataTree):
            yield from _leaves(child, below)
        else:
            yield below, child


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import ArrayLike

    from qiskit._accelerate.quantum_program import DType, QuantumProgram, TensorType, Value, bounded

    # A tracer, or anything `numpy.asarray` reads, which becomes a constant.
    Operand = Tracer | ArrayLike
