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

"""User-space constructor functions for the expression tree, which do some of the inference and
lifting boilerplate work."""

# pylint: disable=redefined-builtin,redefined-outer-name

from __future__ import annotations

__all__ = [
    "lift",
    "bit_not",
    "logic_not",
    "bit_and",
    "bit_or",
    "bit_xor",
    "logic_and",
    "logic_or",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "lift_legacy_condition",
]

import typing

from .expr import Expr, Var, Value, Unary, Binary, Cast, Index
from ..types import CastKind, cast_kind
from .. import types

if typing.TYPE_CHECKING:
    import qiskit


def _coerce_lossless(expr: Expr, type: types.Type) -> Expr:
    """Coerce ``expr`` to ``type`` by inserting a suitable :class:`Cast` node, if the cast is
    lossless.  Otherwise, raise a ``TypeError``."""
    kind = cast_kind(expr.type, type)
    if kind is CastKind.EQUAL:
        return expr
    if kind is CastKind.IMPLICIT:
        return Cast(expr, type, implicit=True)
    if kind is CastKind.LOSSLESS:
        return Cast(expr, type, implicit=False)
    if kind is CastKind.DANGEROUS:
        raise TypeError(f"cannot cast '{expr}' to '{type}' without loss of precision")
    raise TypeError(f"no cast is defined to take '{expr}' to '{type}'")


def lift_legacy_condition(
    condition: tuple[qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister, int], /
) -> Expr:
    """Lift a legacy two-tuple equality condition into a new-style :class:`Expr`.

    Examples:
        Taking an old-style conditional instruction and getting an :class:`Expr` from its
        condition::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.library import HGate
            from qiskit.circuit.classical import expr

            cr = ClassicalRegister(2)
            instr = HGate().c_if(cr, 3)

            lifted = expr.lift_legacy_condition(instr.condition)
    """
    from qiskit.circuit import Clbit  # pylint: disable=cyclic-import

    target, value = condition
    if isinstance(target, Clbit):
        bool_ = types.Bool()
        return Var(target, bool_) if value else Unary(Unary.Op.LOGIC_NOT, Var(target, bool_), bool_)
    left = Var(target, types.Uint(width=target.size))
    if value.bit_length() > target.size:
        left = Cast(left, types.Uint(width=value.bit_length()), implicit=True)
    right = Value(value, left.type)
    return Binary(Binary.Op.EQUAL, left, right, types.Bool())


def lift(value: typing.Any, /, type: types.Type | None = None, *, try_const: bool = True) -> Expr:
    """Lift the given Python ``value`` to a :class:`~.expr.Value` or :class:`~.expr.Var`.

    If an explicit ``type`` is given, the typing in the output will reflect that.

    By default, lifted scalars will be const if they aren't backed by a classical resource.
    To lift scalars to a non-const-typed expression, specify `try_const` as `False`.

    Examples:
        Lifting simple circuit objects to be :class:`~.expr.Var` instances::

            >>> from qiskit.circuit import Clbit, ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.lift(Clbit())
            Var(<clbit>, Bool(const=False))
            >>> expr.lift(ClassicalRegister(3, "c"))
            Var(ClassicalRegister(3, "c"), Uint(3, const=False))

        The type of the return value can be influenced, if the given value could be interpreted
        losslessly as the given type (use :func:`cast` to perform a full set of casting
        operations, include lossy ones)::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr, types
            >>> expr.lift(ClassicalRegister(3, "c"), types.Uint(5))
            Var(ClassicalRegister(3, "c"), Uint(5, const=False))
            >>> expr.lift(5, types.Uint(4))
            Value(5, Uint(4, const=True))

        Lifting non-classical resource scalars to non-const values::

            >>> from qiskit.circuit.classical import expr, types
            >>> expr.lift(7)
            Value(7, Uint(3, const=True))
            >>> expr.lift(7, try_const=False)
            Value(7, Uint(3, const=False))
    """
    if isinstance(value, Expr):
        if type is not None:
            raise ValueError("use 'cast' to cast existing expressions, not 'lift'")
        return value
    from qiskit.circuit import Clbit, ClassicalRegister  # pylint: disable=cyclic-import

    inferred: types.Type
    if value is True or value is False:
        inferred = types.Bool(const=try_const)
        constructor = Value
    elif isinstance(value, Clbit):
        inferred = types.Bool()
        constructor = Var
    elif isinstance(value, ClassicalRegister):
        inferred = types.Uint(width=value.size)
        constructor = Var
    elif isinstance(value, int):
        if value < 0:
            raise ValueError("cannot represent a negative value")
        inferred = types.Uint(width=value.bit_length() or 1, const=try_const)
        constructor = Value
    else:
        raise TypeError(f"failed to infer a type for '{value}'")
    if type is None:
        type = inferred
    if types.is_supertype(type, inferred):
        return constructor(value, type)
    raise TypeError(
        f"the explicit type '{type}' is not suitable for representing '{value}';"
        f" it must be non-strict supertype of '{inferred}'"
    )


def cast(operand: typing.Any, type: types.Type, /) -> Expr:
    """Create an explicit cast from the given value to the given type.

    This can also be used to cast-away const status.

    Examples:
        Add an explicit cast node that explicitly casts a higher precision type to a lower precision
        one::

            >>> from qiskit.circuit.classical import expr, types
            >>> value = expr.Value(5, types.Uint(32))
            >>> expr.cast(value, types.Uint(8))
            Cast(Value(5, types.Uint(32, const=False)), types.Uint(8, const=False), implicit=False)

        Cast-away const status::

            >>> from qiskit.circuit.classical import expr, types
            >>> value = expr.Value(5, types.Uint(32, const=True))
            >>> expr.cast(value, types.Uint(32))
            Cast(Value(5, types.Uint(32, const=True)), types.Uint(32, const=False), implicit=False)
    """
    operand = lift(operand)
    if cast_kind(operand.type, type) is CastKind.NONE:
        raise TypeError(f"cannot cast '{operand}' to '{type}'")
    return Cast(operand, type)


def bit_not(operand: typing.Any, /) -> Expr:
    """Create a bitwise 'not' expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.

    Examples:
        Bitwise negation of a :class:`.ClassicalRegister`::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_not(ClassicalRegister(3, "c"))
            Unary(Unary.Op.BIT_NOT, Var(ClassicalRegister(3, 'c'), Uint(3, const=False)), Uint(3, const=False))
    """
    operand = lift(operand)
    if operand.type.kind not in (types.Bool, types.Uint):
        raise TypeError(f"cannot apply '{Unary.Op.BIT_NOT}' to type '{operand.type}'")
    return Unary(Unary.Op.BIT_NOT, operand, operand.type)


def logic_not(operand: typing.Any, /) -> Expr:
    """Create a logical 'not' expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.

    Examples:
        Logical negation of a :class:`.ClassicalRegister`::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.logic_not(ClassicalRegister(3, "c"))
            Unary(\
Unary.Op.LOGIC_NOT, \
Cast(Var(ClassicalRegister(3, 'c'), Uint(3, const=False)), Bool(const=False), implicit=True), \
Bool(const=False))
    """
    var_or_value = lift(operand)
    operand = _coerce_lossless(var_or_value, types.Bool(const=var_or_value.type.const))
    return Unary(Unary.Op.LOGIC_NOT, operand, operand.type)


def _lift_binary_operands(left: typing.Any, right: typing.Any) -> tuple[Expr, Expr]:
    """Lift two binary operands simultaneously, inferring the widths of integer literals in either
    position to match the other operand.

    Const-ness is handled as follows:
      * If neither operand is an expression, both are lifted to share the same const-ness.
        Both will be const, if possible. Else, neither will be.
      * If only one operand is an expression, the other is lifted with the same const-ness, if possible.
        Otherwise, the returned operands will have different const-ness, and thus may require a cast node
        to be interoperable.
      * If both operands are expressions, they are returned as-is, and may require a cast node.
    """
    left_bool = isinstance(left, bool)
    left_int = isinstance(left, int) and not left_bool
    right_bool = isinstance(right, bool)
    right_int = isinstance(right, int) and not right_bool
    if not (left_int or right_int):
        if left_bool == right_bool:
            # If they're both bool, they'll lift as const here.
            # If neither are, we've already checked for int, so they must be bits,
            # registers, or expressions, none of which will lift to be const.
            left = lift(left, try_const=True)
            right = lift(right, try_const=True)
        elif not right_bool:
            # Left is a bool, which should only be const if right is const.
            right = lift(right)
            left = lift(left, try_const=right.type.const)
        elif not left_bool:
            # Right is a bool, which should only be const if left is const.
            left = lift(left)
            right = lift(right, try_const=left.type.const)
    elif not right_int:
        # Left is an int.
        right = lift(right)
        if right.type.kind is types.Uint:
            if left.bit_length() > right.type.width:
                raise TypeError(
                    f"integer literal '{left}' is wider than the other operand '{right}'"
                )
            # Left will share const-ness of right.
            left = Value(left, right.type)
        else:
            left = lift(left)
    elif not left_int:
        # Right is an int.
        left = lift(left)
        if left.type.kind is types.Uint:
            if right.bit_length() > left.type.width:
                raise TypeError(
                    f"integer literal '{right}' is wider than the other operand '{left}'"
                )
            # Right will share const-ness of left.
            right = Value(right, left.type)
        else:
            right = lift(right)
    else:
        # Both are `int`, so we take our best case to make things work.
        uint = types.Uint(max(left.bit_length(), right.bit_length(), 1), const=True)
        left = Value(left, uint)
        right = Value(right, uint)
    return left, right


def _binary_bitwise(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    type: types.Type
    if left.type.kind is right.type.kind is types.Bool:
        type = types.Bool(const=(left.type.const and right.type.const))
    elif left.type.kind is types.Uint and right.type.kind is types.Uint:
        if left.type != right.type:
            raise TypeError(
                "binary bitwise operations are defined between unsigned integers of the same width,"
                f" but got {left.type.width} and {right.type.width}."
            )
        type = types.Uint(width=left.type.width, const=(left.type.const and right.type.const))
    else:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    return Binary(op, left, right, type)


def bit_and(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'and' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'and' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_and(ClassicalRegister(3, "c"), 0b111)
            Binary(\
Binary.Op.BIT_AND, \
Var(ClassicalRegister(3, 'c'), Uint(3, const=False)), \
Value(7, Uint(3, const=False)), \
Uint(3, const=False))
        """
    return _binary_bitwise(Binary.Op.BIT_AND, left, right)


def bit_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'or' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_or(ClassicalRegister(3, "c"), 0b101)
            Binary(\
Binary.Op.BIT_OR, \
Var(ClassicalRegister(3, 'c'), Uint(3, const=False)), \
Value(5, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _binary_bitwise(Binary.Op.BIT_OR, left, right)


def bit_xor(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'exclusive or' expression node from the given value, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'exclusive or' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_xor(ClassicalRegister(3, "c"), 0b101)
            Binary(\
Binary.Op.BIT_XOR, \
Var(ClassicalRegister(3, 'c'), Uint(3, const=False)), \
Value(5, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _binary_bitwise(Binary.Op.BIT_XOR, left, right)


def _binary_logical(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    left = _coerce_lossless(left, types.Bool(const=left.type.const))
    right = _coerce_lossless(right, types.Bool(const=right.type.const))
    return Binary(op, left, right, types.Bool(const=(left.type.const and right.type.const)))


def logic_and(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a logical 'and' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Logical 'and' of two classical bits::

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> expr.logic_and(Clbit(), Clbit())
            Binary(Binary.Op.LOGIC_AND, Var(<clbit 0>, Bool(const=False)), Var(<clbit 1>, Bool(const=False)), Bool(const=False))
    """
    return _binary_logical(Binary.Op.LOGIC_AND, left, right)


def logic_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a logical 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Logical 'or' of two classical bits

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> expr.logical_and(Clbit(), Clbit())
            Binary(Binary.Op.LOGIC_OR, Var(<clbit 0>, Bool(const=False)), Var(<clbit 1>, Bool(const=False)), Bool(const=False))
    """
    return _binary_logical(Binary.Op.LOGIC_OR, left, right)


def _equal_like(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is not right.type.kind:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    type = types.greater(left.type, right.type)
    return Binary(
        op,
        _coerce_lossless(left, type),
        _coerce_lossless(right, type),
        types.Bool(const=type.const),
    )


def equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create an 'equal' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Equality between a classical register and an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.equal(ClassicalRegister(3, "c"), 7)
            Binary(Binary.Op.EQUAL, \
Var(ClassicalRegister(3, "c"), Uint(3, const=False)), \
Value(7, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _equal_like(Binary.Op.EQUAL, left, right)


def not_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'not equal' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Inequality between a classical register and an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.not_equal(ClassicalRegister(3, "c"), 7)
            Binary(Binary.Op.NOT_EQUAL, \
Var(ClassicalRegister(3, "c"), Uint(3, const=False)), \
Value(7, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _equal_like(Binary.Op.NOT_EQUAL, left, right)


def _binary_relation(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is not right.type.kind or left.type.kind is types.Bool:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    type = types.greater(left.type, right.type)
    return Binary(
        op,
        _coerce_lossless(left, type),
        _coerce_lossless(right, type),
        types.Bool(const=type.const),
    )


def less(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'less than' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is less than an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less(ClassicalRegister(3, "c"), 5)
            Binary(Binary.Op.LESS, \
Var(ClassicalRegister(3, "c"), Uint(3, const=False)), \
Value(5, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _binary_relation(Binary.Op.LESS, left, right)


def less_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'less than or equal to' expression node from the given value, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is less than or equal to another::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less(ClassicalRegister(3, "a"), ClassicalRegister(3, "b"))
            Binary(Binary.Op.LESS_EQUAL, \
Var(ClassicalRegister(3, "a"), Uint(3, const=False)), \
Var(ClassicalRegister(3, "b"), Uint(3, const=False)), \
Uint(3,const=False))
    """
    return _binary_relation(Binary.Op.LESS_EQUAL, left, right)


def greater(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'greater than' expression node from the given value, resolving any implicit casts
    and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is greater than an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less(ClassicalRegister(3, "c"), 5)
            Binary(Binary.Op.GREATER, \
Var(ClassicalRegister(3, "c"), Uint(3, const=False)), \
Value(5, Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _binary_relation(Binary.Op.GREATER, left, right)


def greater_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'greater than or equal to' expression node from the given value, resolving any
    implicit casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is greater than or equal to another::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less(ClassicalRegister(3, "a"), ClassicalRegister(3, "b"))
            Binary(Binary.Op.GREATER_EQUAL, \
Var(ClassicalRegister(3, "a"), Uint(3, const=False)), \
Var(ClassicalRegister(3, "b"), Uint(3, const=False)), \
Uint(3, const=False))
    """
    return _binary_relation(Binary.Op.GREATER_EQUAL, left, right)


def _shift_like(
    op: Binary.Op, left: typing.Any, right: typing.Any, type: types.Type | None
) -> Expr:
    if type is not None and type.kind is not types.Uint:
        raise TypeError(f"type '{type}' is not a valid bitshift operand type")
    if isinstance(left, Expr):
        left = _coerce_lossless(left, type) if type is not None else left
    else:
        left = lift(left, type)
    right = lift(right, try_const=left.type.const)
    if left.type.kind != types.Uint or right.type.kind != types.Uint:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    return Binary(
        op,
        left,
        right,
        types.Uint(width=left.type.width, const=(left.type.const and right.type.const)),
    )


def shift_left(left: typing.Any, right: typing.Any, /, type: types.Type | None = None) -> Expr:
    """Create a 'bitshift left' expression node from the given two values, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    If ``type`` is given, the ``left`` operand will be coerced to it (if possible).

    Examples:
        Shift the value of a standalone variable left by some amount::

            >>> from qiskit.circuit.classical import expr, types
            >>> a = expr.Var.new("a", types.Uint(8))
            >>> expr.shift_left(a, 4)
            Binary(Binary.Op.SHIFT_LEFT, \
Var(<UUID>, Uint(8, const=False), name='a'), \
Value(4, Uint(3, const=True)), \
Uint(8, const=False))

        Shift an integer literal by a variable amount, coercing the type of the literal::

            >>> expr.shift_left(3, a, types.Uint(16))
            Binary(Binary.Op.SHIFT_LEFT, \
Value(3, Uint(16, const=True)), \
Var(<UUID>, Uint(8, const=False), name='a'), \
Uint(16, const=False))
    """
    return _shift_like(Binary.Op.SHIFT_LEFT, left, right, type)


def shift_right(left: typing.Any, right: typing.Any, /, type: types.Type | None = None) -> Expr:
    """Create a 'bitshift right' expression node from the given values, resolving any implicit casts
    and lifting the values into :class:`Value` nodes if required.

    If ``type`` is given, the ``left`` operand will be coerced to it (if possible).

    Examples:
        Shift the value of a classical register right by some amount::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.shift_right(ClassicalRegister(8, "a"), 4)
            Binary(Binary.Op.SHIFT_RIGHT, \
Var(ClassicalRegister(8, "a"), Uint(8, const=False)), \
Value(4, Uint(3, const=True)), \
Uint(8, const=False))
    """
    return _shift_like(Binary.Op.SHIFT_RIGHT, left, right, type)


def index(target: typing.Any, index: typing.Any, /) -> Expr:
    """Index into the ``target`` with the given integer ``index``, lifting the values into
    :class:`Value` nodes if required.

    This can be used as the target of a :class:`.Store`, if the ``target`` is itself an lvalue.

    Examples:
        Index into a classical register with a literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.index(ClassicalRegister(8, "a"), 3)
            Index(Var(ClassicalRegister(8, "a"), Uint(8, const=False)), Value(3, Uint(2, const=True)), Bool(const=False))
    """
    target = lift(target)
    index = lift(index, try_const=target.type.const)
    if target.type.kind is not types.Uint or index.type.kind is not types.Uint:
        raise TypeError(f"invalid types for indexing: '{target.type}' and '{index.type}'")
    return Index(target, index, types.Bool(const=target.type.const and index.type.const))
