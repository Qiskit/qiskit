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

"""Expression-tree nodes."""

# Given the nature of the tree representation and that there are helper functions associated with
# many of the classes whose arguments naturally share names with themselves, it's inconvenient to
# use synonyms everywhere.  This goes for the builtin 'type' as well.
# pylint: disable=redefined-builtin,redefined-outer-name

from __future__ import annotations

__all__ = [
    "Expr",
    "Var",
    "Value",
    "Cast",
    "Unary",
    "Binary",
]

import abc
import enum
import typing

from .. import types

if typing.TYPE_CHECKING:
    import qiskit


_T_co = typing.TypeVar("_T_co", covariant=True)


# If adding nodes, remember to update `visitors.ExprVisitor` as well.


class Expr(abc.ABC):
    """Root base class of all nodes in the expression tree.  The base case should never be
    instantiated directly.

    This must not be subclassed by users; subclasses form the internal data of the representation of
    expressions, and it does not make sense to add more outside of Qiskit library code.

    All subclasses are responsible for setting their ``type`` attribute in their ``__init__``, and
    should not call the parent initialiser."""

    __slots__ = ("type",)

    type: types.Type

    # Sentinel to prevent instantiation of the base class.
    @abc.abstractmethod
    def __init__(self):  # pragma: no cover
        pass

    def accept(
        self, visitor: qiskit.circuit.classical.expr.ExprVisitor[_T_co], /
    ) -> _T_co:  # pragma: no cover
        """Call the relevant ``visit_*`` method on the given :class:`ExprVisitor`.  The usual entry
        point for a simple visitor is to construct it, and then call :meth:`accept` on the root
        object to be visited.  For example::

            expr = ...
            visitor = MyVisitor()
            visitor.accept(expr)

        Subclasses of :class:`Expr` should override this to call the correct virtual method on the
        visitor.  This implements double dispatch with the visitor."""
        return visitor.visit_generic(self)


@typing.final
class Cast(Expr):
    """A cast from one type to another, implied by the use of an expression in a different
    context."""

    __slots__ = ("operand", "implicit")

    def __init__(self, operand: Expr, type: types.Type, implicit: bool = False):
        self.type = type
        self.operand = operand
        self.implicit = implicit

    def accept(self, visitor, /):
        return visitor.visit_cast(self)

    def __eq__(self, other):
        return (
            isinstance(other, Cast)
            and self.type == other.type
            and self.operand == other.operand
            and self.implicit == other.implicit
        )

    def __repr__(self):
        return f"Cast({self.operand}, {self.type}, implicit={self.implicit})"


@typing.final
class Var(Expr):
    """A classical variable."""

    __slots__ = ("var",)

    def __init__(
        self, var: qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister, type: types.Type
    ):
        self.type = type
        self.var = var

    def accept(self, visitor, /):
        return visitor.visit_var(self)

    def __eq__(self, other):
        return isinstance(other, Var) and self.type == other.type and self.var == other.var

    def __repr__(self):
        return f"Var({self.var}, {self.type})"


@typing.final
class Value(Expr):
    """A single scalar value."""

    __slots__ = ("value",)

    def __init__(self, value: typing.Any, type: types.Type):
        self.type = type
        self.value = value

    def accept(self, visitor, /):
        return visitor.visit_value(self)

    def __eq__(self, other):
        return isinstance(other, Value) and self.type == other.type and self.value == other.value

    def __repr__(self):
        return f"Value({self.value}, {self.type})"


@typing.final
class Unary(Expr):
    """A unary expression.

    Args:
        op: The opcode describing which operation is being done.
        operand: The operand of the operation.
        type: The resolved type of the result.
    """

    __slots__ = ("op", "operand")

    class Op(enum.Enum):
        """Enumeration of the opcodes for unary operations.

        The bitwise negation :data:`BIT_NOT` takes a single bit or an unsigned integer of known
        width, and returns a value of the same type.

        The logical negation :data:`LOGIC_NOT` takes an input that is implicitly coerced to a
        Boolean, and returns a Boolean.
        """

        # If adding opcodes, remember to add helper constructor functions in `constructors.py`.
        # The opcode integers should be considered a public interface; they are used by
        # serialisation formats that may transfer data between different versions of Qiskit.
        BIT_NOT = 1
        """Bitwise negation. ``~operand``."""
        LOGIC_NOT = 2
        """Logical negation. ``!operand``."""

        def __str__(self):
            return f"Unary.{super().__str__()}"

        def __repr__(self):
            return f"Unary.{super().__repr__()}"

    def __init__(self, op: Unary.Op, operand: Expr, type: types.Type):
        self.op = op
        self.operand = operand
        self.type = type

    def accept(self, visitor, /):
        return visitor.visit_unary(self)

    def __eq__(self, other):
        return (
            isinstance(other, Unary)
            and self.type == other.type
            and self.op is other.op
            and self.operand == other.operand
        )

    def __repr__(self):
        return f"Unary({self.op}, {self.operand}, {self.type})"


@typing.final
class Binary(Expr):
    """A binary expression.

    Args:
        op: The opcode describing which operation is being done.
        left: The left-hand operand.
        right: The right-hand operand.
        type: The resolved type of the result.
    """

    __slots__ = ("op", "left", "right")

    class Op(enum.Enum):
        """Enumeration of the opcodes for binary operations.

        The bitwise operations :data:`BIT_AND`, :data:`BIT_OR` and :data:`BIT_XOR` apply to two
        operands of the same type, which must be a single bit or an unsigned integer of fixed width.
        The resultant type is the same as the two input types.

        The logical operations :data:`LOGIC_AND` and :data:`LOGIC_OR` first implicitly coerce their
        arguments to Booleans, and then apply the logical operation.  The resultant type is always
        Boolean.

        The binary mathematical relations :data:`EQUAL`, :data:`NOT_EQUAL`, :data:`LESS`,
        :data:`LESS_EQUAL`, :data:`GREATER` and :data:`GREATER_EQUAL` take unsigned integers
        (with an implicit cast to make them the same width), and return a Boolean.
        """

        # If adding opcodes, remember to add helper constructor functions in `constructors.py`
        # The opcode integers should be considered a public interface; they are used by
        # serialisation formats that may transfer data between different versions of Qiskit.
        BIT_AND = 1
        """Bitwise "and". ``lhs & rhs``."""
        BIT_OR = 2
        """Bitwise "or". ``lhs | rhs``."""
        BIT_XOR = 3
        """Bitwise "exclusive or". ``lhs ^ rhs``."""
        LOGIC_AND = 4
        """Logical "and". ``lhs && rhs``."""
        LOGIC_OR = 5
        """Logical "or". ``lhs || rhs``."""
        EQUAL = 6
        """Numeric equality. ``lhs == rhs``."""
        NOT_EQUAL = 7
        """Numeric inequality. ``lhs != rhs``."""
        LESS = 8
        """Numeric less than. ``lhs < rhs``."""
        LESS_EQUAL = 9
        """Numeric less than or equal to. ``lhs <= rhs``"""
        GREATER = 10
        """Numeric greater than. ``lhs > rhs``."""
        GREATER_EQUAL = 11
        """Numeric greater than or equal to. ``lhs >= rhs``."""

        def __str__(self):
            return f"Binary.{super().__str__()}"

        def __repr__(self):
            return f"Binary.{super().__repr__()}"

    def __init__(self, op: Binary.Op, left: Expr, right: Expr, type: types.Type):
        self.op = op
        self.left = left
        self.right = right
        self.type = type

    def accept(self, visitor, /):
        return visitor.visit_binary(self)

    def __eq__(self, other):
        return (
            isinstance(other, Binary)
            and self.type == other.type
            and self.op is other.op
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"Binary({self.op}, {self.left}, {self.right}, {self.type})"
