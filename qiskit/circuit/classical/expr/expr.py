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
    "Stretch",
    "Value",
    "Cast",
    "Unary",
    "Binary",
    "Index",
]

import abc
import enum
import typing
import uuid

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
    should not call the parent initializer."""

    __slots__ = ("type", "const")

    type: types.Type
    const: bool

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
        self.const = operand.const
        self.operand = operand
        self.implicit = implicit

    def accept(self, visitor, /):
        return visitor.visit_cast(self)

    def __eq__(self, other):
        return (
            isinstance(other, Cast)
            and self.type == other.type
            and self.const == other.const
            and self.operand == other.operand
            and self.implicit == other.implicit
        )

    def __repr__(self):
        return f"Cast({self.operand}, {self.type}, implicit={self.implicit})"


@typing.final
class Var(Expr):
    """A classical variable.

    These variables take two forms: a new-style variable that owns its storage location and has an
    associated name; and an old-style variable that wraps a :class:`.Clbit` or
    :class:`.ClassicalRegister` instance that is owned by some containing circuit.  In general,
    construction of variables for use in programs should use :meth:`Var.new` or
    :meth:`.QuantumCircuit.add_var`.

    Variables are immutable after construction, so they can be used as dictionary keys."""

    __slots__ = ("var", "name")

    var: qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister | uuid.UUID
    """A reference to the backing data storage of the :class:`Var` instance.  When lifting
    old-style :class:`.Clbit` or :class:`.ClassicalRegister` instances into a :class:`Var`,
    this is exactly the :class:`.Clbit` or :class:`.ClassicalRegister`.  If the variable is a
    new-style classical variable (one that owns its own storage separate to the old
    :class:`.Clbit`/:class:`.ClassicalRegister` model), this field will be a :class:`~uuid.UUID`
    to uniquely identify it."""
    name: str | None
    """The name of the variable.  This is required to exist if the backing :attr:`var` attribute
    is a :class:`~uuid.UUID`, i.e. if it is a new-style variable, and must be ``None`` if it is
    an old-style variable."""

    def __init__(
        self,
        var: qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister | uuid.UUID,
        type: types.Type,
        *,
        name: str | None = None,
    ):
        super().__setattr__("type", type)
        super().__setattr__("const", False)
        super().__setattr__("var", var)
        super().__setattr__("name", name)

    @classmethod
    def new(cls, name: str, type: types.Type) -> typing.Self:
        """Generate a new named variable that owns its own backing storage."""
        return cls(uuid.uuid4(), type, name=name)

    @property
    def standalone(self) -> bool:
        """Whether this :class:`Var` is a standalone variable that owns its storage
        location, if applicable. If false, this is a wrapper :class:`Var` around a
        pre-existing circuit object."""
        return isinstance(self.var, uuid.UUID)

    def accept(self, visitor, /):
        return visitor.visit_var(self)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f"'Var' object attribute '{key}' is read-only")
        raise AttributeError(f"'Var' object has no attribute '{key}'")

    def __hash__(self):
        return hash((self.type, self.var, self.name))

    def __eq__(self, other):
        return (
            isinstance(other, Var)
            and self.type == other.type
            and self.var == other.var
            and self.name == other.name
        )

    def __repr__(self):
        if self.name is None:
            return f"Var({self.var}, {self.type})"
        return f"Var({self.var}, {self.type}, name='{self.name}')"

    def __getstate__(self):
        return (self.var, self.type, self.name)

    def __setstate__(self, state):
        var, type, name = state
        super().__setattr__("type", type)
        super().__setattr__("const", False)
        super().__setattr__("var", var)
        super().__setattr__("name", name)

    def __copy__(self):
        # I am immutable...
        return self

    def __deepcopy__(self, memo):
        # ... as are all my constituent parts.
        return self


@typing.final
class Stretch(Expr):
    """A stretch variable.

    In general, construction of stretch variables for use in programs should use :meth:`Stretch.new`
    or :meth:`.QuantumCircuit.add_stretch`.
    """

    __slots__ = (
        "var",
        "name",
    )

    var: uuid.UUID
    """A :class:`~uuid.UUID` to uniquely identify this stretch."""
    name: str
    """The name of the stretch variable."""

    def __init__(
        self,
        var: uuid.UUID,
        name: str,
    ):
        super().__setattr__("type", types.Duration())
        super().__setattr__("const", True)
        super().__setattr__("var", var)
        super().__setattr__("name", name)

    @classmethod
    def new(cls, name: str) -> typing.Self:
        """Generate a new named stretch variable."""
        return cls(uuid.uuid4(), name)

    def accept(self, visitor, /):
        return visitor.visit_stretch(self)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f"'Stretch' object attribute '{key}' is read-only")
        raise AttributeError(f"'Stretch' object has no attribute '{key}'")

    def __hash__(self):
        return hash((self.var, self.name))

    def __eq__(self, other):
        return isinstance(other, Stretch) and self.var == other.var and self.name == other.name

    def __repr__(self):
        return f"Stretch({self.var}, {self.name})"

    def __getstate__(self):
        return (self.var, self.name)

    def __setstate__(self, state):
        var, name = state
        super().__setattr__("type", types.Duration())
        super().__setattr__("const", True)
        super().__setattr__("var", var)
        super().__setattr__("name", name)

    def __copy__(self):
        # I am immutable...
        return self

    def __deepcopy__(self, memo):
        # ... as are all my constituent parts.
        return self


@typing.final
class Value(Expr):
    """A single scalar value."""

    __slots__ = ("value",)

    def __init__(self, value: typing.Any, type: types.Type):
        self.type = type
        self.value = value
        self.const = True

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
        # serialization formats that may transfer data between different versions of Qiskit.
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
        self.const = operand.const

    def accept(self, visitor, /):
        return visitor.visit_unary(self)

    def __eq__(self, other):
        return (
            isinstance(other, Unary)
            and self.type == other.type
            and self.const == other.const
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

        The bitshift operations :data:`SHIFT_LEFT` and :data:`SHIFT_RIGHT` can take bit-like
        container types (e.g. unsigned integers) as the left operand, and any integer type as the
        right-hand operand.  In all cases, the output bit width is the same as the input, and zeros
        fill in the "exposed" spaces.

        The binary arithmetic operators :data:`ADD`, :data:`SUB:, :data:`MUL`, and :data:`DIV`
        can be applied to two floats or two unsigned integers, which should be made to be of
        the same width during construction via a cast.
        The :data:`ADD`, :data:`SUB`, and :data:`DIV` operators can be applied on two durations
        yielding another duration, or a float in the case of :data:`DIV`. The :data:`MUL` operator
        can also be applied to a duration and a numeric type, yielding another duration. Finally,
        the :data:`DIV` operator can be used to divide a duration by a numeric type, yielding a
        duration.
        """

        # If adding opcodes, remember to add helper constructor functions in `constructors.py`
        # The opcode integers should be considered a public interface; they are used by
        # serialization formats that may transfer data between different versions of Qiskit.
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
        SHIFT_LEFT = 12
        """Zero-padding bitshift to the left.  ``lhs << rhs``."""
        SHIFT_RIGHT = 13
        """Zero-padding bitshift to the right.  ``lhs >> rhs``."""
        ADD = 14
        """Addition. ``lhs + rhs``."""
        SUB = 15
        """Subtraction. ``lhs - rhs``."""
        MUL = 16
        """Multiplication. ``lhs * rhs``."""
        DIV = 17
        """Division. ``lhs / rhs``."""

        def __str__(self):
            return f"Binary.{super().__str__()}"

        def __repr__(self):
            return f"Binary.{super().__repr__()}"

    def __init__(self, op: Binary.Op, left: Expr, right: Expr, type: types.Type):
        self.op = op
        self.left = left
        self.right = right
        self.type = type
        self.const = left.const and right.const

    def accept(self, visitor, /):
        return visitor.visit_binary(self)

    def __eq__(self, other):
        return (
            isinstance(other, Binary)
            and self.type == other.type
            and self.const == other.const
            and self.op is other.op
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"Binary({self.op}, {self.left}, {self.right}, {self.type})"


@typing.final
class Index(Expr):
    """An indexing expression.

    Args:
        target: The object being indexed.
        index: The expression doing the indexing.
        type: The resolved type of the result.
    """

    __slots__ = ("target", "index")

    def __init__(self, target: Expr, index: Expr, type: types.Type):
        self.target = target
        self.index = index
        self.type = type
        self.const = target.const and index.const

    def accept(self, visitor, /):
        return visitor.visit_index(self)

    def __eq__(self, other):
        return (
            isinstance(other, Index)
            and self.type == other.type
            and self.const == other.const
            and self.target == other.target
            and self.index == other.index
        )

    def __repr__(self):
        return f"Index({self.target}, {self.index}, {self.type})"
