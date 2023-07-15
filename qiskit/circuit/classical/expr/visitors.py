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

"""Expression visitors."""

from __future__ import annotations

__all__ = [
    "ExprVisitor",
    "iter_vars",
]

import typing

from . import expr

_T_co = typing.TypeVar("_T_co", covariant=True)


class ExprVisitor(typing.Generic[_T_co]):
    """Base class for visitors to the :class:`Expr` tree.  Subclasses should override whichever of
    the ``visit_*`` methods that they are able to handle, and should be organised such that
    non-existent methods will never be called."""

    # The method names are self-explanatory and docstrings would just be noise.
    # pylint: disable=missing-function-docstring

    __slots__ = ()

    def visit_generic(self, node: expr.Expr, /) -> _T_co:  # pragma: no cover
        raise RuntimeError(f"expression visitor {self} has no method to handle expr {node}")

    def visit_var(self, node: expr.Var, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_value(self, node: expr.Value, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_unary(self, node: expr.Unary, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_binary(self, node: expr.Binary, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_cast(self, node: expr.Cast, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)


class _VarWalkerImpl(ExprVisitor[typing.Iterable[expr.Var]]):
    __slots__ = ()

    def visit_var(self, node, /):
        yield node

    def visit_value(self, node, /):
        yield from ()

    def visit_unary(self, node, /):
        yield from node.operand.accept(self)

    def visit_binary(self, node, /):
        yield from node.left.accept(self)
        yield from node.right.accept(self)

    def visit_cast(self, node, /):
        yield from node.operand.accept(self)


_VAR_WALKER = _VarWalkerImpl()


def iter_vars(node: expr.Expr) -> typing.Iterator[expr.Var]:
    """Get an iterator over the :class:`~.expr.Var` nodes referenced at any level in the given
    :class:`~.expr.Expr`.

    Examples:
        Print out the name of each :class:`.ClassicalRegister` encountered::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.classical import expr

            cr1 = ClassicalRegister(3, "a")
            cr2 = ClassicalRegister(3, "b")

            for node in expr.iter_vars(expr.bit_and(expr.bit_not(cr1), cr2)):
                if isinstance(node.var, ClassicalRegister):
                    print(node.var.name)
    """
    yield from node.accept(_VAR_WALKER)
