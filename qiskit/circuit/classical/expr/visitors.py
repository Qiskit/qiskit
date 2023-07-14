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
]

import typing

from . import expr

_T_co = typing.TypeVar("_T_co", covariant=True)


class ExprVisitor(typing.Generic[_T_co]):
    """Base class for visitors to the :class:`Expr` tree.  Subclasses should override whichever of
    the ``visit_*`` methods that they are able to handle, and should be organised such that
    non-existent methods will never be called."""

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
