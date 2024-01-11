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

"""
==================================================
Expressions (:mod:`qiskit.circuit.classical.expr`)
==================================================

The necessary components for building expressions are all exported from the
:mod:`~.qiskit.circuit.classical.expr` namespace within :mod:`qiskit.circuit.classical`, so you can
choose whether to use qualified access (for example :class:`.expr.Value`) or import the names you
need directly and call them without the prefix.

There are two pathways for constructing expressions.  The classes that form :ref:`the
representation of the expression system <circuit-classical-expressions-expr-representation>`
have constructors that perform zero type checking; it is up to the caller to ensure that they
are building valid objects.  For a more user-friendly interface to direct construction, there
are helper functions associated with most of the classes that do type validation and inference.
These are described below, in :ref:`circuit-classical-expressions-expr-construction`.

.. _circuit-classical-expressions-expr-representation:

Representation
==============

The expression system is based on tree representation.  All nodes in the tree are final
(uninheritable) instances of the abstract base class:

.. autoclass:: Expr

These objects are mutable and should not be reused in a different location without a copy.

The entry point from general circuit objects to the expression system is by wrapping the object
in a :class:`Var` node and associating a :class:`~.types.Type` with it.

.. autoclass:: Var

Similarly, literals used in comparison (such as integers) should be lifted to :class:`Value` nodes
with associated types.

.. autoclass:: Value

The operations traditionally associated with pre-, post- or infix operators in programming are
represented by the :class:`Unary` and :class:`Binary` nodes as appropriate.  These each take an
operation type code, which are exposed as enumerations inside each class as :class:`Unary.Op`
and :class:`Binary.Op` respectively.

.. autoclass:: Unary
    :members: Op
    :member-order: bysource
.. autoclass:: Binary
    :members: Op
    :member-order: bysource

When constructing expressions, one must ensure that the types are valid for the operation.
Attempts to construct expressions with invalid types will raise a regular Python ``TypeError``.

Expressions in this system are defined to act only on certain sets of types.  However, values
may be cast to a suitable supertype in order to satisfy the typing requirements.  In these
cases, a node in the expression tree is used to represent the promotion.  In all cases where
operations note that they "implicitly cast" or "coerce" their arguments, the expression tree
must have this node representing the conversion.

.. autoclass:: Cast


.. _circuit-classical-expressions-expr-construction:

Construction
============

Constructing the tree representation directly is verbose and easy to make a mistake with the
typing.  In many cases, much of the typing can be inferred, scalar values can automatically
be promoted to :class:`Value` instances, and any required promotions can be resolved into
suitable :class:`Cast` nodes.

The functions and methods described in this section are a more user-friendly way to build the
expression tree, while staying close to the internal representation.  All these functions will
automatically lift valid Python scalar values into corresponding :class:`Var` or :class:`Value`
objects, and will resolve any required implicit casts on your behalf.

.. autofunction:: lift

You can manually specify casts in cases where the cast is allowed in explicit form, but may be
lossy (such as the cast of a higher precision :class:`~.types.Uint` to a lower precision one).

.. autofunction:: cast

There are helper constructor functions for each of the unary operations.

.. autofunction:: bit_not
.. autofunction:: logic_not

Similarly, the binary operations and relations have helper functions defined.

.. autofunction:: bit_and
.. autofunction:: bit_or
.. autofunction:: bit_xor
.. autofunction:: logic_and
.. autofunction:: logic_or
.. autofunction:: equal
.. autofunction:: not_equal
.. autofunction:: less
.. autofunction:: less_equal
.. autofunction:: greater
.. autofunction:: greater_equal

Qiskit's legacy method for specifying equality conditions for use in conditionals is to use a
two-tuple of a :class:`.Clbit` or :class:`.ClassicalRegister` and an integer.  This represents an
exact equality condition, and there are no ways to specify any other relations.  The helper function
:func:`lift_legacy_condition` converts this legacy format into the new expression syntax.

.. autofunction:: lift_legacy_condition

Working with the expression tree
================================

A typical consumer of the expression tree wants to recursively walk through the tree, potentially
statefully, acting on each node differently depending on its type.  This is naturally a
double-dispatch problem; the logic of 'what is to be done' is likely stateful and users should be
free to define their own operations, yet each node defines 'what is being acted on'.  We enable this
double dispatch by providing a base visitor class for the expression tree.

.. autoclass:: ExprVisitor
    :members:
    :undoc-members:

Consumers of the expression tree should subclass the visitor, and override the ``visit_*`` methods
that they wish to handle.  Any non-overridden methods will call :meth:`~ExprVisitor.visit_generic`,
which unless overridden will raise a ``RuntimeError`` to ensure that you are aware if new nodes
have been added to the expression tree that you are not yet handling.

For the convenience of simple visitors that only need to inspect the variables in an expression and
not the general structure, the iterator method :func:`iter_vars` is provided.

.. autofunction:: iter_vars

Two expressions can be compared for direct structural equality by using the built-in Python ``==``
operator.  In general, though, one might want to compare two expressions slightly more semantically,
allowing that the :class:`Var` nodes inside them are bound to different memory-location descriptions
between two different circuits.  In this case, one can use :func:`structurally_equivalent` with two
suitable "key" functions to do the comparison.

.. autofunction:: structurally_equivalent
"""

__all__ = [
    "Expr",
    "Var",
    "Value",
    "Cast",
    "Unary",
    "Binary",
    "ExprVisitor",
    "iter_vars",
    "structurally_equivalent",
    "lift",
    "cast",
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

from .expr import Expr, Var, Value, Cast, Unary, Binary
from .visitors import ExprVisitor, iter_vars, structurally_equivalent
from .constructors import (
    lift,
    cast,
    bit_not,
    logic_not,
    bit_and,
    bit_or,
    bit_xor,
    logic_and,
    logic_or,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    lift_legacy_condition,
)
