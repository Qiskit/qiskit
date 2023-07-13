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
==============================================
Typing (:mod:`qiskit.circuit.classical.types`)
==============================================


The type system of the expression tree is exposed through this module.  This is inherently linked to
the expression system in the :mod:`~.classical.expr` module, as most expressions can only be
understood with the context of the types that they act on.

All types inherit from an abstract base class:

.. autoclass:: Type

Types should be considered immutable objects, and you must not mutate them.  It is permissible to
reuse a :class:`Type` that you take from another object without copying it, and generally this will
be the best approach for performance.  :class:`Type` objects are designed to be small amounts of
data, and it's best to point to the same instance of the data where possible rather than
heap-allocating a new version of the same thing.  Where possible, the class constructors will return
singleton instances to facilitate this.

The two different types available are for Booleans (corresponding to :class:`.Clbit` and the
literals ``True`` and ``False``), and unsigned integers (corresponding to
:class:`.ClassicalRegister` and Python integers).

.. autoclass:: Bool
.. autoclass:: Uint

Note that :class:`Uint` defines a family of types parametrised by their width; it is not one single
type, which may be slightly different to the 'classical' programming languages you are used to.

There are some functions on these types exposed here as well.  These are mostly expected to be used
only in manipulations of the expression tree; users who are building expressions using the
:ref:`user-facing construction interface <circuit-classical-expressions-expr-construction>` should
not need to use these.

The type system is equipped with a partial ordering, where :math:`a < b` is interpreted as
":math:`a` is a strict subtype of :math:`b`".  Note that the partial ordering is a subset of the
directed graph that describes the allowed explicit casting operations between types.  The partial
ordering defines when one type may be lossless directly interpreted as another.

The low-level interface to querying the subtyping relationship is the :func:`order` function.

.. autofunction:: order

The return value is an enumeration :class:`Ordering` that describes what, if any, subtyping
relationship exists between the two types.

.. autoclass:: Ordering
    :member-order: bysource

Some helper methods are then defined in terms of this low-level :func:`order` primitive:

.. autofunction:: is_subtype
.. autofunction:: is_supertype
.. autofunction:: greater
"""

__all__ = [
    "Type",
    "Bool",
    "Uint",
    "Ordering",
    "order",
    "is_subtype",
    "is_supertype",
    "greater",
]

from .types import Type, Bool, Uint
from .ordering import Ordering, order, is_subtype, is_supertype, greater
