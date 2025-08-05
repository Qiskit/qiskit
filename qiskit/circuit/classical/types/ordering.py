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

"""Tools for working with the partial ordering of the type system."""

from __future__ import annotations

__all__ = [
    "Ordering",
    "is_subtype",
    "is_supertype",
    "order",
    "greater",
    "CastKind",
    "cast_kind",
]

import enum

# from .types import Type, Bool, Duration, Float, Uint

from qiskit._accelerate.circuit.classical.types.ordering import (
    # Ordering as OxOrdering,
    order,
    is_subtype,
    is_supertype,
    greater,
    # CastKind as OxCastKind,
    cast_kind,
)

# OxOrdering.__repr__ = lambda self: str(self).capitalize()
# Ordering = OxOrdering
# CastKind = OxCastKind

class _Ordering(enum.Enum):
    """Enumeration listing the possible relations between two types.  Types only have a partial
    ordering, so it's possible for two types to have no sub-typing relationship.

    Note that the sub-/supertyping relationship is not the same as whether a type can be explicitly
    cast from one to another."""

    # !!! YOU MUST ALSO UPDATE the underlying Rust enum if you touch this.

    LESS = 1
    """The left type is a strict subtype of the right type."""
    EQUAL = 2
    """The two types are equal."""
    GREATER = 3
    """The left type is a strict supertype of the right type."""
    NONE = 4
    """There is no typing relationship between the two types."""

    def __repr__(self):
        return str(self)

class _CastKind(enum.Enum):
    """A return value indicating the type of cast that can occur from one type to another."""

    # !!! YOU MUST ALSO UPDATE the underlying Rust enum if you touch this.

    EQUAL = 1
    """The two types are equal; no cast node is required at all."""
    IMPLICIT = 2
    """The 'from' type can be cast to the 'to' type implicitly.  A :class:`~.expr.Cast` node with
    ``implicit==True`` is the minimum required to specify this."""
    LOSSLESS = 3
    """The 'from' type can be cast to the 'to' type explicitly, and the cast will be lossless.  This
    requires a :class:`~.expr.Cast`` node with ``implicit=False``, but there's no danger from
    inserting one."""
    DANGEROUS = 4
    """The 'from' type has a defined cast to the 'to' type, but depending on the value, it may lose
    data.  A user would need to manually specify casts."""
    NONE = 5
    """There is no casting permitted from the 'from' type to the 'to' type."""

Ordering = _Ordering
CastKind = _CastKind