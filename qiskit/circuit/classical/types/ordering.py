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

from qiskit._accelerate.circuit.classical.types.ordering import (
    Ordering,
    order,
    is_subtype,
    is_supertype,
    greater,
    CastKind,
    cast_kind,
)