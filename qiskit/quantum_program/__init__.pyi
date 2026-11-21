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

# The types here are declared by hand because DataTree comes from the compiled extension module,
# which has no signatures of its own. Each member's summary line is repeated so that a hover
# keeps showing one; the full documentation lives on the Rust type. Nothing may be declared here
# that the class does not have, which test_quantum_program.py checks.

from collections.abc import Iterator
from typing import Any, ClassVar

__all__ = ["DataTree"]

class DataTree:
    """A leaf holding one value, or a branch of ordered children."""

    # Equality reads the leaves, which need not be hashable. The ignore is what typeshed uses to
    # narrow an inherited method to None.
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __init__(self, object: Any, /) -> None: ...
    @property
    def is_leaf(self) -> bool:
        """Whether this is a leaf, as opposed to a branch of children."""

    @property
    def leaf(self) -> Any:
        """The value this leaf holds. Raises `TypeError` on a branch."""

    @property
    def is_mapping(self) -> bool:
        """Whether this branch names its children. A branch that names them has a mapping form."""

    def keys(self) -> list[str]:
        """Return the name of every child, in order."""

    def values(self) -> list[Any]:
        """Return every child, in order."""

    def items(self) -> list[tuple[str, Any]]:
        """Return every child with its name, in order."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, key: str | int) -> Any: ...
    def __contains__(self, key: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
