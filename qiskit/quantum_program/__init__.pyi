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

# The classes here are declared by hand because they come from the compiled extension module,
# which has no signatures of its own. Each member's summary line is repeated so that a hover
# keeps showing one; the full documentation lives on the Rust type. Nothing may be declared here
# that the class does not have, which test_quantum_program.py checks.

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

__all__ = [
    "DType",
    "DataTree",
    "TensorType",
    "bit",
    "bounded",
    "c64",
    "c128",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
]

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

class DType:
    """The element type of a tensor."""

    Bit: ClassVar[DType]
    U8: ClassVar[DType]
    U16: ClassVar[DType]
    U32: ClassVar[DType]
    U64: ClassVar[DType]
    I8: ClassVar[DType]
    I16: ClassVar[DType]
    I32: ClassVar[DType]
    I64: ClassVar[DType]
    F32: ClassVar[DType]
    F64: ClassVar[DType]
    C64: ClassVar[DType]
    C128: ClassVar[DType]

    def __getitem__(self, shape: int | bounded | tuple[int | bounded, ...]) -> TensorType:
        """Return a ``TensorType`` of this dtype whose shape is ``shape``."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class bounded:
    """An axis whose size is not known until run time, but is provably at most ``max``."""

    def __init__(self, max: int, /) -> None: ...
    @property
    def max(self) -> int:
        """The largest size this axis can have."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class TensorType:
    """A dtype paired with a shape, describing a tensor without holding one."""

    def __init__(self, dtype: DType, shape: Sequence[int | bounded], /) -> None: ...
    @property
    def dtype(self) -> DType:
        """The element type."""

    @property
    def shape(self) -> tuple[int | bounded, ...]:
        """The size of each axis, an integer or a `bounded()`."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

bit: DType
u8: DType
u16: DType
u32: DType
u64: DType
i8: DType
i16: DType
i32: DType
i64: DType
f32: DType
f64: DType
c64: DType
c128: DType
