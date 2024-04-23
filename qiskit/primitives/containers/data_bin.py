# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Dataclass tools for data namespaces (bins)
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import make_dataclass
from typing import Any


class DataBinMeta(type):
    """Metaclass for :class:`DataBin` that adds the shape to the type name.

    This is so that the class has a custom repr with DataBin<*shape> notation.
    """

    def __repr__(cls):
        name = cls.__name__
        if cls._SHAPE is None:
            return name
        shape = ",".join(map(str, cls._SHAPE))
        return f"{name}<{shape}>"


class DataBin(metaclass=DataBinMeta):
    """Base class for data bin containers.

    Subclasses are typically made via :class:`~make_data_bin`, which is a specialization of
    :class:`make_dataclass`.
    """

    _RESTRICTED_NAMES = {
        "_RESTRICTED_NAMES",
        "_SHAPE",
        "_FIELDS",
        "_FIELD_TYPES",
        "keys",
        "values",
        "items",
    }
    _SHAPE: tuple[int, ...] | None = None
    _FIELDS: tuple[str, ...] = ()
    """The fields allowed in this data bin."""
    _FIELD_TYPES: tuple[type, ...] = ()
    """The types of each field."""

    def __len__(self):
        return len(self._FIELDS)

    def __repr__(self):
        vals = (f"{name}={getattr(self, name)}" for name in self._FIELDS if hasattr(self, name))
        return f"{type(self)}({', '.join(vals)})"

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELDS:
            raise KeyError(f"Key ({key}) does not exist in this data bin.")
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._FIELDS

    def __iter__(self) -> Iterable[str]:
        return iter(self._FIELDS)

    def keys(self) -> Sequence[str]:
        """Return a list of field names."""
        return tuple(self._FIELDS)

    def values(self) -> Sequence[Any]:
        """Return a list of values."""
        return tuple(getattr(self, key) for key in self._FIELDS)

    def items(self) -> Sequence[tuple[str, Any]]:
        """Return a list of field names and values"""
        return tuple((key, getattr(self, key)) for key in self._FIELDS)


def make_data_bin(
    fields: Iterable[tuple[str, type]], shape: tuple[int, ...] | None = None
) -> type[DataBin]:
    """Return a new subclass of :class:`~DataBin` with the provided fields and shape.

    .. code-block:: python

        my_bin = make_data_bin([("alpha", np.NDArray[np.float64])], shape=(20, 30))

        # behaves like a dataclass
        my_bin(alpha=np.empty((20, 30)))

    Args:
        fields: Tuples ``(name, type)`` specifying the attributes of the returned class.
        shape: The intended shape of every attribute of this class.

    Returns:
        A new class.
    """
    field_names, field_types = zip(*fields) if fields else ((), ())
    for name in field_names:
        if name in DataBin._RESTRICTED_NAMES:
            raise ValueError(f"'{name}' is a restricted name for a DataBin.")
    cls = make_dataclass(
        "DataBin",
        dict(zip(field_names, field_types)),
        bases=(DataBin,),
        frozen=True,
        unsafe_hash=True,
        repr=False,
    )
    cls._SHAPE = shape
    cls._FIELDS = field_names
    cls._FIELD_TYPES = field_types
    return cls
