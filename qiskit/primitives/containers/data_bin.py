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
Dataclass tools for data namespaces (bins)
"""

from dataclasses import make_dataclass
from typing import Iterable, Optional, Tuple


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

    Subclasses are typically made via :class:`~make_databin`, which is a specialization of
    :class:`make_dataclass`.
    """

    _RESTRICTED_NAMES = ("_RESTRICTED_NAMES", "_SHAPE", "_FIELDS", "_FIELD_TYPES")
    _SHAPE: Optional[Tuple[int, ...]] = None
    _FIELDS: Tuple[str, ...] = ()
    """The fields allowed in this data bin."""
    _FIELD_TYPES: Tuple[type, ...] = ()
    """The types of each field."""

    def __repr__(self):
        vals = (f"{name}={getattr(self, name)}" for name in self._FIELDS if hasattr(self, name))
        return f"{type(self)}({', '.join(vals)})"


def make_databin(
    fields: Iterable[Tuple[str, type]], shape: Optional[Tuple[int, ...]] = None
) -> DataBinMeta:
    """Return a new subclass of :class:`~DataBin` with the provided fields and shape.

    .. code-block:: python

        my_bin = make_databin([("alpha", np.NDArray[np.float])], shape=(20, 30))

        # behaves like a dataclass
        my_bin(alpha=np.empty((20, 30)))

    Args:
        fields: Tuples ``(name, type)`` specifying the attributes of the returned class.
        shape: The intended shape of every attribute of this class.

    Returns:
        A new class.
    """
    fields = list(fields)
    field_names = tuple(name for name, _ in fields)
    field_types = tuple(field_type for _, field_type in fields)
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
