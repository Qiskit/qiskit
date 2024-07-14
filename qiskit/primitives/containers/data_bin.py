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

from typing import Any, ItemsView, Iterable, KeysView, ValuesView

import numpy as np

from .shape import ShapedMixin, ShapeInput, shape_tuple


def _value_repr(value: Any) -> str:
    """Helper function for :meth:`DataBin.__repr__`."""
    if isinstance(value, np.ndarray):
        return f"np.ndarray(<shape={value.shape}, dtype={value.dtype}>)"
    return repr(value)


class DataBin(ShapedMixin):
    """Namespace for storing data.

    .. code-block:: python

        data = DataBin(
            alpha=BitArray.from_bitstrings(["0010"]),
            beta=np.array([1.2])
        )

        print("alpha data:", data.alpha)
        print("beta data:", data.beta)

    """

    __slots__ = ("_data", "_shape")

    _RESTRICTED_NAMES = frozenset(
        {
            "_RESTRICTED_NAMES",
            "_SHAPE",
            "_FIELDS",
            "_FIELD_TYPES",
            "_data",
            "_shape",
            "keys",
            "values",
            "items",
            "shape",
            "ndim",
            "size",
        }
    )

    def __init__(self, *, shape: ShapeInput = (), **data):
        """
        Args:
            data: Name/value data to place in the data bin.
            shape: The leading shape common to all entries in the data bin. This defaults to
                the trivial leading shape of ``()`` that is compatible with all objects.

        Raises:
            ValueError: If a name overlaps with a method name on this class.
            ValueError: If some value is inconsistent with the provided shape.
        """
        if not self._RESTRICTED_NAMES.isdisjoint(data):
            bad_names = sorted(self._RESTRICTED_NAMES.intersection(data))
            raise ValueError(f"Cannot assign with these field names: {bad_names}")

        _setattr = super().__setattr__
        _setattr("_shape", shape_tuple(shape))
        _setattr("_data", data)

        ndim = len(self._shape)
        for name, value in data.items():
            if getattr(value, "shape", shape)[:ndim] != shape:
                raise ValueError(f"The value of '{name}' does not lead with the shape {shape}.")
            _setattr(name, value)

        super().__init__()

    def __len__(self):
        return len(self._data)

    def __setattr__(self, *_):
        raise NotImplementedError

    def __repr__(self):
        vals = [f"{name}={_value_repr(val)}" for name, val in self.items()]
        if self.ndim:
            vals.append(f"shape={self.shape}")
        return f"{type(self).__name__}({', '.join(vals)})"

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as ex:
            raise KeyError(f"Key ({key}) does not exist in this data bin.") from ex

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def keys(self) -> KeysView[str]:
        """Return a view of field names."""
        return self._data.keys()

    def values(self) -> ValuesView[Any]:
        """Return a view of values."""
        return self._data.values()

    def items(self) -> ItemsView[str, Any]:
        """Return a view of field names and values"""
        return self._data.items()

    # The following properties exist to provide support to legacy private class attributes which
    # gained widespread prior to qiskit 1.1. These properties will be removed once the internal
    # projects have made the appropriate changes.

    @property
    def _FIELDS(self) -> tuple[str, ...]:  # pylint: disable=invalid-name
        return tuple(self._data)

    @property
    def _FIELD_TYPES(self) -> tuple[Any, ...]:  # pylint: disable=invalid-name
        return tuple(map(type, self.values()))

    @property
    def _SHAPE(self) -> tuple[int, ...]:  # pylint: disable=invalid-name
        return self.shape


# pylint: disable=unused-argument
def make_data_bin(
    fields: Iterable[tuple[str, type]], shape: tuple[int, ...] | None = None
) -> type[DataBin]:
    """Return the :class:`~DataBin` type.

    .. note::
        This class used to return a subclass of :class:`~DataBin`. However, that caused confusion
        and didn't have a useful purpose. Several internal projects made use of this internal
        function prior to qiskit 1.1. This function will be removed once these internal projects
        have made the appropriate changes.

    Args:
        fields: Tuples ``(name, type)`` specifying the attributes of the returned class.
        shape: The intended shape of every attribute of this class.

    Returns:
        The :class:`DataBin` type.
    """
    return DataBin
