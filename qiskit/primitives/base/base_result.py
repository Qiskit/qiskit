# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Primitive result abstract base class
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import fields
from typing import Any, Dict

from numpy import ndarray


ExperimentData = Dict[str, Any]


class BasePrimitiveResult(ABC):
    """Primitive result abstract base class.

    Base class for Primitive results meant to provide common functionality to all inheriting
    result dataclasses.
    """

    def __post_init__(self) -> None:
        """
        Verify that all fields in any inheriting result dataclass are consistent, after
        instantiation, with the number of experiments being represented.

        This magic method is specific of `dataclasses.dataclass`, therefore all inheriting
        classes must have this decorator.

        Raises:
            TypeError: If one of the data fields is not a Sequence or ``numpy.ndarray``.
            ValueError: Inconsistent number of experiments across data fields.
        """
        for value in self._field_values:  # type: Sequence
            # TODO: enforce all data fields to be tuples instead of sequences
            if not isinstance(value, (Sequence, ndarray)) or isinstance(value, (str, bytes)):
                raise TypeError(
                    f"Expected sequence or `numpy.ndarray`, provided {type(value)} instead."
                )
            if len(value) != self.num_experiments:
                raise ValueError("Inconsistent number of experiments across data fields.")

    @property  # TODO: functools.cached_property when py37 is droppped
    def num_experiments(self) -> int:
        """Number of experiments in any inheriting result dataclass."""
        value: Sequence = self._field_values[0]
        return len(value)

    @property  # TODO: functools.cached_property when py37 is droppped
    def experiments(self) -> tuple[ExperimentData, ...]:
        """Experiment data dicts in any inheriting result dataclass."""
        return tuple(self._generate_experiments())

    def _generate_experiments(self) -> Iterator[ExperimentData]:
        """Generate experiment data dicts in any inheriting result dataclass."""
        names: tuple[str, ...] = self._field_names
        for values in zip(*self._field_values):
            yield dict(zip(names, values))

    def decompose(self) -> Iterator[BasePrimitiveResult]:
        """Generate single experiment result objects from self."""
        for values in zip(*self._field_values):
            yield self.__class__(*[(v,) for v in values])

    @property  # TODO: functools.cached_property when py37 is droppped
    def _field_names(self) -> tuple[str, ...]:
        """Tuple of field names in any inheriting result dataclass."""
        return tuple(field.name for field in fields(self))

    @property  # TODO: functools.cached_property when py37 is droppped
    def _field_values(self) -> tuple:
        """Tuple of field values in any inheriting result dataclass."""
        return tuple(getattr(self, name) for name in self._field_names)
