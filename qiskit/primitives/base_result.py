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

from abc import ABC
from dataclasses import fields
from typing import Any, Sized, Tuple


class BaseResult(ABC):
    """Primitive result abstract base class.

    Base class for Primitive results meant to provide common functionality to all inheriting
    result dataclasses.
    """

    def __post_init__(self) -> None:
        """
        Verify that all fields in any inheriting result dataclass are consistent, after
        instantiation, with the number of experiments being represented.

        This magic method is especific of `dataclasses.dataclass`, therefore all inheriting
        classes must have this decorator.

        Raises:
            ValueError: Inconsistent number of experiments across data fields.
        """
        for value in self._field_values:  # type: Sized
            if len(value) != self.num_experiments:
                raise ValueError("Inconsistent number of experiments across data fields.")

    @property
    def num_experiments(self) -> int:
        """Number of experiments in any inheriting result dataclass."""
        value: Sized = self._field_values[0]
        return len(value)

    @property
    def experiments(self) -> Tuple[Tuple[Any, ...], ...]:
        """Experiment data tuples from any inheriting result dataclass."""
        return tuple(zip(*self._field_values))

    @property
    def _field_names(self) -> Tuple[str, ...]:
        """Tuple of field names in any inheriting result dataclass."""
        return tuple(field.name for field in fields(self))

    @property
    def _field_values(self) -> Tuple[Any, ...]:
        """Tuple of field values in any inheriting result dataclass."""
        return tuple(getattr(self, name) for name in self._field_names)
