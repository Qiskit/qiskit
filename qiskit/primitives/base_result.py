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
Primitive result abstract class
"""

from abc import ABC
from dataclasses import fields
from typing import Sized


class BaseResult(ABC):
    """Primitive result base class.

    Base class for Primitive results meant to provide common functionality to all inheriting result dataclasses.
    """

    def __post_init__(self) -> None:
        """Verify that all fields are consistent with the number of experiments represented."""
        for val in self._field_values:  # type: Sized
            if len(val) != self.num_experiments:
                raise ValueError("Inconsistent number of experiments across data fields.")

    @property
    def num_experiments(self) -> int:
        """Number of experiments."""
        field_value: Sized = self._field_values.pop()
        return len(field_value)

    @property
    def _field_values(self) -> list:
        """Returns list of field values in any inheriting dataclass."""
        return [getattr(self, field.name) for field in fields(self)]
