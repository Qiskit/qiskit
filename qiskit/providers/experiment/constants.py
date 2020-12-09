# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Experiment constants."""

import enum
from typing import Any


class ResultQuality(enum.Enum):
    """Possible values for analysis result quality."""

    def __new__(cls, description: str, ranking: int = 0) -> 'ResultQuality':
        # ranking is defaulted to 0 to silence linter.
        obj = object.__new__(cls)
        obj._value_ = description
        obj.ranking = ranking
        return obj

    def __ge__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.ranking >= other.ranking  # type: ignore[attr-defined]
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.ranking > other.ranking  # type: ignore[attr-defined]
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.ranking <= other.ranking  # type: ignore[attr-defined]
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.ranking < other.ranking  # type: ignore[attr-defined]
        return NotImplemented

    HUMAN_BAD = 'Human Bad', 1
    COMPUTER_BAD = 'Computer Bad', 2
    NO_INFORMATION = 'No Information', 3
    COMPUTER_GOOD = 'Computer Good', 4
    HUMAN_GOOD = 'Human Good', 5
