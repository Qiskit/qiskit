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

"""PrimitiveResult"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Generic, Optional, TypeVar

from .pubs_result import PubsResult

T = TypeVar("T", bound=PubsResult)


class PrimitiveResult(Generic[T]):
    """A container for multiple pubs results and global metadata."""

    def __init__(self, pubs_results: Iterable[T], metadata: Optional[dict[str, Any]] = None):
        """
        Args:
            pubs_results: Pubs results.
            metadata: Any metadata that doesn't make sense to put inside of pubs results.
        """
        self._pubs_results = list(pubs_results)
        self._metadata = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of this primitive result."""
        return self._metadata

    def __getitem__(self, index) -> T:
        return self._pubs_results[index]

    def __len__(self) -> int:
        return len(self._pubs_results)

    def __repr__(self) -> str:
        return f"PrimitiveResult({self._pubs_results}, metadata={self.metadata})"

    def __iter__(self) -> Iterable[T]:
        return iter(self._pubs_results)
