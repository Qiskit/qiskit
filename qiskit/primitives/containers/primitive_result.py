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
from typing import Any, Optional, Sequence, TypeVar

from .task_result import TaskResult

T = TypeVar("T", bound=TaskResult)


class PrimitiveResult(Sequence[T]):
    """A container for multiple task results and global metadata."""

    def __init__(self, task_results: Iterable[T], metadata: Optional[dict[str, Any]] = None):
        """
        Args:
            task_results: Task results.
            metadata: Any metadata that doesn't make sense to put inside of task results.
        """
        self._task_results = list(task_results)
        self._metadata = metadata if metadata is not None else {}

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of this primitive result."""
        return self._metadata

    def __getitem__(self, index) -> T:
        return self._task_results[index]

    def __len__(self) -> int:
        return len(self._task_results)

    def __repr__(self) -> str:
        return f"PrimitiveResult({self._task_results}, metadata={self.metadata})"
