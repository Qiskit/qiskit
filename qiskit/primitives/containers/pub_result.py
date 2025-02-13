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
Base Pub result class
"""

from __future__ import annotations

from typing import Any

from .data_bin import DataBin


class PubResult:
    """Result of Primitive Unified Bloc."""

    __slots__ = ("_data", "_metadata")

    def __init__(self, data: DataBin, metadata: dict[str, Any] | None = None):
        """Initialize a pub result.

        Args:
            data: Result data.
            metadata: Metadata specific to this pub. Keys are expected to be strings.
        """
        self._data = data
        self._metadata = metadata or {}

    def __repr__(self):
        metadata = f", metadata={self.metadata}" if self.metadata else ""
        return f"{type(self).__name__}(data={self._data}{metadata})"

    @property
    def data(self) -> DataBin:
        """Result data for the pub."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Metadata for the pub."""
        return self._metadata
