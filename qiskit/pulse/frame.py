# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implements a Frame."""

from typing import Optional, Tuple

from qiskit.circuit import Parameter
from qiskit.pulse.utils import validate_index


class Frame:
    """A frame is a frequency and a phase."""

    def __init__(self, identifier: str, parametric_index: Optional[Parameter] = None):
        """
        Args:
            identifier: The index of the frame.
            parametric_index: An optional parameter to specify the numeric part of the index.
        """
        validate_index(parametric_index)
        self._identifier = (identifier, parametric_index)
        self._hash = hash((type(self), self._identifier))

    @property
    def identifier(self) -> Tuple:
        """Return the index of this frame. The index is a label for a frame."""
        return self._identifier

    @property
    def name(self) -> str:
        """Return the shorthand alias for this frame, which is based on its type and index."""
        if self._identifier[1] is None:
            return f"{self._identifier[0]}"

        return f"{self._identifier[0]}{self._identifier[1].name}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __eq__(self, other: "Frame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same index.

        Args:
            other: The frame to compare to this frame.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._identifier == other._identifier

    def __hash__(self):
        return self._hash
