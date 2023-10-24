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
Mixed Frames
"""

from .frames import Frame
from .logical_elements import LogicalElement


class MixedFrame:
    """Representation of a :class:`LogicalElement` and :class:`Frame` combination.

    Most instructions need to be associated with both a :class:`LogicalElement` and a :class:`Frame`.
    The combination
    of the two is called a mixed frame and is represented by a :class:`MixedFrame` object.

    In most cases the :class:`MixedFrame` is used more by the compiler, and a pulse program
    can be written without :class:`MixedFrame` s, by setting :class:`LogicalElement` and
    :class:`Frame` independently. However, in some cases using :class:`MixedFrame` s can
    better convey the meaning of the code, and change the compilation process. One example
    is the use of the shift/set frequency/phase instructions which are not broadcasted to other
    :class:`MixedFrame` s if applied on a specific :class:`MixedFrame` (unlike the behavior
    of :class:`Frame`). User can also use a subclass of :class:`MixedFrame` for a particular
    combination of logical elements and frames as if a syntactic sugar. This might
    increase the readability of a user pulse program. As an example consider the cross
    resonance architecture, in which a pulse is played on a target qubit frame and applied
    to a control qubit logical element.
    """

    def __init__(self, logical_element: LogicalElement, frame: Frame):
        """Create ``MixedFrame``.

        Args:
            logical_element: The logical element associated with the mixed frame.
            frame: The frame associated with the mixed frame.
        """
        self._logical_element = logical_element
        self._frame = frame
        self._hash = hash((self._logical_element, self._frame))

    @property
    def logical_element(self) -> LogicalElement:
        """Return the ``LogicalElement`` of this mixed frame."""
        return self._logical_element

    @property
    def frame(self) -> Frame:
        """Return the ``Frame`` of this mixed frame."""
        return self._frame

    def __repr__(self) -> str:
        return f"MixedFrame({self.logical_element},{self.frame})"

    def __eq__(self, other: "MixedFrame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same logical
        element and frame.

        Args:
            other: The mixed frame to compare to this one.

        Returns:
            True iff equal.
        """
        return self._logical_element == other._logical_element and self._frame == other._frame

    def __hash__(self) -> int:
        return self._hash
