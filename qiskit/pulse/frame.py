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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qiskit.circuit import Parameter
from qiskit.pulse.utils import validate_index
from qiskit.pulse.exceptions import PulseError


class Frame:
    """A frame is a frequency and a phase."""

    def __init__(self, identifier: str, parametric_index: Optional[Parameter] = None):
        """
        Args:
            identifier: The index of the frame.
            parametric_index: An optional parameter to specify the numeric part of the index.

        Raises:
            PulseError: if the frame identifier is not a string.
        """
        if parametric_index is not None:
            validate_index(parametric_index)

        if not isinstance(identifier, str):
            raise PulseError(f"Frame identifiers must be string. Got {type(identifier)}.")

        self._identifier = (identifier, parametric_index)
        self._hash = hash((type(self), self._identifier))

    @property
    def identifier(self) -> Tuple:
        """Return the index of this frame. The index is a label for a frame."""
        return self._identifier

    @property
    def prefix(self) -> str:
        """Return the prefix of the frame."""
        return self._identifier[0]

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


@dataclass
class FrameDefinition:
    """A class to keep track of frame definitions."""

    # The frequency of the frame at time zero.
    frequency: float

    # The duration of the samples in the control electronics.
    sample_duration: float

    # A user-friendly string defining the purpose of the Frame.
    purpose: str = None

    # The phase of the frame at time zero.
    phase: float = 0.0


class FramesConfiguration:
    """A class that specifies how frames on a backend are configured."""

    def __init__(self):
        """Initialize the frames configuration."""
        self._frames = dict()

    @classmethod
    def from_dict(cls, frames_config: Dict[Frame, Dict]) -> "FramesConfiguration":
        """Create a frame configuration from a dict.

        This dict must have frames as keys and a dict as value. This dict is then
        used to instantiate a FramesDefinition instance. Thus refer to FrameDefinition,
        to see what key-value pairs are needed.
        """
        config = FramesConfiguration()

        for frame, definition in frames_config.items():
            config.add_frame(frame, **definition)

        return config

    def to_dict(self) -> Dict:
        """Export the frames configuration to a dictionary"""
        config = dict()

        for frame, definition in self._frames.items():
            config[frame] = definition.__dict__

        return config

    def add_frame(
        self,
        frame: Frame,
        frequency: float,
        sample_duration: Optional[float] = None,
        purpose: Optional[str] = None
    ):
        """Add a frame to the frame configuration.

        Args:
            frame: The frame instance to add.
            frequency: The frequency of the frame.
            sample_duration: The sample duration.
            purpose: A string describing the purpose of the frame.
        """
        self._frames[frame] = FrameDefinition(
            frequency=frequency, sample_duration=sample_duration, purpose=purpose
        )

    @property
    def definitions(self) -> List[FrameDefinition]:
        """Return the definitions for each frame."""
        return [frame_def for frame_def in self._frames.values()]

    def items(self):
        """Return the items in the frames config."""
        return self._frames.items()

    def __getitem__(self, frame: Frame) -> FrameDefinition:
        """Return the frame definition."""
        return self._frames[frame]

    def __setitem__(self, frame: Frame, frame_def: FrameDefinition):
        """Return the frame definition."""
        self._frames[frame] = frame_def
