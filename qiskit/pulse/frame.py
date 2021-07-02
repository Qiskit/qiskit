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
from typing import Dict, List, Optional, Set, Tuple, Union

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.utils import validate_index
from qiskit.pulse.exceptions import PulseError


class Frame:
    """A frame is a frequency and a phase."""

    def __init__(self, prefix: str, index: Union[int, Parameter]):
        """
        Frames are identified using an identifier, such as "Q10". However, parameters
        in Qiskit currently do not support strings as valid assignment values. Therefore,
        to allow for parametric identifiers in frames we separate the string prefix from
        the numeric (and possibly parametric) index. This behaviour may change in the
        future if parameters can be assigned string values.

        Args:
            prefix: The index of the frame.
            index: An optional parameter to specify the numeric part of the index.

        Raises:
            PulseError: if the frame identifier is not a string or if the identifier
                contains numbers which should have been specified as the index.
        """
        if index is not None:
            validate_index(index)

        if not isinstance(prefix, str):
            raise PulseError(f"Frame identifiers must be string. Got {type(prefix)}.")

        if any(char.isdigit() for char in prefix):
            raise PulseError(f"Frame prefixes cannot contain digits. Found {prefix}")

        self._identifier = (prefix, index)
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
    def index(self) -> Union[int, ParameterExpression]:
        """Return the index of the frame."""
        return self._identifier[1]

    @property
    def name(self) -> str:
        """Return the shorthand alias for this frame, which is based on its type and index."""
        if isinstance(self._identifier[1], Parameter):
            return f"{self._identifier[0]}{self._identifier[1].name}"

        return f"{self._identifier[0]}{self._identifier[1]}"

    @property
    def parameters(self) -> Set:
        """Parameters which determine the frame index."""
        if isinstance(self._identifier[1], ParameterExpression):
            return self._identifier[1].parameters

        return set()

    def is_parameterized(self) -> bool:
        """Return true if the identifier has a parameter."""
        return isinstance(self._identifier[1], ParameterExpression)

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

    # A user-friendly string defining the purpose of the Frame.
    purpose: str = None

    # True if this frame is the native frame of a physical channel
    has_physical_channel: bool = False

    # The phase of the frame at time zero.
    phase: float = 0.0

    # Tolerance on phase and frequency shifts. Shifts below this value are ignored.
    tolerance: float = 1.0e-8


class FramesConfiguration:
    """A class that specifies how frames on a backend are configured.

    Internally this class stores the frames configuration in a dictionary where frames
    are keys with a corresponding instance of :class:`FrameDefinition` as value. All
    frames are required to have the same sample duration which is stored in a separate
    property of the :class:`FramesConfiguration`.
    """

    def __init__(self):
        """Initialize the frames configuration."""
        self._frames = dict()
        self._sample_duration = None

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
        has_physical_channel: Optional[bool] = False,
        purpose: Optional[str] = None,
    ):
        """Add a frame to the frame configuration.

        Args:
            frame: The frame instance to add. This instance will be a key in the internal
                dictionary that allows the frame resolution mechanism to identify the
                frequency of the frame.
            frequency: The frequency of the frame which will determine the frequency of
                all pulses played in the frame added to the configuration.
            has_physical_channel: Whether this frame is the native frame of a physical channel.
            purpose: A human readable string describing the purpose of the frame.
        """
        self._frames[frame] = FrameDefinition(
            frequency=frequency, has_physical_channel=has_physical_channel, purpose=purpose
        )

    @property
    def definitions(self) -> List[FrameDefinition]:
        """Return the definitions for each frame."""
        return [frame_def for frame_def in self._frames.values()]

    @property
    def sample_duration(self) -> float:
        """Return the duration of a sample."""
        return self._sample_duration

    @sample_duration.setter
    def sample_duration(self, sample_duration):
        """Set the duration of the samples."""
        self._sample_duration = sample_duration

    def items(self):
        """Return the items in the frames config."""
        return self._frames.items()

    def get(self, frame: Frame, default: FrameDefinition) -> FrameDefinition:
        return self._frames.get(frame, default)

    def __getitem__(self, frame: Frame) -> FrameDefinition:
        """Return the frame definition."""
        return self._frames[frame]

    def __setitem__(self, frame: Frame, frame_def: FrameDefinition):
        """Return the frame definition."""
        self._frames[frame] = frame_def
