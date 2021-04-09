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

from abc import ABC
from typing import List
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel, PulseChannel
from qiskit.pulse.frame import Frame
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.instructions.frequency import SetFrequency, ShiftFrequency
from qiskit.pulse.instructions.phase import SetPhase, ShiftPhase
from qiskit.pulse.exceptions import PulseError


class Tracker(ABC):
    """
    Implements a class to keep track of the phase and frequency of a frame
    or a pulse channel in a given schedule.
    """

    def __init__(self, index: int, sample_duration: float):
        """
        Args:
            index: The index of the Tracker. Corresponds to the index of a
                :class:`~qiskit.pulse.Frame` or a channel.
            sample_duration: Duration of a sample.
        """
        self._index = index
        self._frequencies_phases = []  # List of (time, frequency, phase) tuples
        self._instructions = {}
        self._sample_duration = sample_duration

    @property
    def index(self) -> int:
        """Return the index of the tracker."""
        return self._index

    def frequency(self, time: int) -> float:
        """
        Get the most recent frequency of the tracker before time.

        Args:
            time: The maximum time for which to get the frequency.

        Returns:
            frequency: The frequency of the frame right before time.
        """
        frequency = self._frequencies_phases[0][1]
        for time_freq in self._frequencies_phases:
            if time_freq[0] <= time:
                frequency = time_freq[1]
            else:
                break

        return frequency

    def phase(self, time: int) -> float:
        """
        Get the most recent phase of the tracker.

        Args:
            time: The maximum time for which to get the phase.

        Returns:
             phase: The phase of the frame up until time.
        """
        if len(self._frequencies_phases) == 0:
            return 0.0

        phase = self._frequencies_phases[0][1]
        last_time = self._frequencies_phases[0][0]

        for time_freq in self._frequencies_phases:
            if time_freq[0] <= time:
                phase = time_freq[2]
                last_time = time_freq[0]
            else:
                break

        freq = self.frequency(time)

        return (phase + 2*np.pi*freq*(time-last_time)*self._sample_duration) % (2*np.pi)

    def set_frequency(self, time: int, frequency: float):
        """Insert a new frequency in the time-ordered frequencies."""
        insert_idx = 0
        for idx, time_freq in enumerate(self._frequencies_phases):
            if time_freq[0] < time:
                insert_idx = idx
            else:
                break

        phase = self.phase(time)

        self._frequencies_phases.insert(insert_idx + 1, (time, frequency, phase))

    def set_phase(self, time: int, phase: float):
        """Insert a new phase in the time-ordered phases."""
        insert_idx = 0
        for idx, time_freq in enumerate(self._frequencies_phases):
            if time_freq[0] < time:
                insert_idx = idx
            else:
                break

        frequency = self.frequency(time)

        self._frequencies_phases.insert(insert_idx + 1, (time, frequency, phase))


class ResolvedFrame(Tracker):
    """
    A class to help the assembler determine the frequency and phase of a
    Frame at any given point in time.
    """

    def __init__(self, frame: Frame, frequency: float, phase: float,
                 sample_duration: float, channels: List[Channel]):
        """
        Args:
            frame: The frame to track.
            frequency: The initial frequency of the frame.
            phase: The initial phase of the frame.
            sample_duration: Duration of a sample.
            channels: The list of channels on which the frame instructions apply.

        Raises:
            PulseError: If there are still parameters in the given frame.
        """
        if isinstance(frame.index, ParameterExpression):
            raise PulseError('A parameterized frame cannot be given to ResolvedFrame.')

        super().__init__(frame.index, sample_duration)
        self._frequencies_phases = [(0, frequency, phase)]
        self._channels = channels

        for ch in self._channels:
            if isinstance(ch.index, ParameterExpression):
                raise PulseError('ResolvedFrame does not allow parameterized channels.')

    @property
    def channels(self) -> List[Channel]:
        """Returns the channels that this frame ties together."""
        return self._channels

    def set_frame_instructions(self, schedule: Schedule):
        """
        Add all matching frame instructions in this schedule to self.

        Args:
            schedule: The schedule from which to extract frame operations.

        Raises:
            PulseError: if the internal filtering does not contain the right
                instructions.
        """
        frame_instruction_types = [ShiftPhase, SetPhase, ShiftFrequency, SetFrequency]
        frame_instructions = schedule.filter(instruction_types=frame_instruction_types)

        for time, inst in frame_instructions.instructions:
            if Frame(self._index) == inst.operands[1]:
                if isinstance(inst, ShiftFrequency):
                    self.set_frequency(time, self.frequency(time) + inst.frequency)
                elif isinstance(inst, SetFrequency):
                    self.set_frequency(time, inst.frequency)
                elif isinstance(inst, ShiftPhase):
                    self.set_phase(time, self.phase(time) + inst.phase)
                elif isinstance(inst, SetPhase):
                    self.set_phase(time, inst.phase)
                else:
                    raise PulseError('Unexpected frame operation.')

    def __repr__(self):
        sub_str = '[' + ', '.join([ch.__repr__() for ch in self._channels]) + ']'
        return f'{self.__class__.__name__}({self._index}, {sub_str})'


class ChannelTracker(Tracker):
    """Class to track the phase and frequency of channels when resolving frames."""

    def __init__(self, channel: PulseChannel, sample_duration: float):
        """
        Args:
            channel: The channel that this tracker tracks.
            sample_duration: Duration of a sample.
        """
        super().__init__(channel.index, sample_duration)
        self._channel = channel

    def is_initialized(self) -> bool:
        """Return true if the channel has been initialized."""
        return len(self._frequencies_phases) > 0
