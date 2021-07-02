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
from typing import Optional
from dataclasses import dataclass
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.frame import Frame, FrameDefinition
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.instructions.frequency import SetFrequency, ShiftFrequency
from qiskit.pulse.instructions.phase import SetPhase, ShiftPhase
from qiskit.pulse.exceptions import PulseError


@dataclass
class TimeFrequencyPhase:
    """Class to help keep track of time, frequency and phase."""

    time: float
    frequency: float
    phase: float


class Tracker(ABC):
    """
    Implements a class to keep track of the phase and frequency of a frame
    or a pulse channel in a given schedule.
    """

    def __init__(self, identifier: str, sample_duration: float):
        """
        Args:
            identifier: The identifier of the Tracker.
            sample_duration: Duration of a sample.
        """
        self._identifier = identifier
        self._frequencies_phases = []  # List of TimeFreqPhase instances
        self._instructions = {}
        self._sample_duration = sample_duration

    @property
    def identifier(self) -> str:
        """Return the identifier of the tracker."""
        return self._identifier

    def frequency(self, time: int) -> float:
        """
        Get the most recent frequency of the tracker before time.

        Args:
            time: The maximum time for which to get the frequency.

        Returns:
            frequency: The frequency of the frame right before time.
        """
        frequency = self._frequencies_phases[-1].frequency
        for tfp in reversed(self._frequencies_phases):
            frequency = tfp.frequency

            if tfp.time <= time:
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

        phase = self._frequencies_phases[-1].phase
        last_time = self._frequencies_phases[-1].time

        for tfp in reversed(self._frequencies_phases):
            phase = tfp.phase
            last_time = tfp.time

            if tfp.time <= time:
                break

        freq = self.frequency(time)

        return phase + 2 * np.pi * freq * (time - last_time) * self._sample_duration

    def set_frequency(self, time: int, frequency: float):
        """Insert a new frequency in the time-ordered frequencies.

        Args:
            time: The time in samples (i.e. measured in units of dt).
            frequency: The frequency to which self is set after the given time.
        """
        tfp = TimeFrequencyPhase(time=time, frequency=frequency, phase=self.phase(time))
        self._frequencies_phases = sorted(self._frequencies_phases + [tfp], key=lambda x: x.time)

    def set_phase(self, time: int, phase: float):
        """Insert a new phase in the time-ordered phases.

        Args:
            time: The time in samples (i.e. measured in units of dt).
            phase: The phase to which self is set after the given time.
        """
        tfp = TimeFrequencyPhase(time=time, frequency=self.frequency(time), phase=phase)
        self._frequencies_phases = sorted(self._frequencies_phases + [tfp], key=lambda x: x.time)

    def set_frequency_phase(self, time: int, frequency: float, phase: float):
        """Insert a new frequency and phase at the given time.

        Args:
            time: The time in samples (i.e. measured in units of dt).
            frequency: The frequency to which self is set after the given time.
            phase: The phase to which self is set after the given time.
        """
        tfp = TimeFrequencyPhase(time=time, frequency=frequency, phase=phase)
        self._frequencies_phases = sorted(self._frequencies_phases + [tfp], key=lambda x: x.time)


class ResolvedFrame(Tracker):
    """
    A class to help the assembler determine the frequency and phase of a
    Frame at any given point in time.
    """

    def __init__(self, frame: Frame, definition: FrameDefinition, sample_duration: float):
        """Initialized a resolved frame.

        Args:
            frame: The frame to track.
            definition: An instance of the FrameDefinition dataclass which defines
                the frequency, phase, sample_duration, and purpose of a frame.
            sample_duration: The duration of a sample.

        Raises:
            PulseError: If there are still parameters in the given frame.
        """
        if isinstance(frame.identifier[1], ParameterExpression):
            raise PulseError("A parameterized frame cannot initialize a ResolvedFrame.")

        super().__init__(frame.name, sample_duration)

        self._frequencies_phases = [
            TimeFrequencyPhase(time=0, frequency=definition.frequency, phase=definition.phase)
        ]

        self._purpose = definition.purpose
        self._tolerance = definition.tolerance
        self._has_physical_channel = definition.has_physical_channel
        self._frame = frame

    @property
    def purpose(self) -> str:
        """Return the purpose of the frame."""
        return self.purpose

    @property
    def tolerance(self) -> float:
        """Tolerance on phase and frequency shifts. Shifts below this value are ignored."""
        return self._tolerance

    def set_frame_instructions(self, schedule: Schedule):
        """
        Add all matching frame instructions in this schedule to self if and only if
        self does not correspond to a frame that is the native frame of a channel.

        Args:
            schedule: The schedule from which to extract frame operations.

        Raises:
            PulseError: if the internal filtering does not contain the right
                instructions.
        """
        frame_instruction_types = (ShiftPhase, SetPhase, ShiftFrequency, SetFrequency)
        frame_instructions = schedule.filter(instruction_types=frame_instruction_types)

        for time, inst in frame_instructions.instructions:
            if isinstance(inst, frame_instruction_types) and self._frame.name == inst.frame.name:
                if inst.channel is None:
                    if isinstance(inst, ShiftFrequency):
                        self.set_frequency(time, self.frequency(time) + inst.frequency)
                    elif isinstance(inst, SetFrequency):
                        self.set_frequency(time, inst.frequency)
                    elif isinstance(inst, ShiftPhase):
                        self.set_phase(time, self.phase(time) + inst.phase)
                    elif isinstance(inst, SetPhase):
                        self.set_phase(time, inst.phase)
                    else:
                        raise PulseError("Unexpected frame operation.")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.identifier}, {self.frequency(0)})"


class ChannelTracker(Tracker):
    """Class to track the phase and frequency of channels when resolving frames."""

    def __init__(self, channel: PulseChannel, frequency: float, sample_duration: float):
        """
        Args:
            channel: The channel that this tracker tracks.
            frequency: The starting frequency of the channel tracker.
            sample_duration: The sample duration on the backend.
        """
        super().__init__(channel.name, sample_duration)
        self._channel = channel
        self._frequencies_phases = [TimeFrequencyPhase(time=0, frequency=frequency, phase=0.0)]

    @property
    def frame(self) -> Frame:
        """Return the native frame of this channel."""
        return self._channel.frame
