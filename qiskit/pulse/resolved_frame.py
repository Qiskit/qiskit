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
from typing import Dict, List, Union

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
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

    def __init__(self, index: Union[int, Parameter]):
        """
        Args:
            index: The index of the Tracker. Corresponds to the index of a
            :class:`~qiskit.pulse.Frame` or a channel.
        """
        self._index = index
        self._frequencies = []
        self._phases = []
        self._instructions = {}

        self._parameters = set()
        if isinstance(index, ParameterExpression):
            self._parameters.update(index.parameters)

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
            frequency: The frequency of the frame up until time.
        """
        frequency = self._frequencies[0][1]
        for time_freq in self._frequencies:
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
        phase = self._phases[0][1]
        for time_freq in self._phases:
            if time_freq[0] <= time:
                phase = time_freq[1]
            else:
                break

        return phase

    def set_frequency(self, time: int, frequency: float):
        """Insert a new frequency in the time-ordered frequencies."""

        if time > 0 and time in set([_[0] for _ in self._frequencies]):
            if self.frequency(time) != frequency:
                raise PulseError(f'Frequency already added at time {time}.')

        insert_idx = 0
        for idx, time_freq in enumerate(self._frequencies):
            if time_freq[0] < time:
                insert_idx = idx

        self._frequencies.insert(insert_idx + 1, (time, frequency))

    def set_phase(self, time: int, phase: float):
        """Insert a new phase in the time-ordered phases."""

        if time > 0 and time in set([_[0] for _ in self._phases]):
            if self.phase(time) != phase:
                raise PulseError(f'Phase already added at time {time} for Frame({self.index}).')

        insert_idx = 0
        for idx, time_freq in enumerate(self._phases):
            if time_freq[0] < time:
                insert_idx = idx

        self._phases.insert(insert_idx + 1, (time, phase))


class ResolvedFrame(Tracker):
    """
    A class to help the assembler determine the frequency and phase of a
    Frame at any given point in time.
    """

    def __init__(self, frame: Frame, frequency: float, phase: float,
                 channels: List[Channel]):
        """
        Args:
            frame: The frame to track.
            frequency: The initial frequency of the frame.
            phase: The initial phase of the frame.
            channels: The list of channels on which the frame instructions apply.
        """
        super().__init__(frame.index)
        self._frequencies = [(0, frequency)]
        self._phases = [(0, phase)]
        self._channels = channels

        for ch in self._channels:
            self._parameters.update(ch.parameters)

    @property
    def channels(self) -> List[Channel]:
        """Returns the channels that this frame ties together."""
        return self._channels

    def set_frame_instructions(self, schedule: Schedule):
        """
        Add all matching frame instructions in this schedule to self.

        Args:
            schedule: The schedule from which to extract frame operations.
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

    def assign(self, parameter: Parameter, value: ParameterValueType) -> 'ResolvedFrame':
        """
        Override the base class's assign method to handle any links between the
        parameter of the frame and the parameters of the sub-channels.

        Args:
            parameter: The parameter to assign
            value: The value of the parameter.
        """
        return self.assign_parameters({parameter: value})

    def assign_parameters(self,
                          value_dict: Dict[ParameterExpression, ParameterValueType]
                          ) -> 'ResolvedFrame':
        """
        Assign the value of the parameters.

        Args:
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.
        """
        assigned_sub_channels = self._assign_sub_channels(value_dict)

        new_index = None
        if isinstance(self._index, ParameterExpression):
            for param, value in value_dict.items():
                if param in self._index.parameters:
                    new_index = self._index.assign(param, value)
                    if not new_index.parameters:
                        new_index = int(new_index)

        if new_index is not None:
            return type(self)(new_index, self._frequencies, self._phases, assigned_sub_channels)

        return type(self)(self._index, self._frequencies, self._phases, assigned_sub_channels)

    def _assign_sub_channels(self, value_dict: Dict[ParameterExpression,
                                                    ParameterValueType]) -> List['Channel']:
        """
        Args:
            value_dict: The keys are the parameters to assign and the values are the
                values of the parameters.

        Returns:
             Frame: A Frame in which the parameter has been assigned.
        """
        sub_channels = []
        for ch in self._channels:
            if isinstance(ch.index, ParameterExpression):
                for param, value in value_dict.items():
                    if param in ch.parameters:
                        ch = ch.assign(param, value)

            sub_channels.append(ch)

        return sub_channels

    def __repr__(self):
        sub_str = '[' + ', '.join([ch.__repr__() for ch in self._channels]) + ']'
        return f'{self.__class__.__name__}({self._index}, {sub_str})'


class ChannelTracker(Tracker):
    """Class to track the phase and frequency of channels when resolving frames."""

    def __init__(self, channel: PulseChannel):
        """
        Args:
            channel: The channel that this tracker tracks.
        """
        super().__init__(channel.index)
        self._channel = channel
        self._frequencies = []
        self._phases = []

    def is_initialized(self) -> bool:
        """Return true if the channel has been initialized."""
        return len(self._frequencies) > 0
