# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
from abc import ABCMeta, abstractmethod
from typing import List, Set

# from .pulse import Pulse
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.commands import PulseCommand


class TimedPulseBlock(metaclass=ABCMeta):
    """Common interface of TimedPulse and PulseSchedule (Component in the Composite Pattern)."""

    @abstractmethod
    def start_time(self) -> int:
        pass

    @abstractmethod
    def end_time(self) -> int:
        pass

    @abstractmethod
    def duration(self) -> int:
        pass

    @abstractmethod
    def channels(self) -> Set[PulseChannel]:
        pass

    @abstractmethod
    def children(self) -> List[TimedPulseBlock]:
        pass


class TimedPulse(TimedPulseBlock):
    """TimedPulse = Pulse with start time context."""

    def __init__(self, pulse_command: PulseCommand, to_channel: PulseChannel, start_time: int):
        if isinstance(pulse_command, to_channel.supported): #TODO: refactoring
            self.command = pulse_command
            self.channel = to_channel
            self.start_time = start_time
        else:
            raise Exception("Not supported commands on the channel") #TODO need to make PulseException class

    def start_time(self) -> int:
        return self.start_time

    def end_time(self) -> int:
        return self.start_time + self.command.duration

    def duration(self) -> int:
        return self.command.duration

    # def channels(self) -> Set[PulseChannel]:
    #     return set([self.channel])

    def children(self) -> List[TimedPulseBlock]:
        return None


class PulseSchedule(TimedPulseBlock):
    """Schedule."""

    def __init__(self, config):
        """Create empty schedule.
        Args:
            channels:
        """
        self._config = config   #TODO: refactring later
        self._children = []

    def add(self, timed_pulse: TimedPulseBlock) -> bool:
        """Add `timed_pulse` (pulse command with channel and start time context).
        Args:
            timed_pulse:

        Returns:
            An added pulse with channel and time context.
        """
        if self._is_occupied_time(timed_pulse):
            return False    #TODO: or raise Exception?
        else:
            self._children.append(timed_pulse)
            return True

    def remove(self, timed_pulse: TimedPulseBlock):
        self._children.remove(timed_pulse)

    def start_time(self) -> int:
        raise NotImplementedError()

    def end_time(self) -> int:
        raise NotImplementedError()

    def duration(self) -> int:
        raise NotImplementedError()

    # def channels(self) -> Set[PulseChannel]:
    #     raise NotImplementedError()

    def children(self) -> List[TimedPulseBlock]:
        return self._children

    def _flat_pulse_sequence(self) -> List[TimedPulse]:
        pass

    def qobj(self):
        """Create qobj.
        Returns:

        """
        pulses = self._flat_pulse_sequence()

        raise NotImplementedError()
