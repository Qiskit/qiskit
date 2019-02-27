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
        self.command = pulse_command
        self.channel = to_channel
        self.start_time = start_time

    def start_time(self) -> int:
        return self.start_time

    def end_time(self) -> int:
        return self.start_time + self.command.duration

    def duration(self) -> int:
        return self.command.duration

    def channels(self) -> Set[PulseChannel]:
        return set([self.channel])

    def children(self) -> List[TimedPulseBlock]:
        return None


class PulseSchedule(TimedPulseBlock):
    """Schedule."""

    def __init__(self, channels: List[PulseChannel]):
        """Create empty schedule.
        Args:
            channels:
        """
        self._channels = channels

    def add(self, timed_pulse: TimedPulse) -> bool:
        """Add `pulse_command` to `to_channel` at `start_time`.
        Args:
            pulse_command:
            to_channel:
            start_time:

        Returns:
            An added pulse with channel and time context.
        """
        raise NotImplementedError()

    # def add(self, pulse_command: PulseCommand, to_channel: Channel, start_time: int) -> TimedPulse:
    #     """Add `pulse_command` to `to_channel` at `start_time`.
    #     Args:
    #         pulse_command:
    #         to_channel:
    #         start_time:
    #
    #     Returns:
    #         An added pulse with channel and time context.
    #     """
    #     pass

    def remove(self, timed_pulse: TimedPulse) -> bool:
        raise NotImplementedError()

    def start_time(self) -> int:
        raise NotImplementedError()

    def end_time(self) -> int:
        raise NotImplementedError()

    def duration(self) -> int:
        raise NotImplementedError()

    def channels(self) -> Set[PulseChannel]:
        raise NotImplementedError()

    def children(self) -> List[TimedPulseBlock]:
        raise NotImplementedError()

    def qobj(self) -> dict:
        """Create qobj.
        Returns:

        """
        raise NotImplementedError()
