# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
from abc import ABCMeta, abstractmethod
from typing import List, Set, Union

from qiskit.pulse.channels import PulseChannel, OutputChannel, AcquireChannel, SnapshotChannel
from qiskit.pulse.commands import PulseCommand


class TimedPulseBlock(metaclass=ABCMeta):
    """
    Common interface of TimedPulse and PulseSchedule (Component in the Composite Pattern)."""

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
    def children(self) -> List['TimedPulseBlock']:
        pass


class TimedPulse(TimedPulseBlock):
    """TimedPulse = Pulse with start time context."""

    def __init__(self, pulse_command: PulseCommand, to_channel: PulseChannel, start_time: int):
        if isinstance(pulse_command, to_channel.supported):  # TODO: refactoring
            self.command = pulse_command
            self.channel = to_channel
            self.t0 = start_time
        else:
            raise Exception(
                "Not supported commands on the channel")  # TODO need to make PulseException class

    def start_time(self) -> int:
        return self.t0

    def end_time(self) -> int:
        return self.t0 + self.command.duration

    def duration(self) -> int:
        return self.command.duration

    def children(self) -> List[TimedPulseBlock]:
        return None


class PulseSchedule(TimedPulseBlock):
    """Schedule."""

    def __init__(self,
                 output_channels: List[OutputChannel] = None,
                 acquire_channels: List[AcquireChannel] = None,
                 snapshot_channels: List[SnapshotChannel] = None,
                 name: str = None
                 ):
        """Create empty schedule.

        Args:
            output_channels:
            acquire_channels:
            snapshot_channels:
            name:
        """
        self.name = name
        self._output_channels = output_channels
        self._acquire_channels = acquire_channels
        self._snapshot_channels = snapshot_channels
        self._children = []

    def add(self,
            commands: Union[PulseCommand, List[PulseCommand]],
            channel: PulseChannel,
            start_time: int) -> bool:
        """Add new pulse command(s) with channel and start time context.

        Args:
            commands (PulseCommand|list):
            channel:
            start_time:

        Returns:
            True if succeeded, otherwise False
        """
        if isinstance(commands, PulseCommand):
            return self.add_block(TimedPulse(commands, channel, start_time))
        elif isinstance(commands, list):
            for cmd in commands:
                success = self.add(cmd, channel, start_time)
                if not success:
                    return False
            return True

    def add_block(self, block: TimedPulseBlock) -> bool:
        """Add a new composite pulse `TimedPulseBlock`.

        Args:
            block:

        Returns:
            True if succeeded, otherwise False
        """
        if self._is_occupied_time(block):
            return False  # TODO: or raise Exception?
        else:
            self._children.append(block)
            return True

    def start_time(self) -> int:
        return min([self._start_time(child) for child in self._children])

    def end_time(self) -> int:
        raise max([self._end_time(child) for child in self._children])

    def duration(self) -> int:
        raise self.end_time() - self.start_time()

    def children(self) -> List[TimedPulseBlock]:
        return self._children

    def _start_time(self, block: TimedPulseBlock) -> int:
        if isinstance(block, TimedPulse):
            return block.start_time()
        else:
            return min([self._start_time(child) for child in block.children()])

    def _end_time(self, block: TimedPulseBlock) -> int:
        if isinstance(block, TimedPulse):
            return block.end_time()
        else:
            return max([self._end_time(child) for child in block.children()])

    def _is_occupied_time(self, timed_pulse) -> bool:
        # TODO: This is still a MVP, very very naive implementation
        if not isinstance(timed_pulse, TimedPulse):
            raise NotImplementedError()
        for pulse in self.flat_pulse_sequence():
            if pulse.channel == timed_pulse.channel:
                # interval check
                if pulse.start_time() <= timed_pulse.end_time() \
                        and timed_pulse.start_time() <= pulse.end_time():
                    return False
        return True

    def remove(self, timed_pulse: TimedPulseBlock):
        # TODO: This is still a MVP
        for child in self._children:
            if not isinstance(child, TimedPulse):
                raise NotImplementedError()
        self._children.remove(timed_pulse)

    def command_library(self) -> Set[PulseCommand]:
        # TODO: This is still a MVP
        for child in self._children:
            if not isinstance(child, TimedPulse):
                raise NotImplementedError()
        return {tp.command for tp in self._children}

    def flat_pulse_sequence(self) -> List[TimedPulse]:
        # TODO: This is still a MVP
        for child in self._children:
            if not isinstance(child, TimedPulse):
                raise NotImplementedError()
        return self._children
