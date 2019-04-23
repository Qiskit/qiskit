# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction = Leaf node of schedule.
"""
import logging
from typing import Tuple, List, Iterable
from qiskit.pulse import ops
from qiskit.pulse.channels import Channel
from qiskit.pulse.interfaces import ScheduleComponent
from qiskit.pulse.timeslots import TimeslotCollection

from .pulse_command import PulseCommand

logger = logging.getLogger(__name__)


class Instruction(ScheduleComponent):
    """An abstract class for leaf nodes of schedule."""

    def __init__(self, command: PulseCommand, timeslots: TimeslotCollection, name: str = None):
        self._name = name
        self._command = command
        self._timeslots = timeslots

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self._name

    @property
    def command(self) -> PulseCommand:
        """Acquire command. """
        return self._command

    @property
    def channels(self):
        """Returns channels that this schedule uses.

        Returns:
            Tuple
        """
        return self.timeslots.channels

    @property
    def timeslots(self) -> TimeslotCollection:
        """Occupied time slots by this instruction. """
        return self._timeslots

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction. """
        return self.timeslots.start_time

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction. """
        return self.timeslots.stop_time

    @property
    def duration(self) -> int:
        """Duration of this instruction. """
        return self.timeslots.duration

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        """Instruction has no child nodes. """
        return ()

    def ch_duration(self, *channels: List[Channel]) -> int:
        """Return start time on this schedule or channel.

        Args:
            channels: Supplied channels

        Returns:
            The latest stop time in this collection.
        """
        return self.timeslots.ch_start_time(*channels)

    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Return minimum start time for supplied channels.

        Args:
            channels: Supplied channels

        Returns:
            The latest stop time in this collection.
        """
        return self.timeslots.ch_start_time(*channels)

    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Return maximum start time for supplied channels.

        Args:
            channels: Supplied channels

        Returns:
            The latest stop time in this collection.
        """
        return self.timeslots.ch_stop_time(*channels)

    def union(self, *schedules: List[ScheduleComponent]):
        """Return a new schedule which is the union of `self` and `schedule`.

        Args:
            schedules: Schedules to be take the union with the parent `Schedule`.

        Returns:
            Schedule
        """
        return ops.union(self, *schedules)

    def shift(self: ScheduleComponent, time: int):
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by

        Returns:
            Schedule
        """
        return ops.shift(self, time)

    def insert(self, start_time: int, schedule: ScheduleComponent):
        """Return a new schedule with `schedule` inserted within `self` at `start_time`.

        Args:
            start_time: time to be inserted
            schedule: schedule to be inserted

        Returns:
            Schedule
        """
        return ops.insert(self, start_time, schedule)

    def append(self, schedule: ScheduleComponent):
        """Return a new schedule with `schedule` inserted at the maximum time over
        all channels shared between `self` and `schedule`.

        Args:
            schedule: schedule to be appended

        Returns:
            Schedule
        """
        return ops.append(self, schedule)

    def flatten(self, time: int = 0) -> Iterable[Tuple[int, ScheduleComponent]]:
        """Iterable for flattening Schedule tree.

        Args:
            node: Root of Schedule tree to traverse
            time: Shifted time of this node due to parent
        """
        yield (time, self)

    def __add__(self, schedule: ScheduleComponent):
        """Return a new schedule with `schedule` inserted within `self` at `start_time`.

        Returns:
            Schedule
        """
        return self.compose(schedule)

    def __or__(self, schedule: ScheduleComponent):
        """Return a new schedule which is the union of `self` and `schedule`.

        Returns:
            Schedule
        """
        return self.union(schedule)

    def __lshift__(self, time: int):
        """Return a new schedule which is shifted forward by `time`.

        Returns:
            Schedule
        """
        return self.shift(time)

    def __rshift__(self, time: int):
        """Return a new schedule which is shifted backwards by `time`.

        Returns:
            Schedule
        """
        return self.shift(-time)

    def __repr__(self):
        return "%s" % (self._command)
