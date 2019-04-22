# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
import itertools
import logging
from operator import attrgetter
from typing import List, Tuple

from qiskit.pulse import ops
from qiskit.pulse.channels import Channel
from qiskit.pulse.commands import Instruction
from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)


class Schedule(ScheduleComponent):
    """Schedule of `ScheduleComponent`s. The composite node of a schedule tree."""

    def __init__(self, *schedules, name: str = None):
        """Create empty schedule.

        Args:
            schedules (List[Union[ScheduleComponent,
                       Tuple[ScheduleComponent, int]]): Child Schedules of this parent Schedule.
                       Each element is either a ScheduleComponent, or a tuple pair of the form
                       (ScheduleComponent, t0) where t0 (int) is the starting time of the
                       `ScheduleComponent`.
            name (str, optional): Name of this schedule. Defaults to None.

        Raises:
            PulseError: If timeslots intercept.
        """
        self._name = name

        try:
            timeslots = []
            for sched in schedules:
                if isinstance(sched, (list, tuple)):
                    timeslots.append(sched[0].timeslots.shift(sched[1]))

            self._timeslots = TimeslotCollection(*itertools.chain(
                                *[sched.timeslots for sched in timeslots]))
        except PulseError as ts_err:
            raise PulseError('Child schedules {} overlap.'.format(
                    [sched.name for sched in schedules])) from ts_err
        self._children = tuple(*zip(schedules, timeslots))

    @property
    def name(self) -> str:
        """Name of this schedule.

        Returns:
            str
        """
        return self._name

    @property
    def duration(self) -> int:
        return self.timeslots.duration

    @property
    def start_time(self) -> int:
        return self.timeslots.start_time

    @property
    def stop_time(self) -> int:
        return self.timeslots.stop_time

    @property
    def timeslots(self) -> TimeslotCollection:
        return self._timeslots

    @property
    def channels(self):
        """Returns channels that this schedule uses.

        Returns:
            Tuple
        """
        return self._timeslots.channels

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        return self._children

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

    def union(self, *schedules: List[ScheduleComponent]) -> 'Schedule':
        """Return a new schedule which is the union of the parent `Schedule` and `schedule`.

        Args:
            schedules: Schedules to be take the union with the parent `Schedule`.

        Returns:
            Schedule: a new schedule with `schedule` inserted at `start_time`

        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        return ops.union(self, *schedules)

    def insert(self, start_time: int, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with `schedule` inserted within the parent `Schedule` at `start_time`.

        Args:
            start_time (int): time to be inserted
            schedule (ScheduleComponent): schedule to be inserted

        Returns:
            Schedule: a new schedule with `schedule` inserted at `start_time`

        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        return ops.insert(self, start_time, schedule)

    def append(self, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with `schedule` inserted at the maximum time over
        all channels shared between `self` and `schedule`.

        Args:
            schedule (ScheduleComponent): schedule to be appended

        Returns:
            Schedule: a new schedule appended a `schedule`

        Raises:
            PulseError: when an invalid schedule is specified or failed to append
        """
        return ops.append(self, schedule)

    def flat_instruction_sequence(self) -> List[Instruction]:
        """Return instruction sequence of this schedule.
        Each instruction has absolute start time.
        """
        return list(ops.flatten_generator(self))

    def __add__(self, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with `schedule` inserted within the parent `Schedule` at `start_time`."""
        return self.compose(schedule)

    def __or__(self, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule which is the union of the parent `Schedule` and `schedule`."""
        return self.union(schedule)

    def __lshift__(self, time: int) -> 'Schedule':
        """Return a new schedule which is the parent `Schedule` shifted forward by `time`."""
        return self.shift(time)

    def __rshift__(self, time: int) -> 'Schedule':
        """Return a new schedule which is the parent `Schedule` shifted backwards by `time`."""
        return self.shift(-time)

    def __str__(self):
        res = "Schedule(%s):\n" % (self._name or "")
        instructions = sorted(self.flat_instruction_sequence(), key=attrgetter("start_time"))
        res += '\n'.join([str(i) for i in instructions])
        return res
