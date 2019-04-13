# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
import logging
from typing import List, Tuple

from qiskit.pulse import ops
from qiskit.pulse import Schedule
from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)


class Schedule(ScheduleComponent):
    """Schedule of instructions. The composite node of a schedule tree."""

    def __init__(self, *schedules, name: str = None):
        """Create empty schedule.

        Args:
            name (str, optional): Name of this schedule. Defaults to None.
            start_time (int, optional): Begin time of this schedule. Defaults to 0.

        Raises PulseError
        """
        self._name = name
        self._start_time = 0

        try:
            self._occupancy = TimeslotCollection([sched.occupancy for sched in schedules])
        except PulseError as ts_err:
            raise PulseError('Child schedules {} overlap.'.format(
                    [sched.name for sched in schedules])) from ts_err
        self._children = tuple(*schedules)

    @property
    def name(self) -> str:
        """Name of this schedule."""
        return self._name

    def union(self, *schedules: List[ScheduleComponent]) -> Schedule:
        """Return a new schedule which is the union of the parent `Schedule` and `schedule`.

        Args:
            schedules: Schedules to be take the union with the parent `Schedule`.

        Returns:
            Schedule: a new schedule with `schedule` inserted at `start_time`

        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        return ops.union(self, *schedules)

    def insert(self, start_time: int, schedule: ScheduleComponent) -> Schedule:
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

    @property
    def duration(self) -> int:
        return self.stop_time - self.start_time

    @property
    def occupancy(self) -> TimeslotCollection:
        return self._occupancy

    @property
    def start_time(self) -> int:
        return self._start_time

    @property
    def stop_time(self) -> int:
        return max([slot.interval.end for slot in self._occupancy.timeslots],
                   default=self._start_time)

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        return self._children

    def __add__(self, schedule: ScheduleComponent) -> Schedule:
        """Return a new schedule with `schedule` inserted within the parent `Schedule` at `start_time`."""
        return self.compose(schedule)

    def __or__(self, schedule: ScheduleComponent) -> Schedule:
        """Return a new schedule which is the union of the parent `Schedule` and `schedule`."""
        return self.union(schedule)

    def __lshift__(self, time: int) -> Schedule:
        """Return a new schedule which is the parent `Schedule` shifted forward by `time`."""
        return self.shifted(time)

    def __rshift__(self, time: int) -> Schedule:
        """Return a new schedule which is the parent `Schedule` shifted backwards by `time`."""
        return self.shifted(-time)

    def __str__(self):
        # TODO: Handle schedule of schedules

        child_strs = [str(child) for child in self.children]
        return 'Schedule(name={name},children={children})'.format(name=self.name,
                                                                  children=child_strs)

    def flat_instruction_sequence(self) -> List[ScheduleComponent]:
        """Return instruction sequence of this schedule.
        Each instruction has absolute start time.
        """
        if not self._children:  # empty schedule
            return []
        return [_ for _ in Schedule._flatten_generator(self, self.start_time)]

    @staticmethod
    def _flatten_generator(node: ScheduleComponent, time: int):
        if node.children:
            for child in node.children:
                yield from Schedule._flatten_generator(child, time + node.start_time)
        else:
            yield node.shifted(time)
