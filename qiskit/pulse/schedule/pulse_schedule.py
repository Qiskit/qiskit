# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule.
"""
import logging
from copy import copy
from typing import List, Tuple

from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)


class Schedule(ScheduleComponent):
    """Schedule of instructions. The composite node of a schedule tree."""

    def __init__(self, name: str = None, start_time: int = 0):
        """Create empty schedule.

        Args:
            name (str, optional): Name of this schedule. Defaults to None.
            start_time (int, optional): Begin time of this schedule. Defaults to 0.
        """
        self._name = name
        self._start_time = start_time
        self._occupancy = TimeslotCollection(timeslots=[])
        self._children = ()

    @property
    def name(self) -> str:
        """Name of this schedule."""
        return self._name

    def insert(self, start_time: int, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with inserting a `schedule` at `start_time`.

        Args:
            start_time (int): time to be inserted
            schedule (ScheduleComponent): schedule to be inserted

        Returns:
            Schedule: a new schedule inserted a `schedule` at `start_time`

        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        if not isinstance(schedule, ScheduleComponent):
            raise PulseError("Invalid to be inserted: %s" % schedule.__class__.__name__)
        news = copy(self)
        try:
            news._insert(start_time, schedule)
        except PulseError as err:
            raise PulseError(err.message)
        return news

    def _insert(self, start_time: int, schedule: ScheduleComponent):
        """Insert a new `schedule` at `start_time`.
        Args:
            start_time (int): start time of the schedule
            schedule (ScheduleComponent): schedule to be inserted
        Raises:
            PulseError: when an invalid schedule is specified or failed to insert
        """
        if schedule == self:
            raise PulseError("Cannot insert self to avoid infinite recursion")
        shifted = schedule.occupancy.shifted(start_time)
        if self._occupancy.is_mergeable_with(shifted):
            self._occupancy = self._occupancy.merged(shifted)
            self._children += (schedule.shifted(start_time),)
        else:
            logger.warning("Fail to insert %s at %s due to timing overlap", schedule, start_time)
            raise PulseError("Fail to insert %s at %s due to overlap" % (str(schedule), start_time))

    def append(self, schedule: ScheduleComponent) -> 'Schedule':
        """Return a new schedule with appending a `schedule` at the timing
        just after the last instruction finishes.

        Args:
            schedule (ScheduleComponent): schedule to be appended

        Returns:
            Schedule: a new schedule appended a `schedule`

        Raises:
            PulseError: when an invalid schedule is specified or failed to append
        """
        if not isinstance(schedule, ScheduleComponent):
            raise PulseError("Invalid to be appended: %s" % schedule.__class__.__name__)
        news = copy(self)
        try:
            news._insert(self.stop_time, schedule)
        except PulseError:
            logger.warning("Fail to append %s due to timing overlap", schedule)
            raise PulseError("Fail to append %s due to overlap" % str(schedule))
        return news

    @property
    def duration(self) -> int:
        return self.stop_time - self.start_time

    @property
    def occupancy(self) -> TimeslotCollection:
        return self._occupancy

    def shifted(self, shift: int) -> ScheduleComponent:
        news = copy(self)
        news._start_time += shift
        news._occupancy = self._occupancy.shifted(shift)
        return news

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

    def __add__(self, schedule: ScheduleComponent):
        return self.append(schedule)

    def __or__(self, schedule: ScheduleComponent):
        return self.insert(0, schedule)

    def __str__(self):
        # TODO: Handle schedule of schedules
        for child in self._children:
            if child.children:
                raise NotImplementedError("This version doesn't support schedule of schedules.")
        return '\n'.join([str(child) for child in self._children])

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
