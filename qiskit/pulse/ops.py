# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule operations.
"""
import logging
from copy import copy
from functools import reduce
from typing import List, Tuple

from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError

from .pulse_schedule import Schedule

logger = logging.getLogger(__name__)


def _union(sched1: ScheduleComponent, sched2: ScheduleComponent) ->Schedule:
    """Take the union of two `ScheduleComponent`s."""

    return Schedule(sched1, sched2)
    if sched1.occupancy.is_mergeable_with(sched2):
        self._occupancy = self._occupancy.merged(shifted)
        self._children += (schedule.shifted(start_time),)
    else:
        raise PulseError("Fail to insert %s at %s due to overlap" % (str(schedule), start_time))


def union(self, *schedules: List[ScheduleComponent], name: str = None) -> Schedule:
    """Create a union of all input `Schedule`s.

    Args:
        *schedules: Schedules to take the union of.
        name (str, optional): Name of this schedule. Defaults to None.
    """
    return Schedule(*schedules, name=name)


def insert(self, parent: ScheduleComponent, start_time: int, child: ScheduleComponent) -> Schedule:
    """Return a new schedule with the `child` schedule inserted into the `parent` at `start_time`.

    Args:
        parent: Schedule to be inserted into.
        start_time (int): time to be inserted defined with respect to `parent`.
        schedule (ScheduleComponent): schedule to be inserted

    Returns:
        Schedule: The new inserted `Schedule`

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


def append(self, schedule: ScheduleComponent) -> Schedule:
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


def shifted(self, shift: int) -> ScheduleComponent:
    news = copy(self)
    news._start_time += shift
    news._occupancy = self._occupancy.shifted(shift)
    return news
