# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Schedule operations.
"""
import logging
from typing import List

from .interfaces import ScheduleComponent
from .schedule import Schedule

logger = logging.getLogger(__name__)


def union(*schedules: List[ScheduleComponent], name: str = None) -> Schedule:
    """Create a union of all input `Schedule`s.

    Args:
        *schedules: Schedules to take the union of.
        name (str, optional): Name of this schedule. Defaults to None.
    """
    return Schedule(*schedules, name=name)


def shift(schedule: ScheduleComponent, time: int) -> Schedule:
    """Return schedule shifted by `time`.

    Args:
        schedule (ScheduleComponent): Schedule to shift
        time (int): Time to shift by
    """
    return Schedule(schedule, shift=time)


def insert(parent: ScheduleComponent, start_time: int, child: ScheduleComponent) -> Schedule:
    """Return a new schedule with the `child` schedule inserted into the `parent` at `start_time`.

    Args:
        parent: Schedule to be inserted into.
        start_time (int): time to be inserted defined with respect to `parent`.
        schedule (ScheduleComponent): schedule to be inserted
    """
    return union(parent, shift(child, start_time))


def append(parent: ScheduleComponent, child: ScheduleComponent) -> Schedule:
    """Return a new schedule with by appending `child` to `parent` at
       the last time of the `parent` schedule's channels
       over the intersection of the parent and child schedule's channels.

       $t = \textrm{max}({x.stop\_time |x \in parent.channels \cap child.channels})$

    Args:
        schedule (ScheduleComponent): schedule to be appended
    """
    common_channels = set(parent.channels) & set(child.channels)
    insertion_time = parent.ch_stop_time(*common_channels)
    return insert(parent, insertion_time, child)
