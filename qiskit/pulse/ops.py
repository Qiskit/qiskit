# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Schedule operations.
"""
import logging
from typing import List, Union, Tuple

from .interfaces import ScheduleComponent
from .schedule import Schedule

logger = logging.getLogger(__name__)

# pylint: disable=missing-return-doc,missing-type-doc


def union(*schedules: List[Union[ScheduleComponent, Tuple[int, ScheduleComponent]]],
          name: str = None) -> Schedule:
    """Create a union (and also shift if desired) of all input `Schedule`s.

    Args:
        *schedules: Schedules to take the union of
        name: Name of the new schedule. Defaults to first element of `schedules`
    """
    if name is None and schedules:
        sched = schedules[0]
        if isinstance(sched, (list, tuple)):
            name = sched[1].name
        else:
            name = sched.name
    return Schedule(*schedules, name=name)

# pylint: enable=missing-type-doc


def flatten(schedule: ScheduleComponent, name: str = None) -> Schedule:
    """Create a flattened schedule.

    Args:
        schedule: Schedules to flatten
        name: Name of the new schedule. Defaults to first element of `schedules`
    """
    if name is None:
        name = schedule.name

    return Schedule(*schedule.instructions, name=name)


def shift(schedule: ScheduleComponent, time: int, name: str = None) -> Schedule:
    """Return schedule shifted by `time`.

    Args:
        schedule: The schedule to shift
        time: The time to shift by
        name: Name of shifted schedule. Defaults to name of `schedule`
    """
    if name is None:
        name = schedule.name
    return union((time, schedule), name=name)


def insert(parent: ScheduleComponent, time: int, child: ScheduleComponent, buffer: bool = False,
           name: str = None) -> Schedule:
    """Return a new schedule with the `child` schedule inserted into the `parent` at `start_time`.

    Args:
        parent: Schedule to be inserted into
        time: Time to be inserted defined with respect to `parent`
        child: Schedule to insert
        buffer: Obey buffer when inserting
        name: Name of the new schedule. Defaults to name of parent
    """
    if buffer and child.buffer and time > 0:
        time += parent.buffer
    return union(parent, (time, child), name=name)


def append(parent: ScheduleComponent, child: ScheduleComponent, buffer: bool = True,
           name: str = None) -> Schedule:
    r"""Return a new schedule with by appending `child` to `parent` at
       the last time of the `parent` schedule's channels
       over the intersection of the parent and child schedule's channels.

       $t = \textrm{max}({x.stop\_time |x \in parent.channels \cap child.channels})$

    Args:
        parent: The schedule to be inserted into
        child: The schedule to insert
        buffer: Obey buffer when appending
        name: Name of the new schedule. Defaults to name of parent
    """
    common_channels = set(parent.channels) & set(child.channels)

    time = parent.ch_stop_time(*common_channels)
    if buffer and child.buffer and time > 0:
        time += parent.buffer

    return insert(parent, time, child, name=name)
