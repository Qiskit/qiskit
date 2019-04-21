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
from typing import List

from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.commands import Instruction
from qiskit.pulse.exceptions import PulseError

from .pulse_schedule import Schedule

logger = logging.getLogger(__name__)


def union(self, *schedules: List[ScheduleComponent], name: str = None) -> Schedule:
    """Create a union of all input `Schedule`s.

    Args:
        *schedules: Schedules to take the union of.
        name (str, optional): Name of this schedule. Defaults to None.
    """
    return Schedule(*schedules, name=name)


def shifted(self, shift: int) -> ScheduleComponent:
    news = copy(self)
    news._start_time += shift
    news._occupancy = self._occupancy.shifted(shift)
    return news


def insert(parent: ScheduleComponent, start_time: int, child: ScheduleComponent) -> Schedule:
    """Return a new schedule with the `child` schedule inserted into the `parent` at `start_time`.

    Args:
        parent: Schedule to be inserted into.
        start_time (int): time to be inserted defined with respect to `parent`.
        schedule (ScheduleComponent): schedule to be inserted

    Returns:
        Schedule: The new inserted `Schedule`
    """
    return union(parent, shifted(child, start_time))


def append(parent: ScheduleComponent, child: ScheduleComponent) -> Schedule:
    """Return a new schedule with by appending `child` to `parent` at
       the last time of the `parent` schedule's channels (+ buffer)
       over the intersection of the parent and child schedule's channels.

       $t = buffer + \textrm{max}({x.stop\_time |x \in parent.channels \cap child.channels})$

    Args:
        schedule (ScheduleComponent): schedule to be appended

    Returns:
        Schedule: a new schedule appended a `schedule`

    Raises:
        PulseError: when an invalid schedule is specified or failed to append
    """


def flatten_generator(node: ScheduleComponent, time: int = 0):
    if isinstance(node, Schedule):
        for child in node.children:
            yield from flatten_generator(child, time + node.t0)
    elif isinstance(node, Instruction):
        yield node.shifted(time)
    else:
        raise PulseError("Unknown ScheduleComponent type: %s" % node.__class__.__name__)
