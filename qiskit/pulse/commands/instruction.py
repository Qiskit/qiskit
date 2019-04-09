# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction = Leaf node of schedule.
"""
import logging
from copy import copy
from typing import Tuple

from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotOccupancy
from .pulse_command import PulseCommand

logger = logging.getLogger(__name__)


class Instruction(ScheduleComponent):
    """An abstract class for leaf nodes of schedule."""

    def __init__(self, command: PulseCommand, begin_time: int, occupancy: TimeslotOccupancy):
        self._command = command
        self._begin_time = begin_time
        self._occupancy = occupancy

    @property
    def duration(self) -> int:
        """Duration of this instruction. """
        return self._command.duration

    @property
    def occupancy(self) -> TimeslotOccupancy:
        """Occupied time slots by this instruction. """
        return self._occupancy

    def shifted(self, shift: int) -> ScheduleComponent:
        news = copy(self)
        news._begin_time += shift
        news._occupancy = self._occupancy.shifted(shift)
        return news

    @property
    def begin_time(self) -> int:
        """Relative begin time of this instruction. """
        return self._begin_time

    @property
    def end_time(self) -> int:
        """Relative end time of this instruction. """
        return self.begin_time + self.duration

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        """Instruction has no child nodes. """
        return ()

    def __repr__(self):
        return "%4d: %s" % (self._begin_time, self._command)

    def __lshift__(self, shift: int) -> ScheduleComponent:
        return self.shifted(shift)
