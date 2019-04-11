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
from qiskit.pulse.common.timeslots import TimeslotCollection
from .pulse_command import PulseCommand

logger = logging.getLogger(__name__)


class Instruction(ScheduleComponent):
    """An abstract class for leaf nodes of schedule."""

    def __init__(self, command: PulseCommand, start_time: int, occupancy: TimeslotCollection):
        self._command = command
        self._start_time = start_time
        self._occupancy = occupancy

    @property
    def duration(self) -> int:
        """Duration of this instruction. """
        return self._command.duration

    @property
    def occupancy(self) -> TimeslotCollection:
        """Occupied time slots by this instruction. """
        return self._occupancy

    def shifted(self, shift: int) -> ScheduleComponent:
        news = copy(self)
        news._start_time += shift
        news._occupancy = self._occupancy.shifted(shift)
        return news

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction. """
        return self._start_time

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction. """
        return self.start_time + self.duration

    @property
    def children(self) -> Tuple[ScheduleComponent, ...]:
        """Instruction has no child nodes. """
        return ()

    def __repr__(self):
        return "%4d: %s" % (self._start_time, self._command)
