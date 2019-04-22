# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction = Leaf node of schedule.
"""
import logging
from typing import Tuple, List

from qiskit.pulse import ops
from qiskit.pulse import Schedule
from qiskit.pulse.channels import Channel
from qiskit.pulse.common.interfaces import ScheduleComponent
from qiskit.pulse.common.timeslots import TimeslotCollection
from .pulse_command import PulseCommand

logger = logging.getLogger(__name__)


class Instruction(ScheduleComponent):
    """An abstract class for leaf nodes of schedule."""

    def __init__(self, command: PulseCommand, start_time: int, timeslots: TimeslotCollection):
        self._command = command
        self._timeslots = timeslots

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
    def timeslots(self) -> TimeslotCollection:
        """Occupied time slots by this instruction. """
        return self._timeslots

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

    def shift(self, time: int) -> Schedule:
        return ops.shift(self, time)

    def __repr__(self):
        return "%s" % (self._command)
