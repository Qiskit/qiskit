# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Command schedule.
"""
import logging
from abc import abstractmethod
from typing import List

from .interfaces import Instruction, TimedInstruction
from .timeslots import TimeslotOccupancy

logger = logging.getLogger(__name__)


class PrimitiveInstruction(Instruction):
    """Abstract class for primitive instructions. """

    @property
    @abstractmethod
    def duration(self) -> int:
        """Duration of this instruction. """
        pass

    @property
    @abstractmethod
    def occupancy(self) -> TimeslotOccupancy:
        """Occupied time slots by this instruction. """
        pass

    def at(self, begin_time: int):
        """Return timed instruction. """
        return CommandSchedule(begin_time, self)


class CommandSchedule(TimedInstruction):
    """An instruction with begin time relative to its parent,
    which is a leaf node in a schedule tree."""

    def __init__(self, begin_time: int, block: Instruction):
        self._begin_time = begin_time
        self._block = block

    @property
    def children(self) -> List[TimedInstruction]:
        """Child nodes of this schedule node. """
        return []

    @property
    def begin_time(self) -> int:
        """Relative begin time of this instruction block. """
        return self._begin_time

    @property
    def instruction(self) -> Instruction:
        return self._block

    @property
    def end_time(self) -> int:
        """Relative end time of this instruction block. """
        return self.begin_time + self.duration

    @property
    def duration(self) -> int:
        """Duration of this instruction block. """
        return self._block.duration

    def __repr__(self):
        return "(%d, %s)" % (self._begin_time, self._block)
