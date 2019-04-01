# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction = Command with its operands (Channels).
"""
from abc import ABCMeta, abstractmethod
from typing import List

from .timeslots import TimeslotOccupancy


class Instruction(metaclass=ABCMeta):
    """Common interface for `Command with its operands (Channels)`. """

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

    @abstractmethod
    def at(self, begin_time: int) -> 'TimedInstruction':
        """Return timed instruction. """
        pass


class ScheduleNode(metaclass=ABCMeta):
    """Common interface for nodes of a schedule tree. """

    @property
    @abstractmethod
    def children(self) -> List['TimedInstruction']:
        """Child nodes of this schedule node. """
        pass


class TimedInstruction(ScheduleNode):
    """Common interface for instruction with timing context. """

    @property
    @abstractmethod
    def instruction(self) -> Instruction:
        """Instruction. """
        pass

    @property
    @abstractmethod
    def begin_time(self) -> int:
        """Relative begin time of this instruction. """
        pass

    @property
    @abstractmethod
    def end_time(self) -> int:
        """Relative end time of this instruction. """
        pass
