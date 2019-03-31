# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# TODO: pylint
# pylint: disable=missing-docstring, missing-param-doc, missing-raises-doc

"""
Schedule.
"""
import logging
from typing import List

from qiskit.pulse.common.interfaces import Instruction, TimedInstruction
from qiskit.pulse.common.timeslots import TimeslotOccupancy
from qiskit.pulse.exceptions import ScheduleError

logger = logging.getLogger(__name__)


class SubSchedule(TimedInstruction):
    """An instruction block with begin time relative to its parent,
    which is a non-root node in a schedule tree."""

    def __init__(self, begin_time: int, block: Instruction):
        self._begin_time = begin_time
        self._block = block

    @property
    def children(self) -> List[TimedInstruction]:
        """Child nodes of this schedule node. """
        if isinstance(self._block, Schedule):
            return self._block.children
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


class Schedule(Instruction, TimedInstruction):
    """Schedule of pulses with timing. The root of a schedule tree."""

    def __init__(self,
                 schedules: List[TimedInstruction] = None,
                 name: str = None
                 ):
        """Create schedule.

        Args:
            schedules:
            name:
        """
        self._occupancy = TimeslotOccupancy(timeslots=[])
        self._children = []
        for timed in schedules or []:
            if isinstance(timed, TimedInstruction):
                self.insert(timed.begin_time, timed.instruction)
            else:
                raise ScheduleError("Invalid to be scheduled: %s" % timed.__class__.__name__)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def insert(self, begin_time: int, block: Instruction):
        """Insert a new block at `begin_time`.

        Args:
            begin_time:
            block (Instruction):
        """
        if not isinstance(block, Instruction):
            raise ScheduleError("Invalid to be inserted: %s" % block.__class__.__name__)
        if block == self:
            raise ScheduleError("Cannot insert self to avoid infinite recursion")
        shifted = block.occupancy.shifted(begin_time)
        if self._occupancy.is_mergeable_with(shifted):
            self._occupancy = self._occupancy.merged(shifted)
            self._children.append(SubSchedule(begin_time, block))
        else:
            logger.warning("Fail to insert %s at %s due to timing overlap", block, begin_time)
            raise ScheduleError("Fail to insert %s at %s due to overlap" % (str(block), begin_time))

    def append(self, block: Instruction):
        """Append a new block on a channel at the timing
        just after the last block finishes.

        Args:
            block (Instruction):
        """
        if not isinstance(block, Instruction):
            raise ScheduleError("Invalid to be inserted: %s" % block.__class__.__name__)
        if block == self:
            raise ScheduleError("Cannot insert self to avoid infinite recursion")
        begin_time = self.end_time
        try:
            self.insert(begin_time, block)
        except ScheduleError:
            logger.warning("Fail to append %s due to timing overlap", block)
            raise ScheduleError("Fail to append %s due to overlap" % str(block))

    @property
    def children(self) -> List[TimedInstruction]:
        """Child nodes of this schedule node. """
        return self._children

    @property
    def instruction(self) -> Instruction:
        """Instruction. """
        return self

    @property
    def begin_time(self) -> int:
        return 0

    @property
    def end_time(self) -> int:
        return max([slot.interval.end for slot in self._occupancy.timeslots], default=0)

    @property
    def duration(self) -> int:
        return self.end_time

    @property
    def occupancy(self) -> TimeslotOccupancy:
        return self._occupancy

    def __str__(self):
        # TODO: Handle schedule of schedules
        for child in self._children:
            if child.children:
                raise NotImplementedError("This version doesn't support schedule of schedules.")
        res = []
        for child in self._children:
            res.append((child.begin_time, str(child.instruction)))
        res = sorted(res)
        return '\n'.join(["%4d: %s" % i for i in res])

    def flat_instruction_sequence(self) -> List[SubSchedule]:
        return [_ for _ in Schedule._flatten_generator(self, self.begin_time)]

    @staticmethod
    def _flatten_generator(node: TimedInstruction, time: int):
        if node.children:
            for child in node.children:
                yield from Schedule._flatten_generator(child, time + node.begin_time)
        else:
            yield SubSchedule(time + node.begin_time, node.instruction)
