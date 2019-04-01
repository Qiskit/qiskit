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
import copy
import logging
from typing import List

from qiskit.pulse.common.command_schedule import CommandSchedule, PrimitiveInstruction
from qiskit.pulse.common.interfaces import Instruction, TimedInstruction
from qiskit.pulse.common.timeslots import TimeslotOccupancy
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)


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
        for block in schedules or []:
            if isinstance(block, TimedInstruction):
                self._insert(block.begin_time, block.instruction)
            else:
                raise PulseError("Invalid to be scheduled: %s" % block.__class__.__name__)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def inserted(self, block: TimedInstruction):
        """Return a new schedule with inserting an instruction with timing.

        Args:
            block (TimedInstruction): instruction with timing to be inserted
        """
        if not isinstance(block, TimedInstruction):
            raise PulseError("Invalid to be inserted: %s" % block.__class__.__name__)
        news = copy.copy(self)
        try:
            news._insert(block.begin_time, block.instruction)
        except PulseError as err:
            raise PulseError(err.message)
        return news

    def _insert(self, begin_time: int, inst: Instruction):
        """Insert a new instruction at `begin_time`.

        Args:
            begin_time:
            inst (Instruction):
        """
        if inst == self:
            raise PulseError("Cannot insert self to avoid infinite recursion")
        shifted = inst.occupancy.shifted(begin_time)
        if self._occupancy.is_mergeable_with(shifted):
            self._occupancy = self._occupancy.merged(shifted)
            if isinstance(inst, PrimitiveInstruction):
                self._children.append(CommandSchedule(begin_time, inst))
            elif isinstance(inst, Schedule):
                self._children.append(SubSchedule(begin_time, inst))
            else:
                raise PulseError("Invalid instruction to be inserted: %s" % inst.__class__.__name__)
        else:
            logger.warning("Fail to insert %s at %s due to timing overlap", inst, begin_time)
            raise PulseError("Fail to insert %s at %s due to overlap" % (str(inst), begin_time))

    def appended(self, instruction: Instruction):
        """Return a new schedule with appending an instruction with timing
        just after the last instruction finishes.

        Args:
            instruction (Instruction):
        """
        if not isinstance(instruction, Instruction):
            raise PulseError("Invalid to be appended: %s" % instruction.__class__.__name__)
        news = copy.copy(self)
        try:
            news._insert(self.end_time, instruction)
        except PulseError:
            logger.warning("Fail to append %s due to timing overlap", instruction)
            raise PulseError("Fail to append %s due to overlap" % str(instruction))
        return news

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

    def flat_instruction_sequence(self) -> List[CommandSchedule]:
        return [_ for _ in Schedule._flatten_generator(self, self.begin_time)]

    @staticmethod
    def _flatten_generator(node: TimedInstruction, time: int):
        if node.children:
            for child in node.children:
                yield from Schedule._flatten_generator(child, time + node.begin_time)
        else:
            yield CommandSchedule(time + node.begin_time, node.instruction)


class SubSchedule(TimedInstruction):
    """An instruction block with begin time relative to its parent,
    which is a non-root node in a schedule tree."""

    def __init__(self, begin_time: int, schedule: Schedule):
        self._begin_time = begin_time
        self._block = schedule

    @property
    def children(self) -> List[TimedInstruction]:
        """Child nodes of this schedule node. """
        return self._block.children

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
