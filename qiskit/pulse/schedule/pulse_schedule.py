# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# TODO: Pylint
# pylint: disable=invalid-name, missing-docstring, missing-param-doc, missing-raises-doc

"""
Schedule.
"""
import logging
from typing import List, Tuple, Set

from qiskit.pulse.channels import DeviceSpecification, Channel
from qiskit.pulse.common.interfaces import Pulse, ScheduleNode
from qiskit.pulse.common.timeslots import TimeslotOccupancy
from qiskit.pulse.exceptions import ScheduleError

logger = logging.getLogger(__name__)


class TimedPulse(ScheduleNode):
    """A `Pulse` with begin time relative to its parent,
    which is a non-root node in a schedule tree."""

    def __init__(self, begin_time: int, block: Pulse):
        self._begin_time = begin_time
        self._block = block

    @property
    def pulse(self) -> Pulse:
        return self._block

    @property
    def children(self) -> List[ScheduleNode]:
        """Child nodes of this schedule node. """
        if isinstance(self._block, Schedule):
            return self._block.children
        return []

    @property
    def begin_time(self) -> int:
        """Relative begin time of this pulse. """
        return self._begin_time

    @property
    def end_time(self) -> int:
        """Relative end time of this pulse. """
        return self.begin_time + self.duration

    @property
    def duration(self) -> int:
        """Duration of this pulse. """
        return self._block.duration

    def __str__(self):
        return "(%d, %s)" % (self._begin_time, self._block)


class Schedule(ScheduleNode, Pulse):
    """Schedule of pulses with timing. The root of a schedule tree."""

    def __init__(self,
                 device: DeviceSpecification,
                 schedules: List[Tuple[int, Pulse]] = None,
                 name: str = None
                 ):
        """Create schedule.

        Args:
            device:
            schedules:
            name:
        """
        self._device = device
        self._name = name
        self._occupancy = TimeslotOccupancy(timeslots=[])
        self._children = []
        if schedules:
            for t0, pulse in schedules:
                if isinstance(pulse, Schedule):
                    raise NotImplementedError("This version doesn't support schedule of schedules.")
                elif isinstance(pulse, Pulse):
                    self.insert(t0, pulse)
                else:
                    raise ScheduleError("Invalid to be scheduled: %s" % pulse.__class__.__name__)

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> DeviceSpecification:
        return self._device

    def insert(self, begin_time: int, block: Pulse):
        """Insert a new pulse at `begin_time`.

        Args:
            begin_time:
            block (Pulse):
        """
        if not isinstance(block, Pulse):
            raise ScheduleError("Invalid to be inserted: %s" % block.__class__.__name__)
        # self._check_channels(block)
        shifted = block.occupancy.shifted(begin_time)
        if self._occupancy.is_mergeable_with(shifted):
            self._occupancy = self._occupancy.merged(shifted)
            self._children.append(TimedPulse(begin_time, block))
        else:
            logger.warning("Fail to insert %s at %s due to timing overlap", block, begin_time)
            raise ScheduleError("Fail to insert %s at %s due to overlap" % (str(block), begin_time))

    def append(self, block: Pulse):
        """Append a new pulse on a channel at the timing
        just after the last pulse finishes.

        Args:
            block (Pulse):
        """
        if not isinstance(block, Pulse):
            raise ScheduleError("Invalid to be inserted: %s" % block.__class__.__name__)
        # self._check_channels(block)
        begin_time = self.end_time
        try:
            self.insert(begin_time, block)
        except ScheduleError:
            logger.warning("Fail to append %s due to timing overlap", block)
            raise ScheduleError("Fail to append %s due to overlap" % str(block))

    @property
    def children(self) -> List[ScheduleNode]:
        """Child nodes of this schedule node. """
        return self._children

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
    def channelset(self) -> Set[Channel]:
        return self._occupancy.channelset

    @property
    def occupancy(self) -> TimeslotOccupancy:
        return self._occupancy

    def _check_channels(self, pulse: Pulse):
        if isinstance(pulse, Schedule):
            if pulse._device != self._device:
                raise ScheduleError("Additional schedule must have same device as self")
        else:
            # check if all the channels of pulse are defined in the device
            for ch in pulse.channelset:
                if not self._device.has_channel(ch):
                    raise ScheduleError("%s has no channel %s" % (self._device, ch))

    def __str__(self):
        # TODO: Handle schedule of schedules
        for child in self._children:
            if child.children:
                raise NotImplementedError("This version doesn't support schedule of schedules.")
        res = []
        for child in self._children:
            res.append((child.begin_time(), str(child.pulse)))
        res = sorted(res)
        return '\n'.join(["%4d: %s" % i for i in res])

    def flat_pulse_sequence(self) -> List[TimedPulse]:
        # TODO: Handle schedule of schedules
        for child in self._children:
            if child.children:
                raise NotImplementedError("This version doesn't support schedule of schedules.")
        return self._children
