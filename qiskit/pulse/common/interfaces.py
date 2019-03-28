# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pulse = Command with its operands (Channels).
"""
from abc import ABCMeta, abstractmethod
from typing import Set, Optional, List

from qiskit.pulse.channels import PulseChannel
from .timeslots import TimeslotOccupancy


class Pulse(metaclass=ABCMeta):
    """Common interface for `Command with its operands (Channels)`. """

    @property
    @abstractmethod
    def duration(self) -> int:
        pass

    @property
    @abstractmethod
    def channelset(self) -> Set[PulseChannel]:
        pass

    @property
    @abstractmethod
    def occupancy(self) -> TimeslotOccupancy:
        pass

    def __repr__(self):
        return '%s(duration=%d, channelset=%s)' % \
               (self.__class__.__name__, self.duration, self.channelset)


class ScheduleNode(metaclass=ABCMeta):
    """Common interface for nodes of a schedule tree. """

    @property
    @abstractmethod
    def children(self) -> List['ScheduleNode']:
        """Child nodes of this schedule node. """
        pass

    @property
    @abstractmethod
    def begin_time(self) -> int:
        """Relative begin time of this schedule node. """
        pass

    @property
    @abstractmethod
    def end_time(self) -> int:
        """Relative end time of this schedule node. """
        pass
