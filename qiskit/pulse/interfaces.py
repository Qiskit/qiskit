# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
ScheduleComponent, a common interface for components of schedule (Instruction and Schedule).
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union

from qiskit.pulse.channels import Channel

from .timeslots import TimeslotCollection


class ScheduleComponent(metaclass=ABCMeta):
    """Common interface for components of schedule. """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of ScheduleComponent."""
        pass

    @property
    @abstractmethod
    def channels(self) -> List[Channel]:
        """Return channels used by schedule."""
        pass

    @property
    @abstractmethod
    def duration(self) -> int:
        """Duration of this schedule component. """
        pass

    @property
    @abstractmethod
    def start_time(self) -> int:
        """Starting time of this schedule component. """
        pass

    @property
    @abstractmethod
    def stop_time(self) -> int:
        """Stopping time of this schedule component. """
        pass

    @abstractmethod
    def ch_duration(self, *channels: List[Channel]) -> int:
        """Duration of the `channels` in schedule component. """
        pass

    @abstractmethod
    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Starting time of the `channels` in schedule component. """
        pass

    @abstractmethod
    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Stopping of the `channels` in schedule component. """
        pass

    @property
    @abstractmethod
    def timeslots(self) -> TimeslotCollection:
        """Occupied time slots by this schedule component. """
        pass

    @property
    @abstractmethod
    def children(self) -> Tuple[Union[int, 'ScheduleComponent']]:
        """Child nodes of this schedule component. """
        pass
