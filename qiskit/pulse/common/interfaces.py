# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction = Command with its operands (Channels).
"""
from abc import ABCMeta, abstractmethod

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
