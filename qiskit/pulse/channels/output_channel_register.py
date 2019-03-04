# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
PulseChannel register.
"""
import logging
from typing import List

from qiskit.pulse.exceptions import ChannelsError
from .channel_register import ChannelRegister
from .output_channel import OutputChannel, DriveChannel, ControlChannel, MeasureChannel

logger = logging.getLogger(__name__)


class OutputChannelRegister(ChannelRegister):
    """An abstract output channel register."""

    def __init__(self, channel_cls, size: int, name: str = None, lo_freqs: List[float] = None):
        """Create a new output channel register.
        """
        if not issubclass(channel_cls, OutputChannel):
            raise ChannelsError("Unknown output channel class: %s", channel_cls.__name__)

        super().__init__(channel_cls, size, name)

        if lo_freqs is None:
            lo_freqs = [None] * size

        self._channels = [self.channel_cls(i, lo_freqs[i]) for i in range(self.size)]

    def lo_frequencies(self):
        return [channel.lo_frequency for channel in self._channels]

    def __eq__(self, other):
        """Two output channel registers are the same if they are of the same type
         have the same channel class, name and size.

        Args:
            other (OutputChannelRegister): other OutputChannelRegister

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.channel_cls == other.channel_cls and \
                self.name == other.name and \
                self.size == other.size and \
                self._channels == other._channels:
            return True
        return False

    def __hash__(self):
        """Make object hashable."""
        return hash((type(self), self.channel_cls, self.name, self.size, hash(self._channels)))


class DriveChannelRegister(OutputChannelRegister):
    """Drive channel register."""

    def __init__(self, size: int, name: str = None, lo_freqs: List[float] = None):
        """Create a new drive channel register.
        """
        super().__init__(DriveChannel, size, name, lo_freqs)


class ControlChannelRegister(OutputChannelRegister):
    """Control channel register."""

    def __init__(self, size: int, name: str = None, lo_freqs: List[float] = None):
        """Create a new control channel register.
        """
        super().__init__(ControlChannel, size, name, lo_freqs)


class MeasureChannelRegister(OutputChannelRegister):
    """Measure channel register."""

    def __init__(self, size: int, name: str = None, lo_freqs: List[float] = None):
        """Create a new measure channel register.
        """
        super().__init__(MeasureChannel, size, name, lo_freqs)
