# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
PulseChannel list.
"""
import logging
from typing import List

from qiskit.pulse import ChannelsError
from .channel_list import ChannelList
from .output_channel import OutputChannel, DriveChannel, ControlChannel, MeasureChannel

logger = logging.getLogger(__name__)


class OutputChannelList(ChannelList):
    """An abstract output channel list."""

    def __init__(self, channel_cls, size: int, lo_freqs: List[float] = None):
        """Create a new output channel list.
        """
        if not issubclass(channel_cls, OutputChannel):
            raise ChannelsError("Unknown output channel class: %s", channel_cls.__name__)

        super().__init__(channel_cls, size)

        if lo_freqs is None:
            lo_freqs = [None] * size

        self._channels = [self._channel_cls(i, lo_freqs[i]) for i in range(self._size)]

    def lo_frequencies(self):
        return [channel.lo_frequency for channel in self._channels]

    def __eq__(self, other):
        """Two output channel lists are the same if they are of the same type
         have the same size, channel class and channels.

        Args:
            other (OutputChannelList): other OutputChannelList

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._size == other._size and \
                self._channel_cls == other._channel_cls and \
                self._channels == other._channels:
            return True
        return False


class DriveChannelList(OutputChannelList):
    """Drive channel list."""

    def __init__(self, size: int, lo_freqs: List[float] = None):
        """Create a new drive channel list.
        """
        super().__init__(DriveChannel, size, lo_freqs)


class ControlChannelList(OutputChannelList):
    """Control channel list."""

    def __init__(self, size: int, lo_freqs: List[float] = None):
        """Create a new control channel list.
        """
        super().__init__(ControlChannel, size, lo_freqs)


class MeasureChannelList(OutputChannelList):
    """Measure channel list."""

    def __init__(self, size: int, lo_freqs: List[float] = None):
        """Create a new measure channel list.
        """
        super().__init__(MeasureChannel, size, lo_freqs)
