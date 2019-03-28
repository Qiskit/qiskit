# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support signal output.
"""
from abc import abstractmethod

from .pulse_channel import Channel


class OutputChannel(Channel):
    """Output Channel."""

    @abstractmethod
    def __init__(self, index: int = None, lo_frequency: float = None):
        super().__init__(index)
        self._lo_frequency = lo_frequency

    @property
    def lo_frequency(self) -> float:
        """Get the frequency of local oscillator of this channel."""
        return self._lo_frequency

    def set_lo_frequency(self, lo_frequency: float):
        """Set the frequency of local oscillator of this channel."""
        self._lo_frequency = lo_frequency

    def __eq__(self, other):
        """Two output channels are the same if they are of the same type, and
        have the same index and lo_frequency.

        Args:
            other (OutputChannel): other OutputChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._index == other._index and \
                self._lo_frequency == other._lo_frequency:
            return True
        return False

    def __hash__(self):
        return hash((super().__hash__(), self._lo_frequency))


class DriveChannel(OutputChannel):
    """Drive Channel."""

    prefix = 'd'

    def __init__(self, index: int, lo_frequency: float = None):
        """Create new drive (d) channel.

        Args:
            index (int): index of the channel
            lo_frequency (float): frequency of local oscillator
        """
        super().__init__(index, lo_frequency)


class ControlChannel(OutputChannel):
    """Control Channel."""

    prefix = 'u'

    def __init__(self, index, lo_frequency: float = None):
        """Create new control (u) channel.

        Args:
            index (int): index of the channel
            lo_frequency (float): frequency of local oscillator
        """
        super().__init__(index, lo_frequency)


class MeasureChannel(OutputChannel):
    """Measure Channel."""

    prefix = 'm'

    def __init__(self, index, lo_frequency: float = None):
        """Create new measurement (m) channel.

        Args:
            index (int): index of the channel
            lo_frequency (float): frequency of local oscillator
        """
        super().__init__(index, lo_frequency)
