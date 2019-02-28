# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support actual backends.
"""
from abc import abstractmethod

from qiskit.pulse import commands
from qiskit.pulse.channels.pulse_channel import PulseChannel


class OutputChannel(PulseChannel):
    """Output Channel."""

    supported = (commands.FrameChange,
                 commands.FunctionalPulse,
                 commands.PersistentValue,
                 commands.SamplePulse)

    @abstractmethod
    def __init__(self, index: int, lo_frequency: float = None):
        self.index = index
        self.lo_frequency = lo_frequency

    def __eq__(self, other):
        """Two output channels are the same if they are of the same type, and
        have the same index and lo_frequency.

        Args:
            other (PulseChannel): other PulseChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.index == other.index and \
                self.lo_frequency == other.lo_frequency:
            return True
        return False


class AcquireChannel(PulseChannel):
    """Acquire Channel."""

    supported = commands.Acquire
    prefix = 'a'

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index


class SnapshotChannel(PulseChannel):
    """Snapshot Channel."""

    supported = commands.Snapshot
    prefix = 's'

    def __init__(self, index):
        """Create new snapshot channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
