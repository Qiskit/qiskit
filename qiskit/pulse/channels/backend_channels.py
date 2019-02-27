# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support actual backends.
"""

from qiskit.pulse.channels.pulse_channel import PulseChannel
from qiskit.pulse import commands


class OutputChannel(PulseChannel):
    """Output Channel."""

    supported = (commands.FrameChange,
                 commands.FrameChange,
                 commands.PersistentValue,
                 commands.sample_pulse)

    def __init__(self, index):
        """Create new output channel.

        Args:
            index (int): Index of the channel.
        """
        super(OutputChannel, self).__init__(index)

    def __str__(self):
        return 'O%d' % self.index


class AcquireChannel(PulseChannel):
    """Acquire Channel."""

    supported = commands.Acquire

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        super(AcquireChannel, self).__init__(index)

    def __str__(self):
        return 'A%d' % self.index


class SnapshotChannel(PulseChannel):
    """Snapshot Channel."""

    supported = commands.Snapshot

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        super(SnapshotChannel, self).__init__(index)

    def __str__(self):
        return 'S%d' % self.index
