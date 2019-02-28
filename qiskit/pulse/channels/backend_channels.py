# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support actual backends.
"""

from qiskit.pulse import commands
from qiskit.pulse.channels.pulse_channel import PulseChannel


class OutputChannel(PulseChannel):
    """Output Channel."""

    supported = (commands.FrameChange,
                 commands.FunctionalPulse,
                 commands.PersistentValue,
                 commands.SamplePulse)


class AcquireChannel(PulseChannel):
    """Acquire Channel."""

    supported = commands.Acquire
    prefix = 'A'

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index


class SnapshotChannel(PulseChannel):
    """Snapshot Channel."""

    supported = commands.Snapshot
    prefix = 'S'

    def __init__(self, index):
        """Create new snapshot channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
