# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pulse Channels.
"""

from qiskit.pulse import commands


class PulseChannel:
    """Pulse Channel."""

    supported = commands.PulseCommand

    def __init__(self, index):
        """Create new channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index

    def __str__(self):
        return 'C%d' % self.index
