# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support signal output.
"""

from qiskit.pulse.channels.backend_channels import OutputChannel


class DriveChannel(OutputChannel):
    """Drive Channel."""

    prefix = 'D'

    def __init__(self, index):
        """Create new drive (D) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index


class ControlChannel(OutputChannel):
    """Control Channel."""

    prefix = 'U'

    def __init__(self, index):
        """Create new control (U) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index


class MeasureChannel(OutputChannel):
    """Measure Channel."""

    prefix = 'M'

    def __init__(self, index):
        """Create new measurement (M) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
