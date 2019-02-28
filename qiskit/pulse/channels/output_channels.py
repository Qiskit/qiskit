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

    supported = OutputChannel.supported
    prefix = 'd'

    def __init__(self, index: int, lo_frequency: float = None):
        """Create new drive (D) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
        self.lo_frequency = lo_frequency


class ControlChannel(OutputChannel):
    """Control Channel."""

    supported = OutputChannel.supported
    prefix = 'c'

    def __init__(self, index, lo_frequency: float = None):
        """Create new control (U) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
        self.lo_frequency = lo_frequency


class MeasureChannel(OutputChannel):
    """Measure Channel."""

    supported = OutputChannel.supported
    prefix = 'm'

    def __init__(self, index, lo_frequency: float = None):
        """Create new measurement (M) channel.

        Args:
            index (int): Index of the channel.
        """
        self.index = index
        self.lo_frequency = lo_frequency


