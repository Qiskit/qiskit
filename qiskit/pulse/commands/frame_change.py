# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Frame change pulse.
"""

from qiskit.pulse.commands.pulse_command import PulseCommand


class FrameChange(PulseCommand):
    """Frame change pulse."""

    def __init__(self, phase, name=None):
        """Create new frame change pulse.

        Args:
            phase (float): Frame change phase in radians.
                The allowable precision is device specific.
            name (str): Unique name to identify the command object.
        """

        super(FrameChange, self).__init__(duration=0, name=name)

        self.phase = phase
