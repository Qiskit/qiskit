# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,missing-param-doc

"""
Frame change pulse.
"""

from qiskit.pulse.commands._pulse_command import PulseCommand


class FrameChange(PulseCommand):
    """Frame change pulse"""

    def __init__(self, phase):
        """create new frame change pulse

        Args:
            phase (double): frame change phase in radians.
                the allowable precision is device specific.
        """

        super(FrameChange, self).__init__(0)

        self.phase = phase
