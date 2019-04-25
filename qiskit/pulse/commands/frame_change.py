# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Frame change pulse.
"""

from qiskit.pulse.channels import OutputChannel
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection
from .instruction import Instruction
from .pulse_command import PulseCommand


class FrameChange(PulseCommand):
    """Frame change pulse."""

    def __init__(self, phase):
        """Create new frame change pulse.

        Args:
            phase (float): Frame change phase in radians.
                The allowable precision is device specific.
        """
        super().__init__(duration=0)
        self._phase = phase

    @property
    def phase(self):
        """Framechange phase."""
        return self._phase

    def __eq__(self, other):
        """Two FrameChanges are the same if they are of the same type
        and have the same phase.

        Args:
            other (FrameChange): other FrameChange

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.phase == other.phase:
            return True
        return False

    def __repr__(self):
        return '%s(%s, phase=%.3f)' % (self.__class__.__name__, self.name, self.phase)

    def __call__(self, channel: OutputChannel, name=None) -> 'FrameChangeInstruction':
        return FrameChangeInstruction(self, channel, name=name)


class FrameChangeInstruction(Instruction):
    """Instruction to change frame of an `OutputChannel`. """

    def __init__(self, command: FrameChange, channel: OutputChannel, name=None):
        slots = [Timeslot(Interval(0, 0), channel)]
        super().__init__(command, TimeslotCollection(*slots), name=name)

    def __repr__(self):
        return '%s -> %s' % (self.command, self.channel)
