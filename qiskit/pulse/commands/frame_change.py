# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Frame change pulse.
"""

from qiskit.pulse.channels import OutputChannel
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotCollection
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
        self.phase = phase

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

    def __call__(self, channel: OutputChannel) -> 'FrameChangeInstruction':
        return FrameChangeInstruction(self, channel)


class FrameChangeInstruction(Instruction):
    """Instruction to change frame of an `OutputChannel`. """

    def __init__(self, command: FrameChange, channel: OutputChannel, start_time: int = 0):
        slots = [Timeslot(Interval(start_time, start_time), channel)]
        super().__init__(command, start_time, TimeslotCollection(slots))
        self._channel = channel

    @property
    def command(self) -> FrameChange:
        """FrameChange command. """
        return self._command

    @property
    def channel(self) -> OutputChannel:
        """OutputChannel channel. """
        return self._channel

    def __repr__(self):
        return '%4d: %s -> %s' % (self._start_time, self._command, self._channel)
