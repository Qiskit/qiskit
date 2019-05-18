# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Frame change pulse.
"""

from qiskit.pulse.channels import PulseChannel
from .instruction import Instruction
from .command import Command


class FrameChange(Command):
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

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel, name=None) -> 'FrameChangeInstruction':
        return FrameChangeInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class FrameChangeInstruction(Instruction):
    """Instruction to change frame of an `PulseChannel`. """

    def __init__(self, command: FrameChange, channel: PulseChannel, name=None):
        super().__init__(command, channel, name=name)
        self._buffer = 0
