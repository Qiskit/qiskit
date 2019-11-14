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
Set channel frequency.
"""
from typing import Optional

from qiskit.pulse.channels import PulseChannel
from .instruction import Instruction
from .command import Command


class SetChannelFrequency(Command):
    """
    Set the channel frequency
    """

    prefix = 'scf'

    def __init__(self, frequency: float, name: Optional[str] = None):
        """Creates a new set channel frequency pulse.

        Args:
            frequency: new frequency of the channel in GHz
            name: name of this set channel frequency command
        """
        super().__init__(duration=0)
        self._frequency = float(frequency)
        self._name = SetChannelFrequency.create_name(name)

    @property
    def frequency(self):
        """New frequency."""
        return self._frequency

    def __eq__(self, other: 'SetChannelFrequency'):
        """
        Two set channel frequency commands are the same if they have the same type and frequency.

        Args:
            other: other SetChannelFrequency

        Returns:
            bool: are self and other equal
        """
        return super().__eq__(other) and (self.frequency == other.frequency)

    def __hash__(self):
        return hash((super().__hash__(), self.frequency))

    def __repr__(self):
        return '%s(%s, frequency=%.3f)' % (self.__class__.__name__, self.name, self.frequency)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel, name=None) -> 'SetChannelFrequencyInstruction':
        return SetChannelFrequencyInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class SetChannelFrequencyInstruction(Instruction):
    """Instruction to change the frequency of a `PulseChannel`."""

    def __init__(self, command: SetChannelFrequency, channel: PulseChannel, name=None):
        super().__init__(command, channel, name=name)
