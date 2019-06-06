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
Persistent value.
"""

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .command import Command


class PersistentValue(Command):
    """Persistent value."""

    def __init__(self, value):
        """create new persistent value command.

        Args:
            value (complex): Complex value to apply, bounded by an absolute value of 1.
                The allowable precision is device specific.
        Raises:
            PulseError: when input value exceed 1.
        """
        super().__init__(duration=0)

        if abs(value) > 1:
            raise PulseError("Absolute value of PV amplitude exceeds 1.")

        self._value = complex(value)

    @property
    def value(self):
        """Persistent value amplitude."""
        return self._value

    def __eq__(self, other):
        """Two PersistentValues are the same if they are of the same type
        and have the same value.

        Args:
            other (PersistentValue): other PersistentValue

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.value == other.value:
            return True
        return False

    def __repr__(self):
        return '%s(%s, value=%s)' % (self.__class__.__name__, self.name, self.value)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel, name=None) -> 'PersistentValueInstruction':
        return PersistentValueInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class PersistentValueInstruction(Instruction):
    """Instruction to keep persistent value. """

    def __init__(self, command: PersistentValue, channel: PulseChannel, name=None):
        super().__init__(command, channel, name=name)
