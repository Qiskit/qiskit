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
import warnings

from typing import Optional

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .command import Command


class PersistentValue(Command):
    """Persistent value."""

    prefix = 'pv'

    def __init__(self, value: complex, name: Optional[str] = None):
        """create new persistent value command.

        Args:
            value: Complex value to apply, bounded by an absolute value of 1
                The allowable precision is device specific
            name: Name of this command.
        Raises:
            PulseError: when input value exceed 1
        """
        super().__init__(duration=0)

        if abs(value) > 1:
            raise PulseError("Absolute value of PV amplitude exceeds 1.")

        warnings.warn("The PersistentValue command is deprecated. Use qiskit.pulse.ConstantPulse "
                      "instead.", DeprecationWarning)

        self._value = complex(value)
        self._name = PersistentValue.create_name(name)

    @property
    def value(self):
        """Persistent value amplitude."""
        return self._value

    def __eq__(self, other: 'PersistentValue') -> bool:
        """Two PersistentValues are the same if they are of the same type
        and have the same value.

        Args:
            other: other PersistentValue

        Returns:
            bool: are self and other equal
        """
        return super().__eq__(other) and self.value == other.value

    def __hash__(self):
        return hash((super().__hash__(), self.value))

    def __repr__(self):
        return '%s(value=%s, name="%s")' % (self.__class__.__name__,
                                            self.value,
                                            self.name)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel, name=None) -> 'PersistentValueInstruction':
        return PersistentValueInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class PersistentValueInstruction(Instruction):
    """Instruction to keep persistent value."""

    def __init__(self, command: PersistentValue, channel: PulseChannel, name=None):
        super().__init__(command, channel, name=name)
