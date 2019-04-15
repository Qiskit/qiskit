# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Persistent value.
"""

from qiskit.pulse.channels import OutputChannel
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .pulse_command import PulseCommand


class PersistentValue(PulseCommand):
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

        self.value = value

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

    def __call__(self, channel: OutputChannel) -> 'PersistentValueInstruction':
        return PersistentValueInstruction(self, channel)


class PersistentValueInstruction(Instruction):
    """Instruction to keep persistent value. """

    def __init__(self, command: PersistentValue, channel: OutputChannel, start_time: int = 0):
        slots = [Timeslot(Interval(start_time, start_time), channel)]
        super().__init__(command, start_time, TimeslotCollection(slots))
        self._channel = channel

    @property
    def command(self) -> PersistentValue:
        """PersistentValue command. """
        return self._command

    @property
    def channel(self) -> OutputChannel:
        """OutputChannel channel."""
        return self._channel

    def __repr__(self):
        return '%4d: %s -> %s' % (self._start_time, self._command, self._channel)
