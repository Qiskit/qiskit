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
Base command.
"""
from abc import ABCMeta, abstractmethod

from qiskit.pulse.exceptions import PulseError

from .instruction import Instruction


class Command(metaclass=ABCMeta):
    """Super abstract class of command group."""

    pulseIndex = 0

    @abstractmethod
    def __init__(self, duration: int = None, name: str = None):
        """Create a new command.

        Args:
            duration (int): Duration of this command.
            name (str): Name of this command.
        Raises:
            PulseError: when duration is not number of points.
        """
        if isinstance(duration, int):
            self._duration = duration
        else:
            raise PulseError('Pulse duration should be integer.')

        if name:
            self._name = name
        else:
            self._name = 'p%d' % Command.pulseIndex
            Command.pulseIndex += 1

    @property
    def duration(self) -> int:
        """Duration of this command. """
        return self._duration

    @property
    def name(self) -> str:
        """Name of this command. """
        return self._name

    @abstractmethod
    def to_instruction(self, command, *channels, timeslots=None, name=None) -> Instruction:
        """Create an instruction from command."""
        pass

    def __call__(self, *args, **kwargs):
        """Creates an Instruction obtained from call to `to_instruction` wrapped in a Schedule."""
        return self.to_instruction(*args, **kwargs)

    def __eq__(self, other):
        """Two Commands are the same if they are of the same type
        and have the same duration and name.

        Args:
            other (Command): other Command.

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._duration == other._duration and \
                self._name == other._name:
            return True
        return False

    def __hash__(self):
        return hash((type(self), self._duration, self._name))

    def __repr__(self):
        return '%s(name=%s, duration=%d)' % (self.__class__.__name__,
                                             self._name, self._duration)
