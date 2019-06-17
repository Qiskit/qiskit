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
import re

from abc import ABCMeta, abstractmethod

from qiskit.pulse.exceptions import PulseError

from .instruction import Instruction


class MetaCount(ABCMeta):
    """Meta class to count class instances."""
    def __new__(mcs, name, bases, namespace):
        new_cls = super(MetaCount, mcs).__new__(mcs, name, bases, namespace)
        new_cls.instances_counter = 0
        return new_cls


class Command(metaclass=MetaCount):
    """Super abstract class of command group."""

    # Counter for the number of instances in this class
    prefix = 'c'

    @abstractmethod
    def __init__(self, duration: int = None):
        """Create a new command.

        Args:
            duration (int): Duration of this command.
        Raises:
            PulseError: when duration is not number of points.
        """
        if isinstance(duration, int):
            self._duration = duration
        else:
            raise PulseError('Pulse duration should be integer.')

        self._name = Command.create_name()

    @classmethod
    def create_name(cls, name: str = None) -> str:
        """Method to create names for pulse commands."""
        if name is None:
            try:
                name = '%s%i' % (cls.prefix, cls.instances_counter)  # pylint: disable=E1101
            except TypeError:
                raise PulseError("prefix and counter must be non-None when name is None.")
        else:
            try:
                name = str(name)
            except Exception:
                raise PulseError("The pulse command name should be castable to a string "
                                 "(or None for autogenerate a name).")
            name_format = re.compile('[a-z][a-zA-Z0-9_]*')
            if name_format.match(name) is None:
                raise PulseError("%s is an invalid OpenPulse command name." % name)

        cls.instances_counter += 1  # pylint: disable=E1101

        return name

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
