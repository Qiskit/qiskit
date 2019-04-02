# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Base command.
"""

from qiskit.pulse.exceptions import CommandsError


class PulseCommand:
    """Super class of command group."""

    pulseIndex = 0

    def __init__(self, duration, name=None):
        """Create new pulse commands.

        Args:
            duration (int): Duration of pulse.
            name (str): Name of pulse command.
        Raises:
            CommandsError: when duration is not number of points.
        """

        if isinstance(duration, int):
            self.duration = duration
        else:
            raise CommandsError('Pulse duration should be integer.')

        if name:
            self.name = name
        else:
            self.name = 'p%d' % PulseCommand.pulseIndex
            PulseCommand.pulseIndex += 1

    def __eq__(self, other):
        """Two PulseCommands are the same if they are of the same type
        and have the same duration and name.

        Args:
            other (PulseCommand): other PulseCommand.

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.duration == other.duration and\
                self.name == other.name:
            return True
        return False

    def __repr__(self):
        return '%s(name=%s, duration=%d)' % (self.__class__.__name__,
                                             self.name, self.duration)
