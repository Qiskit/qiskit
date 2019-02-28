# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Base command.
"""

from qiskit.exceptions import QiskitError


class PulseCommand:
    """Super class of command group."""

    def __init__(self, duration, name):
        """Create new pulse commands.

        Args:
            duration (int): Duration of pulse.
            name (str): Name of pulse command.
        Raises:
            QiskitError: when duration is not number of points.
        """

        if isinstance(duration, int):
            self.duration = duration
        else:
            raise QiskitError('Pulse duration should be integer.')

        self.name = name

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
