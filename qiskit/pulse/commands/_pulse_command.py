# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Base command.
"""

from qiskit.exceptions import QiskitError


class PulseCommand:
    """Super class of command group."""

    def __init__(self, duration, name=None):
        """Create new pulse commands.

        Args:
            duration (int): Duration of pulse.
            name (str): Unique name to identify the command object.
        Raises:
            QiskitError: when duration is not number of points.
        """

        if isinstance(duration, int):
            self.duration = duration
        else:
            raise QiskitError('Pulse duration should be integer.')

        if name:
            _name = name
        else:
            _name = str(self.__hash__())

        self.name = _name
