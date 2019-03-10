# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Persistent value.
"""

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands.pulse_command import PulseCommand


class PersistentValue(PulseCommand):
    """Persistent value."""

    def __init__(self, value):
        """create new persistent value command.

        Args:
            value (complex): Complex value to apply, bounded by an absolute value of 1.
                The allowable precision is device specific.
        Raises:
            QiskitError: when input value exceed 1.
        """

        super(PersistentValue, self).__init__(duration=0, name='pv')

        if abs(value) > 1:
            raise QiskitError("Absolute value of PV amplitude exceeds 1.")

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
