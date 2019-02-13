# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Persistent value.
"""

import warnings

from qiskit.pulse.commands._pulse_command import PulseCommand


class PersistentValue(PulseCommand):
    """Persistent value"""

    def __init__(self, value):
        """create new persistent value command

        Args:
            value (complex): complex value to apply, bounded by an absolute value of 1.
                the allowable precision is device specific.
        """

        super(PersistentValue, self).__init__(0)

        if abs(value) > 1:
            warnings.warn("Absolute value of pulse amplitude exceeds 1.")
            _value = value/abs(value)
        else:
            _value = value

        self.value = _value
