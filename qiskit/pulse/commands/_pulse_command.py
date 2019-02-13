# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Super class of command group.
"""

import uuid

from qiskit.exceptions import QiskitError


class PulseCommand:
    """Super class of command group."""

    def __init__(self, duration):
        """create new pulse commands.

        Args:
            duration (int): duration of pulse
        Raises:
            QiskitError: when duration is not number of points
        """

        if isinstance(duration, int):
            self.duration = duration
        else:
            raise QiskitError('Pulse duration should be integer.')

        self.pulse_id = str(uuid.uuid4())
