# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The pulse program. A collection of pulses schedules."""

import copy
from typing import List, Optional

from qiskit.pulse.schedule import Schedule


class Program:
    """A pulse program, which is a collection of schedules."""
    def __init__(self, schedules: Optional[List[Schedule]] = None):
        """Qiskit pulse program.

        A collection of :class:`qiskit.pulse.schedule.Schedule`s.

        Args:
            schedules: Schedules to run in this program.
        """
        self._schedules = []

        if schedules or isinstance(schedules, Schedule):
            try:
                iter(schedules)
            except TypeError:
                schedules = [schedules]

            for schedule in schedules:
                self.append_schedule(schedule)

    @property
    def schedules(self):
        """Schedules in this program."""
        return copy.copy(self._schedules)

    def append_schedule(self, schedule: Schedule):
        """Append a schedule to this program."""
        self._schedules.append(schedule)

    def replace_schedule(self, idx: int, schedule: Schedule):
        """Replace schedule at index."""
        self._schedules[idx] = schedule
