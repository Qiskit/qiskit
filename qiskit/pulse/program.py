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

from typing import List

from qiskit.pulse.schedule import Schedule


class Program:
    def __init__(self, schedules: List[Schedule]):
        """Qiskit pulse program.

        A collection of :class:`qiskit.pulse.schedule.Schedule`s.

        Args:
            schedules: Schedules to run in this program.
        """
        try:
            iter(schedules)
        except TypeError:
            schedules = [schedules]
        self._schedules = schedules

    @property
    def schedules(self):
        """Schedules in this program."""
        return self._schedules

    def append(self, schedule: Schedule):
        """Append a schedule to this program."""
        self._schedules.append(schedule)
