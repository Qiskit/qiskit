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

"""Test cases for the utilities."""

import unittest

from qiskit.pulse.utils import replace_implicit_acquires, align_measures
from qiskit.test import QiskitTestCase


class TestAutoMerge(QiskitTestCase):
    """"""

   def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.device = DeviceSpecification.create_from(self.backend)
        self.config = backend.configuration()
        self.defaults = backend.defaults()

    def test_align_measures():
        schedule = pulse.Schedule(name='fake_experiment')
        schedule = schedule.insert(0, short_pulse(self.device.q[0].drive))
        schedule = schedule.insert(1, acquire(self.device.q[0], device.mem[0]))
        schedule = schedule.insert(10, acquire(self.device.q[1], device.mem[1]))

        schedule = auto_align_measures(schedule, self.defaults.cmd_def,
                                       self.defaults.pulse_library)


if __name__ == '__main__':
    unittest.main()
