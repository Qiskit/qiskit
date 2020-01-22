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

"""Test cases for Pulse Utility functions."""

from qiskit.pulse import (Schedule, AcquireChannel, Acquire,
                          MeasureChannel, MemorySlot, measure)

from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.test import QiskitTestCase


class TestUtils(QiskitTestCase):
    """Scheduling tests."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.cmd_def = self.backend.defaults().build_cmd_def()

    def test_measure(self):
        """Test utility function - measure."""
        sched = Schedule()
        sched = measure(qubits=[0],
                        schedule=sched,
                        backend=self.backend,
                        inst_map=None,
                        meas_map=self.backend.configuration().meas_map)
        expected = Schedule(
            self.cmd_def.get('measure', [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(duration=10)([AcquireChannel(0), AcquireChannel(1)],
                                 [MemorySlot(0), MemorySlot(1)]))
        self.assertEqual(sched.instructions, expected.instructions)

    def test_fail_measure(self):
        """Test failing measure."""
        sched = Schedule()
        with self.assertRaises(AttributeError):
            sched = measure(qubits=[0],
                            schedule=sched,
                            backend=None,
                            inst_map=None,
                            meas_map=self.backend.configuration().meas_map)
        with self.assertRaises(AttributeError):
            sched = measure(qubits=[0],
                            schedule=sched,
                            backend=None,
                            inst_map=self.backend.defaults().inst_map,
                            meas_map=None)
