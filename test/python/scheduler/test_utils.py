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
                          MeasureChannel, MemorySlot)
from qiskit.scheduler.utils import measure, measure_all
from qiskit.pulse.exceptions import PulseError
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.test import QiskitTestCase


class TestUtils(QiskitTestCase):
    """Scheduling tests."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.cmd_def = self.backend.defaults().build_cmd_def()

    def test_measure(self):
        """Test utility function - measure."""
        acquire = Acquire(duration=10)
        sched_with_qubit_mem_slots = measure(qubits=[0],
                                             backend=self.backend,
                                             qubit_mem_slots={0: 1})
        sched_without_qubit_mem_slots = measure(qubits=[0],
                                                backend=self.backend)
        sched_with_meas_map_list = measure(qubits=[0],
                                           backend=self.backend,
                                           meas_map=[[0, 1]])
        sched_with_meas_map_dict = measure(qubits=[0],
                                           backend=self.backend,
                                           meas_map={0: [0, 1], 1: [0, 1]})
        expected1 = Schedule(
            self.cmd_def.get('measure', [0, 1]).filter(channels=[MeasureChannel(0)]),
            acquire(AcquireChannel(0), MemorySlot(1)),
            acquire(AcquireChannel(1), MemorySlot(0)))
        expected2 = Schedule(
            self.cmd_def.get('measure', [0, 1]).filter(channels=[MeasureChannel(0)]),
            acquire(AcquireChannel(0), MemorySlot(0)),
            acquire(AcquireChannel(1), MemorySlot(1)))
        self.assertEqual(sched_with_qubit_mem_slots.instructions, expected1.instructions)
        self.assertEqual(sched_without_qubit_mem_slots.instructions, expected2.instructions)
        self.assertEqual(sched_with_meas_map_list.instructions, expected2.instructions)
        self.assertEqual(sched_with_meas_map_dict.instructions, expected2.instructions)

    def test_fail_measure(self):
        """Test failing measure."""
        with self.assertRaises(PulseError):
            measure(qubits=[0],
                    meas_map=self.backend.configuration().meas_map)
        with self.assertRaises(PulseError):
            measure(qubits=[0],
                    inst_map=self.backend.defaults().circuit_instruction_map)

    def test_measure_all(self):
        """Test measure_all function."""
        sched = measure_all(self.backend)
        expected = Schedule(self.cmd_def.get('measure', [0, 1]))
        self.assertEqual(sched.instructions, expected.instructions)
