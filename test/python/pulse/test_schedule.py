# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Test cases for the pulse schedule."""
import unittest
import numpy as np
import pprint

from qiskit.pulse.channels import DeviceSpecification, Qubit
from qiskit.pulse.channels import DriveChannel, AcquireChannel, RegisterSlot, ControlChannel
from qiskit.pulse.commands import function, FrameChange, Acquire, PersistentValue, Snapshot
from qiskit.pulse.schedule import Schedule
from qiskit.test import QiskitTestCase


class TestSchedule(QiskitTestCase):
    """Schedule tests."""

    def setUp(self):
        qubits = [
            Qubit(0, drive_channels=[DriveChannel(0, 1.2)], control_channels=[ControlChannel(0)]),
            Qubit(1, drive_channels=[DriveChannel(1, 3.4)], acquire_channels=[AcquireChannel(1)])
        ]
        registers = [RegisterSlot(i) for i in range(2)]
        self.two_qubit_device = DeviceSpecification(qubits, registers)

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error.
        """
        device = self.two_qubit_device

        @function
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        gp0 = gaussian(duration=20, name='pulse0', amp=0.7, t0=9.5, sig=3)
        gp1 = gaussian(duration=20, name='pulse1', amp=0.5, t0=9.5, sig=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        sched = Schedule()
        sched = sched.appended(gp0(device.q[0].drive))
        sched = sched.inserted(0, PersistentValue(value=0.2 + 0.4j)(device.q[0].control))
        sched = sched.inserted(30, gp1(device.q[1].drive))
        sched = sched.inserted(60, FrameChange(phase=-1.57)(device.q[0].drive))
        sched = sched.inserted(60, gp0(device.q[0].control))
        sched = sched.inserted(80, Snapshot("label", "snap_type"))
        sched = sched.inserted(90, fc_pi_2(device.q[0].drive))
        sched = sched.inserted(90, acquire(device.q[1], device.mem[1], device.c[1]))
        # print(sched)

        new_sched = Schedule()
        new_sched = new_sched.appended(sched)
        new_sched = new_sched.appended(sched)
        _ = new_sched.flat_instruction_sequence()
        # print(pprint.pformat(new_sched.flat_instruction_sequence()))

        self.assertTrue(True)

    def test_absolute_begin_time_of_grandchild(self):
        """Test correct calculation of begin time of grandchild of a schedule.
        """
        device = self.two_qubit_device

        acquire = Acquire(10)
        children = Schedule()
        children = children.inserted(20, acquire(device.q[1], device.mem[1]))  # grand child 1
        children = children.appended(acquire(device.q[1], device.mem[1]))      # grand child 2

        sched = Schedule()
        sched = sched.appended(acquire(device.q[1], device.mem[1]))   # child
        sched = sched.appended(children)

        begin_times = sorted([i.begin_time for i in sched.flat_instruction_sequence()])

        self.assertEqual([0, 30, 40], begin_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its children after attached to another schedule.
        """
        device = self.two_qubit_device

        acquire = Acquire(10)
        children = Schedule()\
            .inserted(20, acquire(device.q[1], device.mem[1]))\
            .appended(acquire(device.q[1], device.mem[1]))
        self.assertEqual(2, len(children.flat_instruction_sequence()))

        sched = Schedule()\
            .appended(acquire(device.q[1], device.mem[1]))\
            .appended(children)
        self.assertEqual(3, len(sched.flat_instruction_sequence()))

        children = children.inserted(100, acquire(device.q[1], device.mem[1]))
        self.assertEqual(3, len(children.flat_instruction_sequence()))
        # sched must keep 3 instructions (must not update to 4 instructions)
        self.assertEqual(3, len(sched.flat_instruction_sequence()))


if __name__ == '__main__':
    unittest.main()
