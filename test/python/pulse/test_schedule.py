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
        sched = sched.appended(gp0.to(device.q[0].drive))
        sched = sched.inserted(PersistentValue(value=0.2+0.4j).to(device.q[0].control).at(0))
        sched = sched.inserted(gp1.to(device.q[1].drive).at(30))
        sched = sched.inserted(FrameChange(phase=-1.57).to(device.q[0].drive).at(60))
        sched = sched.inserted(gp0.to(device.q[0].control).at(60))
        sched = sched.inserted(Snapshot("label", "snap_type").at(80))
        sched = sched.inserted(fc_pi_2.to(device.q[0].drive).at(90))
        sched = sched.inserted(acquire.to(device.q[1], device.mem[1], device.c[1]).at(90))
        # print(sched)

        new_sched = Schedule()
        new_sched = new_sched.appended(sched)
        new_sched = new_sched.appended(sched)
        _ = new_sched.flat_instruction_sequence()
        # print(pprint.pformat(new_sched.flat_instruction_sequence()))

        self.assertTrue(True)

    @unittest.skip("wait a moment")
    def test_can_create_valid_schedule_with_syntax_suger(self):
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
        sched += gp0 >> device.q[0].drive
        sched += PersistentValue(value=0.2+0.4j) >> device.q[0].control | 0
        sched += gp1 >> device.q[1].drive | 30
        sched += FrameChange(phase=-1.57) >> device.q[0].drive | 60
        sched += gp0 >> device.q[0].control | 60
        sched += Snapshot("label", "snap_type") | 80
        sched += fc_pi_2 >> device.q[0].drive | 90
        sched += acquire >> device.q[1], device.mem[1], device.c[1] | 90
        # print(sched)

        new_sched = Schedule()
        new_sched += sched
        new_sched += sched
        _ = new_sched.flat_instruction_sequence()
        # print(pprint.pformat(new_sched.flat_instruction_sequence()))

        self.assertTrue(True)

    def test_absolute_begin_time_of_grandchild(self):
        """Test correct calculation of begin time of grandchild of a schedule.
        """
        device = self.two_qubit_device

        acquire = Acquire(10)
        children = Schedule()
        children = children.inserted(acquire.to(device.q[1], device.mem[1]).at(20))  # grand child 1
        children = children.appended(acquire.to(device.q[1], device.mem[1]))         # grand child 2

        sched = Schedule()
        sched = sched.appended(acquire.to(device.q[1], device.mem[1]))   # child
        sched = sched.appended(children)

        begin_times = sorted([i.begin_time for i in sched.flat_instruction_sequence()])
        # print(pprint.pformat(sched.flat_instruction_sequence()))

        self.assertEqual(begin_times, [0, 30, 40])
