# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Test cases for the pulse schedule."""
# import pprint
import unittest
import numpy as np

from qiskit.pulse.channels import DeviceSpecification, Qubit, RegisterSlot, MemorySlot
from qiskit.pulse.channels import DriveChannel, AcquireChannel, ControlChannel
from qiskit.pulse.commands import functional_pulse, FrameChange, Acquire, PersistentValue, Snapshot
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
        mem_slots = [MemorySlot(i) for i in range(2)]
        self.two_qubit_device = DeviceSpecification(qubits, registers, mem_slots)

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error.
        """
        device = self.two_qubit_device

        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        gp0 = gaussian(duration=20, name='pulse0', amp=0.7, t0=9.5, sig=3)
        gp1 = gaussian(duration=20, name='pulse1', amp=0.5, t0=9.5, sig=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        sched = Schedule()
        sched = sched.append(gp0(device.q[0].drive))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(device.q[0].control))
        sched = sched.insert(30, gp1(device.q[1].drive))
        sched = sched.insert(60, FrameChange(phase=-1.57)(device.q[0].drive))
        sched = sched.insert(60, gp0(device.q[0].control))
        sched = sched.insert(80, Snapshot("label", "snap_type"))
        sched = sched.insert(90, fc_pi_2(device.q[0].drive))
        sched = sched.insert(90, acquire(device.q[1], device.mem[1], device.c[1]))
        # print(sched)
        new_sched = Schedule()
        new_sched = new_sched.append(sched)
        new_sched = new_sched.append(sched)
        _ = new_sched.flat_instruction_sequence()
        # print(pprint.pformat(new_sched.flat_instruction_sequence()))

    def test_can_create_valid_schedule_with_syntax_sugar(self):
        """Test valid schedule creation using syntax sugar without error.
        """
        device = self.two_qubit_device

        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        gp0 = gaussian(duration=20, name='pulse0', amp=0.7, t0=9.5, sig=3)
        gp1 = gaussian(duration=20, name='pulse1', amp=0.5, t0=9.5, sig=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        sched = Schedule()
        sched += gp0(device.q[0].drive)
        sched |= PersistentValue(value=0.2 + 0.4j)(device.q[0].control).shifted(0)
        sched |= gp1(device.q[1].drive).shifted(30)
        sched |= FrameChange(phase=-1.57)(device.q[0].drive).shifted(60)
        sched |= gp0(device.q[0].control).shifted(60)
        sched |= Snapshot("label", "snap_type").shifted(80)
        sched |= fc_pi_2(device.q[0].drive).shifted(90)
        sched |= acquire(device.q[1], device.mem[1], device.c[1]).shifted(90)
        # print(sched)
        new_sched = Schedule() + sched + sched
        _ = new_sched.flat_instruction_sequence()
        # print(pprint.pformat(new_sched.flat_instruction_sequence()))

    def test_absolute_start_time_of_grandchild(self):
        """Test correct calculation of start time of grandchild of a schedule.
        """
        device = self.two_qubit_device

        acquire = Acquire(10)
        children = Schedule()
        children = children.insert(20, acquire(device.q[1], device.mem[1]))  # grand child 1
        children = children.append(acquire(device.q[1], device.mem[1]))      # grand child 2

        sched = Schedule()
        sched = sched.append(acquire(device.q[1], device.mem[1]))   # child
        sched = sched.append(children)

        start_times = sorted([i.start_time for i in sched.flat_instruction_sequence()])

        self.assertEqual([0, 30, 40], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its children after attached to another schedule.
        """
        device = self.two_qubit_device

        acquire = Acquire(10)
        children = Schedule()\
            .insert(20, acquire(device.q[1], device.mem[1]))\
            .append(acquire(device.q[1], device.mem[1]))
        self.assertEqual(2, len(children.flat_instruction_sequence()))

        sched = Schedule()\
            .append(acquire(device.q[1], device.mem[1]))\
            .append(children)
        self.assertEqual(3, len(sched.flat_instruction_sequence()))

        children = children.insert(100, acquire(device.q[1], device.mem[1]))
        self.assertEqual(3, len(children.flat_instruction_sequence()))
        # sched must keep 3 instructions (must not update to 4 instructions)
        self.assertEqual(3, len(sched.flat_instruction_sequence()))

    def test_flat_instruction_sequence_returns_empty_list_for_empty_schedule(self):
        """Test if flat_instruction_sequence returns empty list for empty schedule.
        """
        sched = Schedule()
        self.assertEqual([], sched.flat_instruction_sequence())


if __name__ == '__main__':
    unittest.main()
