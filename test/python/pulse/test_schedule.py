# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the pulse schedule."""
import unittest

import numpy as np

from qiskit.pulse.channels import DeviceSpecification, Qubit, RegisterSlot, MemorySlot
from qiskit.pulse.channels import DriveChannel, AcquireChannel, ControlChannel
from qiskit.pulse.commands import FrameChange, Acquire, PersistentValue, Snapshot
from qiskit.pulse.commands import functional_pulse, Instruction
from qiskit.pulse.common.timeslots import TimeslotCollection
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule
from qiskit.test import QiskitTestCase


class TestSchedule(QiskitTestCase):
    """Schedule tests."""

    def setUp(self):
        @functional_pulse
        def linear(duration, slope, intercept):
            x = np.linspace(0, duration - 1, duration)
            return slope * x + intercept

        self.linear = linear

        qubits = [Qubit(0, drive_channels=[DriveChannel(0, 1.2)],
                        control_channels=[ControlChannel(0)]),
                  Qubit(1, drive_channels=[DriveChannel(1, 3.4)],
                        acquire_channels=[AcquireChannel(1)])]
        registers = [RegisterSlot(i) for i in range(2)]
        mem_slots = [MemorySlot(i) for i in range(2)]
        self.two_qubit_device = DeviceSpecification(qubits, registers, mem_slots)

    def test_append_an_instruction_to_empty_schedule(self):
        """Test append instructions to an emtpy schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(3, sched.stop_time)

    def test_append_instructions_applying_to_different_channels(self):
        """Test append instructions to schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))
        sched = sched.append(lp0(device.q[1].drive))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(6, sched.stop_time)

    def test_insert_an_instruction_into_empty_schedule(self):
        """Test insert an instruction into an emtpy schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.q[0].drive))
        self.assertEqual(10, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_insert_an_instruction_before_an_existing_instruction(self):
        """Test insert an instruction before an existing instruction."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.q[0].drive))
        sched = sched.insert(5, lp0(device.q[0].drive))
        self.assertEqual(5, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_fail_to_insert_instruction_into_occupied_timing(self):
        """Test insert an instruction before an existing instruction."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.q[0].drive))
        with self.assertRaises(PulseError):
            sched.insert(11, lp0(device.q[0].drive))

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error."""
        device = self.two_qubit_device

        # pylint: disable=invalid-name
        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        gp0 = gaussian(duration=20, amp=0.7, t0=9.5, sig=3)
        gp1 = gaussian(duration=20, amp=0.5, t0=9.5, sig=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        sched = Schedule()
        sched = sched.append(gp0(device.q[0].drive))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(device.q[0].control))
        sched = sched.insert(60, FrameChange(phase=-1.57)(device.q[0].drive))
        sched = sched.insert(30, gp1(device.q[1].drive))
        sched = sched.insert(60, gp0(device.q[0].control))
        sched = sched.insert(80, Snapshot("label", "snap_type"))
        sched = sched.insert(90, fc_pi_2(device.q[0].drive))
        sched = sched.insert(90, acquire(device.q[1], device.mem[1], device.c[1]))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(100, sched.stop_time)
        self.assertEqual(100, sched.duration)
        new_sched = Schedule()
        new_sched = new_sched.append(sched)
        new_sched = new_sched.append(sched)
        self.assertEqual(0, new_sched.start_time)
        self.assertEqual(200, new_sched.stop_time)
        self.assertEqual(200, new_sched.duration)

    def test_can_create_valid_schedule_with_syntax_sugar(self):
        """Test valid schedule creation using syntax sugar without error."""
        device = self.two_qubit_device

        # pylint: disable=invalid-name
        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        gp0 = gaussian(duration=20, amp=0.7, t0=9.5, sig=3)
        gp1 = gaussian(duration=20, amp=0.5, t0=9.5, sig=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        sched = Schedule()
        sched += gp0(device.q[0].drive)
        sched |= PersistentValue(value=0.2 + 0.4j)(device.q[0].control).shifted(0)
        sched |= FrameChange(phase=-1.57)(device.q[0].drive).shifted(60)
        sched |= gp1(device.q[1].drive).shifted(30)
        sched |= gp0(device.q[0].control).shifted(60)
        sched |= Snapshot("label", "snap_type").shifted(80)
        sched |= fc_pi_2(device.q[0].drive).shifted(90)
        sched |= acquire(device.q[1], device.mem[1], device.c[1]).shifted(90)
        _ = Schedule() + sched + sched

    def test_empty_schedule(self):
        """Test empty schedule."""
        sched = Schedule()
        self.assertEqual(0, sched.start_time)
        self.assertEqual(0, sched.stop_time)
        self.assertEqual(0, sched.duration)
        self.assertEqual((), sched.children)
        self.assertEqual(TimeslotCollection([]), sched.occupancy)
        self.assertEqual([], sched.flat_instruction_sequence())

    def test_flat_instruction_sequence_returns_instructions(self):
        """Test if `flat_instruction_sequence` returns `Instruction`s."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        # empty schedule with empty schedule
        empty = Schedule().append(Schedule())
        for i in empty.flat_instruction_sequence():
            self.assertIsInstance(i, Instruction)

        # normal schedule
        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.q[0].drive))  # grand child 1
        subsched = subsched.append(lp0(device.q[0].drive))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))   # child
        sched = sched.append(subsched)
        for i in sched.flat_instruction_sequence():
            self.assertIsInstance(i, Instruction)

    def test_absolute_start_time_of_grandchild(self):
        """Test correct calculation of start time of grandchild of a schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.q[0].drive))  # grand child 1
        subsched = subsched.append(lp0(device.q[0].drive))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))   # child
        sched = sched.append(subsched)

        start_times = sorted([i.start_time for i in sched.flat_instruction_sequence()])
        self.assertEqual([0, 30, 40], start_times)

    def test_shifted_schedule(self):
        """Test shifted schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.q[0].drive))  # grand child 1
        subsched = subsched.append(lp0(device.q[0].drive))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))   # child
        sched = sched.append(subsched)

        shifted = sched.shifted(100)

        start_times = sorted([i.start_time for i in shifted.flat_instruction_sequence()])
        self.assertEqual([100, 130, 140], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its children after attached to another schedule."""
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

        # add 2 instructions to children (2 instructions -> 4 instructions)
        children = children.append(acquire(device.q[1], device.mem[1]))
        children = children.insert(100, acquire(device.q[1], device.mem[1]))
        self.assertEqual(4, len(children.flat_instruction_sequence()))
        # sched must keep 3 instructions (must not update to 5 instructions)
        self.assertEqual(3, len(sched.flat_instruction_sequence()))


if __name__ == '__main__':
    unittest.main()
