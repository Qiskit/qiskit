# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the pulse schedule."""
import unittest

import numpy as np

from qiskit.pulse.channels import (DeviceSpecification, Qubit, RegisterSlot, MemorySlot,
                                   DriveChannel, AcquireChannel, ControlChannel, MeasureChannel)
from qiskit.pulse.commands import (FrameChange, Acquire, PersistentValue, Snapshot,
                                   functional_pulse, Instruction)
from qiskit.pulse import pulse_lib
from qiskit.pulse.timeslots import TimeslotCollection
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

        qubits = [Qubit(0, DriveChannel(0), AcquireChannel(0), MeasureChannel(0),
                        control_channels=[ControlChannel(0)]),
                  Qubit(1, DriveChannel(1), MeasureChannel(0), AcquireChannel(1))]
        registers = [RegisterSlot(i) for i in range(2)]
        mem_slots = [MemorySlot(i) for i in range(2)]
        self.two_qubit_device = DeviceSpecification(qubits, registers, mem_slots)

    def test_append_an_instruction_to_empty_schedule(self):
        """Test append instructions to an empty schedule."""
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
        # appending to separate channel so should be at same time.
        self.assertEqual(3, sched.stop_time)

    def test_insert_an_instruction_into_empty_schedule(self):
        """Test insert an instruction into an empty schedule."""
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

        gp0 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)

        sched = Schedule()
        sched = sched.append(gp0(device.q[0].drive))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(device.q[0].controls[0]))
        sched = sched.insert(60, FrameChange(phase=-1.57)(device.q[0].drive))
        sched = sched.insert(30, gp1(device.q[1].drive))
        sched = sched.insert(60, gp0(device.q[0].controls[0]))
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
        """Test that in place operations on schedule are still immutable
           and return equivalent schedules."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)
        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)

        sched = Schedule()
        sched += gp0(device.q[0].drive)
        sched |= PersistentValue(value=0.2 + 0.4j)(device.q[0].controls[0])
        sched |= FrameChange(phase=-1.57)(device.q[0].drive) << 60
        sched |= gp1(device.q[1].drive) << 30
        sched |= gp0(device.q[0].controls[0]) << 60
        sched |= Snapshot("label", "snap_type") << 60
        sched |= fc_pi_2(device.q[0].drive) << 90
        sched |= acquire(device.q[1], device.mem[1], device.c[1]) << 90
        sched += sched

    def test_immutability(self):
        """Test that operations are immutable."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = gp1(device.q[0].drive) << 100
        # if schedule was mutable the next two sequences would overlap and an error
        # would be raised.
        sched.union(gp0(device.q[0].drive))
        sched.union(gp0(device.q[0].drive))

    def test_inplace(self):
        """Test that in place operations on schedule are still immutable."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = Schedule()
        sched = sched + gp1(device.q[0].drive)
        sched2 = sched
        sched += gp0(device.q[0].drive)
        self.assertNotEqual(sched, sched2)

    def test_empty_schedule(self):
        """Test empty schedule."""
        sched = Schedule()
        self.assertEqual(0, sched.start_time)
        self.assertEqual(0, sched.stop_time)
        self.assertEqual(0, sched.duration)
        self.assertEqual((), sched._children)
        self.assertEqual(TimeslotCollection(), sched.timeslots)
        self.assertEqual([], list(sched.instructions))

    def test_flat_instruction_sequence_returns_instructions(self):
        """Test if `flat_instruction_sequence` returns `Instruction`s."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        # empty schedule with empty schedule
        empty = Schedule().append(Schedule())
        for _, instr in empty.instructions:
            self.assertIsInstance(instr, Instruction)

        # normal schedule
        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.q[0].drive))  # grand child 1
        subsched = subsched.append(lp0(device.q[0].drive))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))   # child
        sched = sched.append(subsched)
        for _, instr in sched.instructions:
            self.assertIsInstance(instr, Instruction)

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

        start_times = sorted([shft+instr.start_time for shft, instr in sched.instructions])
        self.assertEqual([0, 30, 40], start_times)

    def test_shift_schedule(self):
        """Test shift schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.q[0].drive))  # grand child 1
        subsched = subsched.append(lp0(device.q[0].drive))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.q[0].drive))   # child
        sched = sched.append(subsched)

        shift = sched.shift(100)

        start_times = sorted([shft+instr.start_time for shft, instr in shift.instructions])

        self.assertEqual([100, 130, 140], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its _children after attached to another schedule."""
        device = self.two_qubit_device

        acquire = Acquire(10)
        _children = (acquire(device.q[1], device.mem[1]).shift(20) +
                     acquire(device.q[1], device.mem[1]))
        self.assertEqual(2, len(list(_children.instructions)))

        sched = acquire(device.q[1], device.mem[1]).append(_children)
        self.assertEqual(3, len(list(sched.instructions)))

        # add 2 instructions to _children (2 instructions -> 4 instructions)
        _children = _children.append(acquire(device.q[1], device.mem[1]))
        _children = _children.insert(100, acquire(device.q[1], device.mem[1]))
        self.assertEqual(4, len(list(_children.instructions)))
        # sched must keep 3 instructions (must not update to 5 instructions)
        self.assertEqual(3, len(list(sched.instructions)))

    def test_name_inherited(self):
        """Test that schedule keeps name if an instruction is added."""
        device = self.two_qubit_device

        acquire = Acquire(10)
        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3, name='pulse_name')
        pv0 = PersistentValue(0.1)
        fc0 = FrameChange(0.1)
        snapshot = Snapshot('snapshot_label', 'state')

        sched1 = Schedule(name='test_name')
        sched2 = Schedule(name=None)
        sched3 = sched1 | sched2
        self.assertEqual(sched3.name, 'test_name')

        sched_acq = acquire(device.q[1], device.mem[1], name='acq_name') | sched1
        self.assertEqual(sched_acq.name, 'acq_name')

        sched_pulse = gp0(device.q[0].drive) | sched1
        self.assertEqual(sched_pulse.name, 'pulse_name')

        sched_pv = pv0(device.q[0].drive, name='pv_name') | sched1
        self.assertEqual(sched_pv.name, 'pv_name')

        sched_fc = fc0(device.q[0].drive, name='fc_name') | sched1
        self.assertEqual(sched_fc.name, 'fc_name')

        sched_snapshot = snapshot | sched1
        self.assertEqual(sched_snapshot.name, 'snapshot_label')

    def test_flatten(self):
        """Test schedule flattening."""

        device = self.two_qubit_device
        chan = device.q[0].drive
        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3, name='pulse_name')

        sched = Schedule()
        for _ in range(10):
            sched += gp0(chan)

        flat_sched = sched.flatten()

        self.assertEqual(len(flat_sched._children), 10)

        self.assertEqual(flat_sched.instructions, sched.instructions)

    def test_buffering(self):
        """Test channel buffering."""
        buffer_chan = DriveChannel(0, buffer=5)
        measure_chan = MeasureChannel(0, buffer=10)
        acquire_chan = AcquireChannel(0, buffer=10)
        memory_slot = MemorySlot(0)
        gp0 = pulse_lib.gaussian(duration=10, amp=0.7, sigma=3)
        fc_pi_2 = FrameChange(phase=1.57)

        # no initial buffer
        sched = Schedule()
        sched += gp0(buffer_chan)

        self.assertEqual(sched.duration, 10)

        # this pulse should be buffered
        sched += gp0(buffer_chan)

        self.assertEqual(sched.duration, 25)

        # should not be buffered as framechange
        sched += fc_pi_2(buffer_chan)

        self.assertEqual(sched.duration, 25)

        # use buffer with insert
        sched = sched.insert(sched.duration, gp0(buffer_chan), buffer=True)

        self.assertEqual(sched.duration, 40)

        sched = Schedule()

        sched = gp0(measure_chan) + Acquire(duration=10)(acquire_chan, memory_slot)

        self.assertEqual(sched.duration, 10)


if __name__ == '__main__':
    unittest.main()
