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

from qiskit.pulse.channels import (PulseChannelSpec, MemorySlot, DriveChannel, AcquireChannel,
                                   MeasureChannel, SnapshotChannel)
from qiskit.pulse.commands import (FrameChange, Acquire, PersistentValue, Snapshot, Delay,
                                   functional_pulse, Instruction, AcquireInstruction,
                                   PulseInstruction, FrameChangeInstruction)
from qiskit.pulse import pulse_lib, SamplePulse, CmdDef
from qiskit.pulse.timeslots import TimeslotCollection, Interval
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule
from qiskit.test import QiskitTestCase


class BaseTestSchedule(QiskitTestCase):
    """Schedule tests."""

    def setUp(self):
        @functional_pulse
        def linear(duration, slope, intercept):
            x = np.linspace(0, duration - 1, duration)
            return slope * x + intercept

        self.linear = linear
        self.two_qubit_device = PulseChannelSpec(n_qubits=2, n_control=1, n_registers=2)


class TestScheduleBuilding(BaseTestSchedule):
    """Test construction of schedules."""

    def test_append_an_instruction_to_empty_schedule(self):
        """Test append instructions to an empty schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(device.drives[0]))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(3, sched.stop_time)

    def test_append_instructions_applying_to_different_channels(self):
        """Test append instructions to schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(device.drives[0]))
        sched = sched.append(lp0(device.drives[1]))
        self.assertEqual(0, sched.start_time)
        # appending to separate channel so should be at same time.
        self.assertEqual(3, sched.stop_time)

    def test_insert_an_instruction_into_empty_schedule(self):
        """Test insert an instruction into an empty schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.drives[0]))
        self.assertEqual(10, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_insert_an_instruction_before_an_existing_instruction(self):
        """Test insert an instruction before an existing instruction."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.drives[0]))
        sched = sched.insert(5, lp0(device.drives[0]))
        self.assertEqual(5, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_fail_to_insert_instruction_into_occupied_timing(self):
        """Test insert an instruction before an existing instruction."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(device.drives[0]))
        with self.assertRaises(PulseError):
            sched.insert(11, lp0(device.drives[0]))

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)

        sched = Schedule()
        sched = sched.append(gp0(device.drives[0]))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(device.controls[0]))
        sched = sched.insert(60, FrameChange(phase=-1.57)(device.drives[0]))
        sched = sched.insert(30, gp1(device.drives[1]))
        sched = sched.insert(60, gp0(device.controls[0]))
        sched = sched.insert(80, Snapshot("label", "snap_type"))
        sched = sched.insert(90, fc_pi_2(device.drives[0]))
        sched = sched.insert(90, acquire(device.acquires[1],
                                         device.memoryslots[1],
                                         device.registers[1]))
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
        sched += gp0(device.drives[0])
        sched |= PersistentValue(value=0.2 + 0.4j)(device.controls[0])
        sched |= FrameChange(phase=-1.57)(device.drives[0]) << 60
        sched |= gp1(device.drives[1]) << 30
        sched |= gp0(device.controls[0]) << 60
        sched |= Snapshot("label", "snap_type") << 60
        sched |= fc_pi_2(device.drives[0]) << 90
        sched |= acquire(device.acquires[1],
                         device.memoryslots[1],
                         device.registers[1]) << 90
        sched += sched

    def test_immutability(self):
        """Test that operations are immutable."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = gp1(device.drives[0]) << 100
        # if schedule was mutable the next two sequences would overlap and an error
        # would be raised.
        sched.union(gp0(device.drives[0]))
        sched.union(gp0(device.drives[0]))

    def test_inplace(self):
        """Test that in place operations on schedule are still immutable."""
        device = self.two_qubit_device

        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = Schedule()
        sched = sched + gp1(device.drives[0])
        sched2 = sched
        sched += gp0(device.drives[0])
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
        subsched = subsched.insert(20, lp0(device.drives[0]))  # grand child 1
        subsched = subsched.append(lp0(device.drives[0]))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.drives[0]))   # child
        sched = sched.append(subsched)
        for _, instr in sched.instructions:
            self.assertIsInstance(instr, Instruction)

    def test_absolute_start_time_of_grandchild(self):
        """Test correct calculation of start time of grandchild of a schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.drives[0]))  # grand child 1
        subsched = subsched.append(lp0(device.drives[0]))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.drives[0]))   # child
        sched = sched.append(subsched)

        start_times = sorted([shft+instr.start_time for shft, instr in sched.instructions])
        self.assertEqual([0, 30, 40], start_times)

    def test_shift_schedule(self):
        """Test shift schedule."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(device.drives[0]))  # grand child 1
        subsched = subsched.append(lp0(device.drives[0]))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(device.drives[0]))   # child
        sched = sched.append(subsched)

        shift = sched.shift(100)

        start_times = sorted([shft+instr.start_time for shft, instr in shift.instructions])

        self.assertEqual([100, 130, 140], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its _children after attached to another schedule."""
        device = self.two_qubit_device

        acquire = Acquire(10)
        _children = (acquire(device.acquires[1], device.memoryslots[1]).shift(20) +
                     acquire(device.acquires[1], device.memoryslots[1]))
        self.assertEqual(2, len(list(_children.instructions)))

        sched = acquire(device.acquires[1], device.memoryslots[1]).append(_children)
        self.assertEqual(3, len(list(sched.instructions)))

        # add 2 instructions to _children (2 instructions -> 4 instructions)
        _children = _children.append(acquire(device.acquires[1], device.memoryslots[1]))
        _children = _children.insert(100, acquire(device.acquires[1], device.memoryslots[1]))
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

        sched_acq = acquire(device.acquires[1], device.memoryslots[1], name='acq_name') | sched1
        self.assertEqual(sched_acq.name, 'acq_name')

        sched_pulse = gp0(device.drives[0]) | sched1
        self.assertEqual(sched_pulse.name, 'pulse_name')

        sched_pv = pv0(device.drives[0], name='pv_name') | sched1
        self.assertEqual(sched_pv.name, 'pv_name')

        sched_fc = fc0(device.drives[0], name='fc_name') | sched1
        self.assertEqual(sched_fc.name, 'fc_name')

        sched_snapshot = snapshot | sched1
        self.assertEqual(sched_snapshot.name, 'snapshot_label')

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

    def test_multiple_parameters_not_returned(self):
        """Constructing ParameterizedSchedule object from multiple ParameterizedSchedules sharing
        arguments should not produce repeated parameters in resulting ParameterizedSchedule
        object."""
        device = self.two_qubit_device

        def my_test_par_sched_one(x, y, z):
            result = PulseInstruction(
                SamplePulse(np.array([x, y, z]), name='sample'),
                device.drives[0]
            )
            return 0, result

        def my_test_par_sched_two(x, y, z):
            result = PulseInstruction(
                SamplePulse(np.array([x, y, z]), name='sample'),
                device.drives[0]
            )
            return 5, result

        par_sched_in_0 = ParameterizedSchedule(
            my_test_par_sched_one, parameters={'x': 0, 'y': 1, 'z': 2}
        )
        par_sched_in_1 = ParameterizedSchedule(
            my_test_par_sched_two, parameters={'x': 0, 'y': 1, 'z': 2}
        )
        par_sched = ParameterizedSchedule(par_sched_in_0, par_sched_in_1)

        cmd_def = CmdDef()
        cmd_def.add('test', 0, par_sched)

        actual = cmd_def.get('test', 0, 0.01, 0.02, 0.03)
        expected = par_sched_in_0.bind_parameters(0.01, 0.02, 0.03) |\
            par_sched_in_1.bind_parameters(0.01, 0.02, 0.03)
        self.assertEqual(actual.start_time, expected.start_time)
        self.assertEqual(actual.stop_time, expected.stop_time)

        self.assertEqual(cmd_def.get_parameters('test', 0), ('x', 'y', 'z'))


class TestDelay(BaseTestSchedule):
    """Test Delay Instruction"""

    def setUp(self):
        super().setUp()
        self.delay_time = 10
        self.delay = Delay(self.delay_time)

    def test_delay_drive_channel(self):
        """Test Delay on DriveChannel"""

        drive_ch = self.two_qubit_device.qubits[0].drive
        pulse = SamplePulse(np.full(10, 0.1))
        # should pass as is an append
        sched = self.delay(drive_ch) + pulse(drive_ch)
        self.assertIsInstance(sched, Schedule)
        pulse_instr = sched.instructions[-1]
        # assert last instruction is pulse
        self.assertIsInstance(pulse_instr[1], PulseInstruction)
        # assert pulse is scheduled at time 10
        self.assertEqual(pulse_instr[0], 10)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(drive_ch) | pulse(drive_ch)

    def test_delay_measure_channel(self):
        """Test Delay on MeasureChannel"""

        measure_ch = self.two_qubit_device.qubits[0].measure
        pulse = SamplePulse(np.full(10, 0.1))
        # should pass as is an append
        sched = self.delay(measure_ch) + pulse(measure_ch)
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(measure_ch) | pulse(measure_ch)

    def test_delay_control_channel(self):
        """Test Delay on ControlChannel"""

        control_ch = self.two_qubit_device.controls[0]
        pulse = SamplePulse(np.full(10, 0.1))
        # should pass as is an append
        sched = self.delay(control_ch) + pulse(control_ch)
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(control_ch) | pulse(control_ch)
            self.assertIsInstance(sched, Schedule)

    def test_delay_acquire_channel(self):
        """Test Delay on DriveChannel"""

        acquire_ch = self.two_qubit_device.qubits[0].acquire
        acquire = Acquire(10)
        # should pass as is an append
        sched = self.delay(acquire_ch) + acquire(acquire_ch, MemorySlot(0))
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(acquire_ch) | acquire(acquire_ch)
            self.assertIsInstance(sched, Schedule)

    def test_delay_snapshot_channel(self):
        """Test Delay on DriveChannel"""

        snapshot_ch = SnapshotChannel()
        snapshot = Snapshot(label='test')
        # should pass as is an append
        sched = self.delay(snapshot_ch) + snapshot
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(snapshot_ch) | snapshot << 5
            self.assertIsInstance(sched, Schedule)


class TestScheduleFilter(BaseTestSchedule):
    """Test Schedule filtering methods."""

    def test_filter_channels(self):
        """Test filtering over channels."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(device.drives[0]))
        sched = sched.insert(10, lp0(device.drives[1]))
        sched = sched.insert(30, FrameChange(phase=-1.57)(device.drives[0]))
        sched = sched.insert(60, acquire(device.acquires, device.memoryslots))
        sched = sched.insert(90, lp0(device.drives[0]))

        self.assertEqual(len(sched.filter(channels=[AcquireChannel(1)]).instructions), 1)
        channels = [AcquireChannel(1), DriveChannel(1)]
        has_chan_1 = sched.filter(channels=channels)
        for _, inst in has_chan_1.instructions:
            self.assertTrue(any([chan in channels for chan in inst.channels]))

    def test_filter_inst_types(self):
        """Test filtering on instruction types."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(device.drives[0]))
        sched = sched.insert(10, lp0(device.drives[1]))
        sched = sched.insert(30, FrameChange(phase=-1.57)(device.drives[0]))
        sched = sched.insert(60, acquire(device.acquires, device.memoryslots))
        sched = sched.insert(90, lp0(device.drives[0]))

        only_acquires = sched.filter(instruction_types=[AcquireInstruction])
        for _, inst in only_acquires.instructions:
            self.assertIsInstance(inst, AcquireInstruction)
        only_pulse_and_fc = sched.filter(instruction_types=[PulseInstruction,
                                                            FrameChangeInstruction])
        for _, inst in only_pulse_and_fc.instructions:
            self.assertIsInstance(inst, (PulseInstruction, FrameChangeInstruction))
        self.assertEqual(len(only_pulse_and_fc.instructions), 4)
        only_fc = sched.filter(instruction_types={FrameChangeInstruction})
        self.assertEqual(len(only_fc.instructions), 1)

    def test_filter_intervals(self):
        """Test filtering on intervals."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(device.drives[0]))
        sched = sched.insert(10, lp0(device.drives[1]))
        sched = sched.insert(30, FrameChange(phase=-1.57)(device.drives[0]))
        sched = sched.insert(60, acquire(device.acquires, device.memoryslots))
        sched = sched.insert(90, lp0(device.drives[0]))

        intervals_a = sched.filter(time_ranges=((0, 13),))
        for time, inst in intervals_a.instructions:
            self.assertTrue(0 <= time <= 13)
            self.assertTrue(inst.timeslots.timeslots[0].interval.stop <= 13)
        self.assertEqual(len(intervals_a.instructions), 2)

        intervals_b = sched.filter(time_ranges=[(59, 65)])
        self.assertEqual(len(intervals_b.instructions), 1)
        self.assertEqual(intervals_b.instructions[0][0], 60)
        self.assertIsInstance(intervals_b.instructions[0][1], AcquireInstruction)

        non_full_intervals = sched.filter(time_ranges=[(0, 2), (8, 11), (61, 70)])
        self.assertEqual(len(non_full_intervals.instructions), 0)

        multi_interval = sched.filter(time_ranges=[(10, 15), (63, 93)])
        self.assertEqual(len(multi_interval.instructions), 2)

        multi_interval = sched.filter(intervals=[Interval(10, 15), Interval(63, 93)])
        self.assertEqual(len(multi_interval.instructions), 2)

    def test_filter_multiple(self):
        """Test filter composition."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(device.drives[0]))
        sched = sched.insert(10, lp0(device.drives[1]))
        sched = sched.insert(30, FrameChange(phase=-1.57)(device.drives[0]))
        sched = sched.insert(60, acquire(device.acquires, device.memoryslots))
        sched = sched.insert(90, lp0(device.drives[0]))

        filtered = sched.filter(channels={0}, instruction_types=[PulseInstruction],
                                time_ranges=[(25, 100)])
        for time, inst in filtered.instructions:
            self.assertIsInstance(inst, PulseInstruction)
            self.assertTrue(any([chan.index == 0 for chan in inst.channels]))
            self.assertTrue(25 <= time <= 100)

        filtered_b = sched.filter(instruction_types=[PulseInstruction, FrameChangeInstruction],
                                  time_ranges=[(25, 100), (0, 30)])
        for time, inst in filtered_b.instructions:
            self.assertIsInstance(inst, (FrameChangeInstruction, PulseInstruction))
        self.assertTrue(len(filtered_b.instructions), 4)

    def test_filter(self):
        """Test _filter method."""
        device = self.two_qubit_device
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(device.drives[0]))
        sched = sched.insert(10, lp0(device.drives[1]))
        sched = sched.insert(30, FrameChange(phase=-1.57)(device.drives[0]))
        for i in sched._filter([lambda x: True]).instructions:
            self.assertTrue(i in sched.instructions)
        self.assertEqual(len(sched._filter([lambda x: False]).instructions), 0)
        self.assertEqual(len(sched._filter([lambda x: x[0] < 30]).instructions), 2)


class TestScheduleEquality(BaseTestSchedule):
    """Test equality of schedules."""

    def test_different_channels(self):
        """Test equality is False if different channels."""
        self.assertNotEqual(Schedule(FrameChange(0)(DriveChannel(0))),
                            Schedule(FrameChange(0)(DriveChannel(1))))

    def test_same_time_equal(self):
        """Test equal if instruction at same time."""

        self.assertEqual(Schedule((0, FrameChange(0)(DriveChannel(1)))),
                         Schedule((0, FrameChange(0)(DriveChannel(1)))))

    def test_different_time_not_equal(self):
        """Test that not equal if instruction at different time."""
        self.assertNotEqual(Schedule((0, FrameChange(0)(DriveChannel(1)))),
                            Schedule((1, FrameChange(0)(DriveChannel(1)))))

    def test_single_channel_out_of_order(self):
        """Test that schedule with single channel equal when out of order."""
        instructions = [(0, FrameChange(0)(DriveChannel(0))),
                        (15, SamplePulse(np.ones(10))(DriveChannel(0))),
                        (5, SamplePulse(np.ones(10))(DriveChannel(0)))]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_multiple_channels_out_of_order(self):
        """Test that schedule with multiple channels equal when out of order."""
        instructions = [(0, FrameChange(0)(DriveChannel(1))),
                        (1, Acquire(10)(AcquireChannel(0), MemorySlot(1)))]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_different_name_equal(self):
        """Test that names are ignored when checking equality."""

        self.assertEqual(Schedule((0, FrameChange(0, name='fc1')(DriveChannel(1))), name='s1'),
                         Schedule((0, FrameChange(0, name='fc2')(DriveChannel(1))), name='s2'))


if __name__ == '__main__':
    unittest.main()
