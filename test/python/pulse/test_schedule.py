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

from qiskit.pulse.channels import (MemorySlot, RegisterSlot, DriveChannel, AcquireChannel,
                                   SnapshotChannel)
from qiskit.pulse.commands import (FrameChange, Acquire, PersistentValue, Snapshot, Delay,
                                   functional_pulse, Instruction, AcquireInstruction,
                                   PulseInstruction, FrameChangeInstruction)
from qiskit.pulse import pulse_lib, SamplePulse, CmdDef
from qiskit.pulse.timeslots import TimeslotCollection, Interval
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class BaseTestSchedule(QiskitTestCase):
    """Schedule tests."""

    def setUp(self):
        @functional_pulse
        def linear(duration, slope, intercept):
            x = np.linspace(0, duration - 1, duration)
            return slope * x + intercept
        self.linear = linear
        self.config = FakeOpenPulse2Q().configuration()


class TestScheduleBuilding(BaseTestSchedule):
    """Test construction of schedules."""

    def test_append_an_instruction_to_empty_schedule(self):
        """Test append instructions to an empty schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(self.config.drive(0)))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(3, sched.stop_time)

    def test_append_instructions_applying_to_different_channels(self):
        """Test append instructions to schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(lp0(self.config.drive(0)))
        sched = sched.append(lp0(self.config.drive(1)))
        self.assertEqual(0, sched.start_time)
        # appending to separate channel so should be at same time.
        self.assertEqual(3, sched.stop_time)

    def test_insert_an_instruction_into_empty_schedule(self):
        """Test insert an instruction into an empty schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(self.config.drive(0)))
        self.assertEqual(10, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_insert_an_instruction_before_an_existing_instruction(self):
        """Test insert an instruction before an existing instruction."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(self.config.drive(0)))
        sched = sched.insert(5, lp0(self.config.drive(0)))
        self.assertEqual(5, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_fail_to_insert_instruction_into_occupied_timing(self):
        """Test insert an instruction before an existing instruction."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, lp0(self.config.drive(0)))
        with self.assertRaises(PulseError):
            sched.insert(11, lp0(self.config.drive(0)))

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error."""
        gp0 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)

        sched = Schedule()
        sched = sched.append(gp0(self.config.drive(0)))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(self.config.control(0)))
        sched = sched.insert(60, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(30, gp1(self.config.drive(0)))
        sched = sched.insert(60, gp0(self.config.control(0)))
        sched = sched.insert(80, Snapshot("label", "snap_type"))
        sched = sched.insert(90, fc_pi_2(self.config.drive(0)))
        sched = sched.insert(90, acquire(self.config.acquire(0),
                                         MemorySlot(0),
                                         RegisterSlot(0)))
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
        gp0 = pulse_lib.gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)
        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)

        sched = Schedule()
        sched += gp0(self.config.drive(0))
        sched |= PersistentValue(value=0.2 + 0.4j)(self.config.control(0))
        sched |= FrameChange(phase=-1.57)(self.config.drive(0)) << 60
        sched |= gp1(self.config.drive(0)) << 30
        sched |= gp0(self.config.control(0)) << 60
        sched |= Snapshot("label", "snap_type") << 60
        sched |= fc_pi_2(self.config.drive(0)) << 90
        sched |= acquire(self.config.acquire(0),
                         MemorySlot(0),
                         RegisterSlot(0)) << 90
        sched += sched

    def test_immutability(self):
        """Test that operations are immutable."""
        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = gp1(self.config.drive(0)) << 100
        # if schedule was mutable the next two sequences would overlap and an error
        # would be raised.
        sched.union(gp0(self.config.drive(0)))
        sched.union(gp0(self.config.drive(0)))

    def test_inplace(self):
        """Test that in place operations on schedule are still immutable."""
        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = pulse_lib.gaussian(duration=20, amp=0.5, sigma=3)

        sched = Schedule()
        sched = sched + gp1(self.config.drive(0))
        sched2 = sched
        sched += gp0(self.config.drive(0))
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
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        # empty schedule with empty schedule
        empty = Schedule().append(Schedule())
        for _, instr in empty.instructions:
            self.assertIsInstance(instr, Instruction)

        # normal schedule
        subsched = Schedule()
        subsched = subsched.insert(20, lp0(self.config.drive(0)))  # grand child 1
        subsched = subsched.append(lp0(self.config.drive(0)))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(self.config.drive(0)))   # child
        sched = sched.append(subsched)
        for _, instr in sched.instructions:
            self.assertIsInstance(instr, Instruction)

    def test_absolute_start_time_of_grandchild(self):
        """Test correct calculation of start time of grandchild of a schedule."""
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(self.config.drive(0)))  # grand child 1
        subsched = subsched.append(lp0(self.config.drive(0)))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(self.config.drive(0)))   # child
        sched = sched.append(subsched)

        start_times = sorted([shft+instr.start_time for shft, instr in sched.instructions])
        self.assertEqual([0, 30, 40], start_times)

    def test_shift_schedule(self):
        """Test shift schedule."""
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, lp0(self.config.drive(0)))  # grand child 1
        subsched = subsched.append(lp0(self.config.drive(0)))      # grand child 2

        sched = Schedule()
        sched = sched.append(lp0(self.config.drive(0)))   # child
        sched = sched.append(subsched)

        shift = sched.shift(100)

        start_times = sorted([shft+instr.start_time for shft, instr in shift.instructions])

        self.assertEqual([100, 130, 140], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its _children after attached to another schedule."""
        acquire = Acquire(10)
        _children = (acquire(self.config.acquire(0), MemorySlot(0)).shift(20) +
                     acquire(self.config.acquire(0), MemorySlot(0)))
        self.assertEqual(2, len(list(_children.instructions)))

        sched = acquire(self.config.acquire(0), MemorySlot(0)).append(_children)
        self.assertEqual(3, len(list(sched.instructions)))

        # add 2 instructions to _children (2 instructions -> 4 instructions)
        _children = _children.append(acquire(self.config.acquire(0), MemorySlot(0)))
        _children = _children.insert(100, acquire(self.config.acquire(0),
                                                  MemorySlot(0)))
        self.assertEqual(4, len(list(_children.instructions)))
        # sched must keep 3 instructions (must not update to 5 instructions)
        self.assertEqual(3, len(list(sched.instructions)))

    def test_name_inherited(self):
        """Test that schedule keeps name if an instruction is added."""
        acquire = Acquire(10)
        gp0 = pulse_lib.gaussian(duration=100, amp=0.7, sigma=3, name='pulse_name')
        pv0 = PersistentValue(0.1)
        fc0 = FrameChange(0.1)
        snapshot = Snapshot('snapshot_label', 'state')

        sched1 = Schedule(name='test_name')
        sched2 = Schedule(name=None)
        sched3 = sched1 | sched2
        self.assertEqual(sched3.name, 'test_name')

        sched_acq = acquire(self.config.acquire(1),
                            MemorySlot(1),
                            name='acq_name') | sched1
        self.assertEqual(sched_acq.name, 'acq_name')

        sched_pulse = gp0(self.config.drive(0)) | sched1
        self.assertEqual(sched_pulse.name, 'pulse_name')

        sched_pv = pv0(self.config.drive(0), name='pv_name') | sched1
        self.assertEqual(sched_pv.name, 'pv_name')

        sched_fc = fc0(self.config.drive(0), name='fc_name') | sched1
        self.assertEqual(sched_fc.name, 'fc_name')

        sched_snapshot = snapshot | sched1
        self.assertEqual(sched_snapshot.name, 'snapshot_label')

    def test_buffering(self):
        """Test channel buffering."""
        buffer_chan = DriveChannel(0, buffer=5)
        gp0 = pulse_lib.gaussian(duration=10, amp=0.7, sigma=3)
        fc_pi_2 = FrameChange(phase=1.57)

        # no initial buffer
        sched = Schedule()
        sched += gp0(buffer_chan)
        self.assertEqual(sched.duration, 10)
        # this pulse should not be buffered
        sched += gp0(buffer_chan)
        self.assertEqual(sched.duration, 20)
        # should not be buffered as framechange
        sched += fc_pi_2(buffer_chan)
        self.assertEqual(sched.duration, 20)
        # use buffer with insert
        sched = sched.insert(sched.duration, gp0(buffer_chan), buffer=True)
        self.assertEqual(sched.duration, 30)

    def test_multiple_parameters_not_returned(self):
        """Constructing ParameterizedSchedule object from multiple ParameterizedSchedules sharing
        arguments should not produce repeated parameters in resulting ParameterizedSchedule
        object."""
        def my_test_par_sched_one(x, y, z):
            result = PulseInstruction(
                SamplePulse(np.array([x, y, z]), name='sample'),
                self.config.drive(0)
            )
            return 0, result

        def my_test_par_sched_two(x, y, z):
            result = PulseInstruction(
                SamplePulse(np.array([x, y, z]), name='sample'),
                self.config.drive(0)
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
        drive_ch = self.config.drive(0)
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

        measure_ch = self.config.measure(0)
        pulse = SamplePulse(np.full(10, 0.1))
        # should pass as is an append
        sched = self.delay(measure_ch) + pulse(measure_ch)
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = self.delay(measure_ch) | pulse(measure_ch)

    def test_delay_control_channel(self):
        """Test Delay on ControlChannel"""

        control_ch = self.config.control(0)
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

        acquire_ch = self.config.acquire(0)
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
    """Test Schedule filtering methods"""

    def test_filter_channels(self):
        """Test filtering over channels."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(60, acquire([AcquireChannel(0), AcquireChannel(1)],
                                         [MemorySlot(0), MemorySlot(1)]))
        sched = sched.insert(90, lp0(self.config.drive(0)))

        # split instructions for those on AcquireChannel(1) and those not
        filtered, excluded = self._filter_and_test_consistency(sched, channels=[AcquireChannel(1)])
        self.assertEqual(len(filtered.instructions), 1)
        self.assertEqual(len(excluded.instructions), 4)

        # Split schedule into the part with channels on 1 and into a part without
        channels = [AcquireChannel(1), DriveChannel(1)]
        filtered, excluded = self._filter_and_test_consistency(sched, channels=channels)
        for _, inst in filtered.instructions:
            self.assertTrue(any([chan in channels for chan in inst.channels]))

        for _, inst in excluded.instructions:
            self.assertFalse(any([chan in channels for chan in inst.channels]))

    def test_filter_inst_types(self):
        """Test filtering on instruction types."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(60, acquire([self.config.acquire(i) for i in range(2)],
                                         [MemorySlot(i) for i in range(2)]))
        sched = sched.insert(90, lp0(self.config.drive(0)))

        # test on Acquire
        only_acquire, no_acquire = \
            self._filter_and_test_consistency(sched, instruction_types=[AcquireInstruction])
        for _, inst in only_acquire.instructions:
            self.assertIsInstance(inst, AcquireInstruction)
        for _, inst in no_acquire.instructions:
            self.assertFalse(isinstance(inst, AcquireInstruction))

        # test two instruction types
        only_pulse_and_fc, no_pulse_and_fc = \
            self._filter_and_test_consistency(sched, instruction_types=[PulseInstruction,
                                                                        FrameChangeInstruction])
        for _, inst in only_pulse_and_fc.instructions:
            self.assertIsInstance(inst, (PulseInstruction, FrameChangeInstruction))
        for _, inst in no_pulse_and_fc.instructions:
            self.assertFalse(isinstance(inst, (PulseInstruction, FrameChangeInstruction)))
        self.assertEqual(len(only_pulse_and_fc.instructions), 4)
        self.assertEqual(len(no_pulse_and_fc.instructions), 1)

        # test on FrameChange
        only_fc, no_fc = \
            self._filter_and_test_consistency(sched, instruction_types={FrameChangeInstruction})
        self.assertEqual(len(only_fc.instructions), 1)
        self.assertEqual(len(no_fc.instructions), 4)

    def test_filter_intervals(self):
        """Test filtering on intervals."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(60, acquire([self.config.acquire(i) for i in range(2)],
                                         [MemorySlot(i) for i in range(2)]))
        sched = sched.insert(90, lp0(self.config.drive(0)))

        # split schedule into instructions occuring in (0,13), and those outside
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=((0, 13),))
        for start_time, inst in filtered.instructions:
            self.assertTrue((start_time >= 0) and (start_time + inst.stop_time <= 13))
        for start_time, inst in excluded.instructions:
            self.assertFalse((start_time >= 0) and (start_time + inst.stop_time <= 13))
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 3)

        # split into schedule occuring in and outside of interval (59,65)
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=[(59, 65)])
        self.assertEqual(len(filtered.instructions), 1)
        self.assertEqual(filtered.instructions[0][0], 60)
        self.assertIsInstance(filtered.instructions[0][1], AcquireInstruction)
        self.assertEqual(len(excluded.instructions), 4)
        self.assertEqual(excluded.instructions[3][0], 90)
        self.assertIsInstance(excluded.instructions[3][1], PulseInstruction)

        # split instructions based on the interval
        # (none should be, though they have some overlap with some of the instructions)
        filtered, excluded = \
            self._filter_and_test_consistency(sched, time_ranges=[(0, 2), (8, 11), (61, 70)])
        self.assertEqual(len(filtered.instructions), 0)
        self.assertEqual(len(excluded.instructions), 5)

        # split instructions from multiple non-overlapping intervals, specified
        # as time ranges
        filtered, excluded = \
            self._filter_and_test_consistency(sched, time_ranges=[(10, 15), (63, 93)])
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 3)

        # split instructions from non-overlapping intervals, specified as Intervals
        filtered, excluded = \
            self._filter_and_test_consistency(sched, intervals=[Interval(10, 15), Interval(63, 93)])
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 3)

    def test_filter_multiple(self):
        """Test filter composition."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(60, acquire([self.config.acquire(i) for i in range(2)],
                                         [MemorySlot(i) for i in range(2)]))
        sched = sched.insert(90, lp0(self.config.drive(0)))

        # split instructions with filters on channel 0, of type PulseInstruction,
        # occuring in the time interval (25, 100)
        filtered, excluded = self._filter_and_test_consistency(sched,
                                                               channels={self.config.drive(0)},
                                                               instruction_types=[PulseInstruction],
                                                               time_ranges=[(25, 100)])
        for time, inst in filtered.instructions:
            self.assertIsInstance(inst, PulseInstruction)
            self.assertTrue(all([chan.index == 0 for chan in inst.channels]))
            self.assertTrue(25 <= time <= 100)
        self.assertEqual(len(excluded.instructions), 4)
        self.assertTrue(excluded.instructions[0][1].channels[0] == DriveChannel(0))
        self.assertTrue(excluded.instructions[2][0] == 30)

        # split based on PulseInstructions in the specified intervals
        filtered, excluded = self._filter_and_test_consistency(sched,
                                                               instruction_types=[PulseInstruction],
                                                               time_ranges=[(25, 100), (0, 11)])
        self.assertTrue(len(excluded.instructions), 3)
        for time, inst in filtered.instructions:
            self.assertIsInstance(inst, (FrameChangeInstruction, PulseInstruction))
        self.assertTrue(len(filtered.instructions), 4)
        # make sure the PulseInstruction not in the intervals is maintained
        self.assertIsInstance(excluded.instructions[0][1], PulseInstruction)

    def test_custom_filters(self):
        """Test custom filters."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: True)
        for i in filtered.instructions:
            self.assertTrue(i in sched.instructions)
        for i in excluded.instructions:
            self.assertFalse(i in sched.instructions)

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: False)
        self.assertEqual(len(filtered.instructions), 0)
        self.assertEqual(len(excluded.instructions), 3)

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: x[0] < 30)
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 1)

        # multiple custom filters
        filtered, excluded = self._filter_and_test_consistency(sched,
                                                               lambda x: x[0] > 0,
                                                               lambda x: x[0] < 30)
        self.assertEqual(len(filtered.instructions), 1)
        self.assertEqual(len(excluded.instructions), 2)

    def test_empty_filters(self):
        """Test behavior on empty filters."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        acquire = Acquire(5)
        sched = Schedule(name='fake_experiment')
        sched = sched.insert(0, lp0(self.config.drive(0)))
        sched = sched.insert(10, lp0(self.config.drive(1)))
        sched = sched.insert(30, FrameChange(phase=-1.57)(self.config.drive(0)))
        sched = sched.insert(60, acquire([self.config.acquire(i) for i in range(2)],
                                         [MemorySlot(i) for i in range(2)]))
        sched = sched.insert(90, lp0(self.config.drive(0)))

        # empty channels
        filtered, excluded = self._filter_and_test_consistency(sched, channels=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 5)

        # empty instruction_types
        filtered, excluded = self._filter_and_test_consistency(sched, instruction_types=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 5)

        # empty time_ranges
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 5)

        # empty intervals
        filtered, excluded = self._filter_and_test_consistency(sched, intervals=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 5)

        # empty channels with other non-empty filters
        filtered, excluded = self._filter_and_test_consistency(sched,
                                                               channels=[],
                                                               instruction_types=[PulseInstruction])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 5)

    def _filter_and_test_consistency(self, schedule: Schedule, *args, **kwargs):
        """
        Returns the tuple
        (schedule.filter(*args, **kwargs), schedule.exclude(*args, **kwargs)),
        including a test that schedule.filter | schedule.exclude == schedule
        """
        filtered = schedule.filter(*args, **kwargs)
        excluded = schedule.exclude(*args, **kwargs)
        self.assertEqual(filtered | excluded, schedule)
        return filtered, excluded


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
