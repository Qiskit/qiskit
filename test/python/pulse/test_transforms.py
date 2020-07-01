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

"""Test cases for the pulse Schedule transforms."""
import unittest
from typing import List, Set

import numpy as np

from qiskit import pulse
from qiskit.pulse import (Play, Delay, Acquire, Schedule, SamplePulse, Drag,
                          Gaussian, GaussianSquare, Constant, instructions,
                          pulse_lib, transforms)
from qiskit.pulse.channels import MeasureChannel, MemorySlot, DriveChannel, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.passmanager import PassManager
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q

from qiskit.pulse.transforms import add_implicit_acquires, align_measures, pad, compress_pulses


class TestAlignMeasures(QiskitTestCase):
    """Test the helper function which aligns acquires."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.inst_map = self.backend.defaults().instruction_schedule_map
        self.short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                             name='p0')

    def test_align_measures(self):
        """Test that one acquire is delayed to match the time of the later acquire."""
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(1, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = sched.insert(10, Acquire(5, self.config.acquire(1), MemorySlot(1)))
        sched = sched.insert(10, Play(self.short_pulse, self.config.measure(0)))
        sched = sched.insert(10, Play(self.short_pulse, self.config.measure(1)))
        sched = align_measures([sched], self.inst_map)[0]
        self.assertEqual(sched.name, 'fake_experiment')
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 10)
        sched = align_measures([sched], self.inst_map, align_time=20)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 20)
            if isinstance(inst.channels[0], MeasureChannel):
                self.assertEqual(time, 20)

    def test_align_post_u3(self):
        """Test that acquires are scheduled no sooner than the duration of the longest X gate.
        """
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(1, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = align_measures([sched], self.inst_map)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 4)
        sched = align_measures([sched], self.inst_map, max_calibration_duration=10)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 10)

    def test_multi_acquire(self):
        """Test that an error is raised if multiple acquires occur on the same channel."""
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(4, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = sched.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        with self.assertRaises(PulseError):
            align_measures([sched], self.inst_map)

        # Test for measure channel
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(10, Play(self.short_pulse, self.config.measure(0)))
        sched = sched.insert(30, Play(self.short_pulse, self.config.measure(0)))
        with self.assertRaises(PulseError):
            align_measures([sched], self.inst_map)

        # Test both using inst_map
        sched = pulse.Schedule()
        sched += self.inst_map.get('measure', (0, 1))
        align_measures([sched], align_time=50)
        sched += self.inst_map.get('measure', (0, 1))
        with self.assertRaises(PulseError):
            align_measures([sched], align_time=50)

    def test_error_post_acquire_pulse(self):
        """Test that an error is raised if a pulse occurs on a channel after an acquire."""
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(4, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        # No error with separate channel
        sched = sched.insert(10, Play(self.short_pulse, self.config.drive(1)))
        align_measures([sched], self.inst_map)
        sched = sched.insert(10, Play(self.short_pulse, self.config.drive(0)))
        with self.assertRaises(PulseError):
            align_measures([sched], self.inst_map)

    def test_align_across_schedules(self):
        """Test that acquires are aligned together across multiple schedules."""
        sched1 = pulse.Schedule(name='fake_experiment')
        sched1 = sched1.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched1 = sched1.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched2 = pulse.Schedule(name='fake_experiment')
        sched2 = sched2.insert(3, Play(self.short_pulse, self.config.drive(0)))
        sched2 = sched2.insert(25, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        schedules = align_measures([sched1, sched2], self.inst_map)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 25)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 25)


class TestAddImplicitAcquires(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                             name='p0')
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(5, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = sched.insert(5, Acquire(5, self.config.acquire(1), MemorySlot(1)))
        self.sched = sched

    def test_add_implicit(self):
        """Test that implicit acquires are made explicit according to the meas map."""
        sched = add_implicit_acquires(self.sched, [[0, 1]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1})

    def test_add_across_meas_map_sublists(self):
        """Test that implicit acquires in separate meas map sublists are all added."""
        sched = add_implicit_acquires(self.sched, [[0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_dont_add_all(self):
        """Test that acquires aren't added if no qubits in the sublist aren't being acquired."""
        sched = add_implicit_acquires(self.sched, [[4, 5], [0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_multiple_acquires(self):
        """Test for multiple acquires."""
        sched = pulse.Schedule()
        acq_q0 = pulse.Acquire(1200, AcquireChannel(0), MemorySlot(0))
        sched += acq_q0
        sched += acq_q0 << sched.duration
        sched = add_implicit_acquires(sched, meas_map=[[0]])
        self.assertEqual(sched.instructions, ((0, acq_q0), (2400, acq_q0)))


class TestPad(QiskitTestCase):
    """Test padding of schedule with delays."""

    def test_padding_empty_schedule(self):
        """Test padding of empty schedule."""
        self.assertEqual(pulse.Schedule(), pad(pulse.Schedule()))

    def test_padding_schedule(self):
        """Test padding schedule."""
        delay = 10
        sched = (Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(1)).shift(10))

        ref_sched = (sched |
                     Delay(delay, DriveChannel(0)) |
                     Delay(delay, DriveChannel(0)).shift(20) |
                     Delay(delay, DriveChannel(1)) |
                     Delay(2 * delay, DriveChannel(1)).shift(20))

        self.assertEqual(pad(sched), ref_sched)

    def test_padding_schedule_inverse_order(self):
        """Test padding schedule is insensitive to order in which commands were added.

        This test is the same as `test_adding_schedule` but the order by channel
        in which commands were added to the schedule to be padded has been reversed.
        """
        delay = 10
        sched = (Delay(delay, DriveChannel(1)).shift(10) +
                 Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(0)).shift(10))

        ref_sched = (sched |
                     Delay(delay, DriveChannel(0)) |
                     Delay(delay, DriveChannel(0)).shift(20) |
                     Delay(delay, DriveChannel(1)) |
                     Delay(2 * delay, DriveChannel(1)).shift(20))

        self.assertEqual(pad(sched), ref_sched)

    def test_padding_until_less(self):
        """Test padding until time that is less than schedule duration."""
        delay = 10

        sched = (Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(1)))

        ref_sched = (sched |
                     Delay(delay, DriveChannel(0)) |
                     Delay(5, DriveChannel(1)).shift(10))

        self.assertEqual(pad(sched, until=15), ref_sched)

    def test_padding_until_greater(self):
        """Test padding until time that is greater than schedule duration."""
        delay = 10

        sched = (Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(1)))

        ref_sched = (sched |
                     Delay(delay, DriveChannel(0)) |
                     Delay(30, DriveChannel(0)).shift(20) |
                     Delay(40, DriveChannel(1)).shift(10))

        self.assertEqual(pad(sched, until=50), ref_sched)

    def test_padding_supplied_channels(self):
        """Test padding of only specified channels."""
        delay = 10
        sched = (Delay(delay, DriveChannel(0)).shift(10) +
                 Delay(delay, DriveChannel(1)))

        ref_sched = (sched |
                     Delay(delay, DriveChannel(0)) |
                     Delay(2 * delay, DriveChannel(2)))

        channels = [DriveChannel(0), DriveChannel(2)]

        self.assertEqual(pad(sched, channels=channels), ref_sched)

    def test_padding_less_than_sched_duration(self):
        """Test that the until arg is respected even for less than the input schedule duration."""
        delay = 10
        sched = (Delay(delay, DriveChannel(0)) +
                 Delay(delay, DriveChannel(0)).shift(20))
        ref_sched = (sched | pulse.Delay(5, DriveChannel(0)).shift(10))
        self.assertEqual(pad(sched, until=15), ref_sched)


def get_pulse_ids(schedules: List[Schedule]) -> Set[int]:
    """Returns ids of pulses used in Schedules."""
    ids = set()
    for schedule in schedules:
        for _, inst in schedule.instructions:
            ids.add(inst.pulse.id)
    return ids


class TestCompressTransform(QiskitTestCase):
    """Compress function test."""

    def test_with_duplicates(self):
        """Test compression of schedule."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_sample_pulse_with_clipping(self):
        """Test sample pulses with clipping."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 1.0]), drive_channel)
        schedule += Play(SamplePulse([0.0, 1.001], epsilon=1e-3), drive_channel)
        schedule += Play(SamplePulse([0.0, 1.0000000001]), drive_channel)

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 3)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_no_duplicates(self):
        """Test with no pulse duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 1.0]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.9]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.3]), drive_channel)

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_parametric_pulses_with_duplicates(self):
        """Test with parametric pulses."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 8)
        self.assertEqual(len(compressed_pulse_ids), 4)

    def test_parametric_pulses_with_no_duplicates(self):
        """Test parametric pulses with no duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.49j), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.19,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.1 + 0.41j), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.31j, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_with_different_channels(self):
        """Test with different channels."""
        schedule = Schedule()
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(1))

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)

    def test_sample_pulses_with_tolerance(self):
        """Test sample pulses with tolerance."""
        schedule = Schedule()
        schedule += Play(SamplePulse([0.0, 0.1001], epsilon=1e-3), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1], epsilon=1e-3), DriveChannel(1))

        compressed_schedule = compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)

    def test_multiple_schedules(self):
        """Test multiple schedules."""
        schedules = []
        for _ in range(2):
            schedule = Schedule()
            drive_channel = DriveChannel(0)
            schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
            schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
            schedule += Play(SamplePulse([0.0, 0.2]), drive_channel)
            schedules.append(schedule)

        compressed_schedule = compress_pulses(schedules)
        original_pulse_ids = get_pulse_ids(schedules)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 6)
        self.assertEqual(len(compressed_pulse_ids), 2)


class TestFoldShiftPhase(QiskitTestCase):
    """Test compress shift phase transforms."""
    def compress_shift_phase(self, schedule):
        pm = PassManager()
        pm.append(transforms.FoldShiftPhase())
        return pm.run(pulse.Program(schedules=[schedule])).schedules[0]

    def test_consecutive_phase(self):
        d0 = DriveChannel(0)
        sched = Schedule()
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.Delay(10, d0)

        opt_sched = self.compress_shift_phase(sched.flatten())

        ref_sched = Schedule()
        ref_sched += instructions.ShiftPhase(np.pi, d0)
        ref_sched += instructions.Delay(10, d0)

        self.assertEqual(opt_sched, ref_sched)

    def test_different_channels(self):
        d0 = DriveChannel(0)
        d1 = DriveChannel(1)
        sched = Schedule()
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.ShiftPhase(np.pi/2, d1)
        sched += instructions.Delay(10, d0)
        sched += instructions.Delay(10, d1)

        ref_sched = Schedule()
        ref_sched += instructions.ShiftPhase(np.pi, d0)
        ref_sched += instructions.ShiftPhase(np.pi/2, d1)
        ref_sched += instructions.Delay(10, d0)
        ref_sched += instructions.Delay(10, d1)

        opt_sched = self.compress_shift_phase(sched.flatten())
        self.assertEqual(opt_sched, ref_sched)

    def test_interruped_phases(self):
        d0 = DriveChannel(0)
        sched = Schedule()
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.Delay(10, d0)
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.ShiftPhase(np.pi/2, d0)
        sched += instructions.Delay(10, d0)

        opt_sched = self.compress_shift_phase(sched.flatten())

        ref_sched = Schedule()
        ref_sched += instructions.ShiftPhase(np.pi/2, d0)
        ref_sched += instructions.Delay(10, d0)
        ref_sched += instructions.ShiftPhase(np.pi, d0)
        ref_sched += instructions.Delay(10, d0)

        self.assertEqual(opt_sched, ref_sched)


class TestFoldShiftFrequency(QiskitTestCase):
    """Test compress shift frequency transforms."""
    def compress_shift_frequency(self, schedule):
        pm = PassManager()
        pm.append(transforms.FoldShiftFrequency())
        return pm.run(pulse.Program(schedules=[schedule])).schedules[0]

    def test_consecutive_frequencies(self):
        d0 = DriveChannel(0)
        sched = Schedule()
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.Delay(10, d0)

        opt_sched = self.compress_shift_frequency(sched.flatten())

        ref_sched = Schedule()
        ref_sched += instructions.ShiftFrequency(1, d0)
        ref_sched += instructions.Delay(10, d0)

        self.assertEqual(opt_sched, ref_sched)

    def test_different_channels(self):
        d0 = DriveChannel(0)
        d1 = DriveChannel(1)
        sched = Schedule()
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.ShiftFrequency(1/2, d1)
        sched += instructions.Delay(10, d0)
        sched += instructions.Delay(10, d1)

        ref_sched = Schedule()
        ref_sched += instructions.ShiftFrequency(1, d0)
        ref_sched += instructions.ShiftFrequency(1/2, d1)
        ref_sched += instructions.Delay(10, d0)
        ref_sched += instructions.Delay(10, d1)

        opt_sched = self.compress_shift_frequency(sched.flatten())
        self.assertEqual(opt_sched, ref_sched)

    def test_interruped_frequencies(self):
        d0 = DriveChannel(0)
        sched = Schedule()
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.Delay(10, d0)
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.ShiftFrequency(1/2, d0)
        sched += instructions.Delay(10, d0)

        opt_sched = self.compress_shift_frequency(sched.flatten())

        ref_sched = Schedule()
        ref_sched += instructions.ShiftFrequency(1/2, d0)
        ref_sched += instructions.Delay(10, d0)
        ref_sched += instructions.ShiftFrequency(1, d0)
        ref_sched += instructions.Delay(10, d0)

        self.assertEqual(opt_sched, ref_sched)


class TestTruncateWaveformPrecision(QiskitTestCase):
    """Test truncation of waveform precision."""
    def truncate_pulses(self, schedule):
        pm = PassManager()
        pm.append(transforms.TruncateWaveformPrecision())
        return pm.run(pulse.Program(schedules=[schedule])).schedules[0]

    def test_truncation_of_waveform(self):
        d0 = DriveChannel(0)
        d1 = DriveChannel(1)

        pulse0 = pulse_lib.SamplePulse([0.000055, 1.])
        trunc_pulse0 = pulse_lib.SamplePulse([0.00006, 1.])
        pulse1 = pulse_lib.SamplePulse([1., 1.])
        trunc_pulse1 = pulse_lib.SamplePulse([1., 1.])

        sched = Schedule()
        sched += Play(pulse0, d0)
        sched += Play(pulse0, d1)
        sched += Play(pulse1, d0)
        sched += Play(pulse1, d1)
        sched = self.truncate_pulses(sched)

        ref_sched = Schedule()
        ref_sched += Play(trunc_pulse0, d0)
        ref_sched += Play(trunc_pulse0, d1)
        ref_sched += Play(trunc_pulse1, d0)
        ref_sched += Play(trunc_pulse1, d1)

        self.assertEqual(sched, ref_sched)


class TestTruncatePhasePrecision(QiskitTestCase):
    """Test truncation of waveform precision."""
    def truncate_phases(self, schedule):
        pm = PassManager()
        pm.append(transforms.TruncatePhasePrecision())
        return pm.run(pulse.Program(schedules=[schedule])).schedules[0]

    def test_phase_truncation(self):
        d0 = DriveChannel(0)

        shift_instr = instructions.ShiftPhase(0.00000055, d0)
        shift_twopi_instr = instructions.ShiftPhase(2*np.pi, d0)

        trunc_shift_instr = instructions.ShiftPhase(0.0000006, d0)
        trunc_shift_twopi_instr = instructions.ShiftPhase(0., d0)

        set_instr = instructions.SetPhase(0.00000055, d0)
        set_twopi_instr = instructions.SetPhase(2*np.pi, d0)

        trunc_set_instr = instructions.SetPhase(0.0000006, d0)
        trunc_set_twopi_instr = instructions.SetPhase(0., d0)

        sched = Schedule()
        sched += shift_instr
        sched += shift_instr
        sched += set_instr
        sched += set_instr
        sched += shift_twopi_instr
        sched += set_twopi_instr

        sched = self.truncate_phases(sched)

        ref_sched = Schedule()
        ref_sched += trunc_shift_instr
        ref_sched += trunc_shift_instr
        ref_sched += trunc_set_instr
        ref_sched += trunc_set_instr
        ref_sched += trunc_shift_twopi_instr
        ref_sched += trunc_set_twopi_instr

        self.assertEqual(sched, ref_sched)


if __name__ == '__main__':
    unittest.main()
