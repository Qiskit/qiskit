# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test cases for parameter manager."""

from copy import deepcopy

import numpy as np

from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.pulse.parameter_manager import ParameterGetter, ParameterSetter
from qiskit.pulse.transforms import AlignEquispaced, AlignLeft, inline_subroutines
from qiskit.test import QiskitTestCase


class ParameterTestBase(QiskitTestCase):
    """A base class for parameter manager unittest, providing test schedule."""

    def setUp(self):
        """Just some useful, reusable Parameters, constants, schedules."""
        super().setUp()

        self.amp1_1 = Parameter("amp1_1")
        self.amp1_2 = Parameter("amp1_2")
        self.amp2 = Parameter("amp2")
        self.amp3 = Parameter("amp3")

        self.dur1 = Parameter("dur1")
        self.dur2 = Parameter("dur2")
        self.dur3 = Parameter("dur3")

        self.parametric_waveform1 = pulse.Gaussian(
            duration=self.dur1, amp=self.amp1_1 + self.amp1_2, sigma=self.dur1 / 4
        )

        self.parametric_waveform2 = pulse.Gaussian(
            duration=self.dur2, amp=self.amp2, sigma=self.dur2 / 5
        )

        self.parametric_waveform3 = pulse.Gaussian(
            duration=self.dur3, amp=self.amp3, sigma=self.dur3 / 6
        )

        self.ch1 = Parameter("ch1")
        self.ch2 = Parameter("ch2")
        self.ch3 = Parameter("ch3")

        self.d1 = pulse.DriveChannel(self.ch1)
        self.d2 = pulse.DriveChannel(self.ch2)
        self.d3 = pulse.DriveChannel(self.ch3)

        self.phi1 = Parameter("phi1")
        self.phi2 = Parameter("phi2")
        self.phi3 = Parameter("phi3")

        self.meas_dur = Parameter("meas_dur")
        self.mem1 = Parameter("s1")
        self.reg1 = Parameter("m1")

        self.context_dur = Parameter("context_dur")

        # schedule under test
        subroutine = pulse.ScheduleBlock(alignment_context=AlignLeft())
        subroutine += pulse.ShiftPhase(self.phi1, self.d1)
        subroutine += pulse.Play(self.parametric_waveform1, self.d1)

        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(self.phi3, self.d3)

        long_schedule = pulse.ScheduleBlock(
            alignment_context=AlignEquispaced(self.context_dur), name="long_schedule"
        )

        long_schedule += subroutine
        long_schedule += pulse.ShiftPhase(self.phi2, self.d2)
        long_schedule += pulse.Play(self.parametric_waveform2, self.d2)
        with self.assertWarns(DeprecationWarning):
            long_schedule += pulse.Call(sched)
        long_schedule += pulse.Play(self.parametric_waveform3, self.d3)

        long_schedule += pulse.Acquire(
            self.meas_dur,
            pulse.AcquireChannel(self.ch1),
            mem_slot=pulse.MemorySlot(self.mem1),
            reg_slot=pulse.RegisterSlot(self.reg1),
        )

        self.test_sched = long_schedule


class TestParameterGetter(ParameterTestBase):
    """Test getting parameters."""

    def test_get_parameter_from_channel(self):
        """Test get parameters from channel."""
        test_obj = pulse.DriveChannel(self.ch1 + self.ch2)

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.ch1, self.ch2}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_pulse(self):
        """Test get parameters from pulse instruction."""
        test_obj = self.parametric_waveform1

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.amp1_1, self.amp1_2, self.dur1}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_acquire(self):
        """Test get parameters from acquire instruction."""
        test_obj = pulse.Acquire(16000, pulse.AcquireChannel(self.ch1), pulse.MemorySlot(self.ch1))

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.ch1}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_inst(self):
        """Test get parameters from instruction."""
        test_obj = pulse.ShiftPhase(self.phi1 + self.phi2, pulse.DriveChannel(0))

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.phi1, self.phi2}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_call(self):
        """Test get parameters from instruction."""
        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(self.phi1, self.d1)

        with self.assertWarns(DeprecationWarning):
            test_obj = pulse.Call(subroutine=sched)

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.phi1, self.ch1}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_with_function(self):
        """Test ParameterExpressions formed trivially in a function."""

        def get_shift(variable):
            return variable - 1

        test_obj = pulse.ShiftPhase(get_shift(self.phi1), self.d1)

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.phi1, self.ch1}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_alignment_context(self):
        """Test get parameters from alignment context."""
        test_obj = AlignEquispaced(duration=self.context_dur + self.dur1)

        visitor = ParameterGetter()
        visitor.visit(test_obj)

        ref_params = {self.context_dur, self.dur1}

        self.assertSetEqual(visitor.parameters, ref_params)

    def test_get_parameter_from_complex_schedule(self):
        """Test get parameters from complicated schedule."""
        test_block = deepcopy(self.test_sched)

        visitor = ParameterGetter()
        visitor.visit(test_block)

        self.assertEqual(len(visitor.parameters), 17)


class TestParameterSetter(ParameterTestBase):
    """Test setting parameters."""

    def test_set_parameter_to_channel(self):
        """Test set parameters from channel."""
        test_obj = pulse.DriveChannel(self.ch1 + self.ch2)

        value_dict = {self.ch1: 1, self.ch2: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.DriveChannel(3)

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_pulse(self):
        """Test set parameters from pulse instruction."""
        test_obj = self.parametric_waveform1

        value_dict = {self.amp1_1: 0.1, self.amp1_2: 0.2, self.dur1: 160}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.Gaussian(duration=160, amp=0.3, sigma=40)

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_acquire(self):
        """Test set parameters to acquire instruction."""
        test_obj = pulse.Acquire(16000, pulse.AcquireChannel(self.ch1), pulse.MemorySlot(self.ch1))

        value_dict = {self.ch1: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.Acquire(16000, pulse.AcquireChannel(2), pulse.MemorySlot(2))

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_inst(self):
        """Test get parameters from instruction."""
        test_obj = pulse.ShiftPhase(self.phi1 + self.phi2, pulse.DriveChannel(0))

        value_dict = {self.phi1: 0.123, self.phi2: 0.456}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.ShiftPhase(0.579, pulse.DriveChannel(0))

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_call(self):
        """Test get parameters from instruction."""
        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(self.phi1, self.d1)

        with self.assertWarns(DeprecationWarning):
            test_obj = pulse.Call(subroutine=sched)

        value_dict = {self.phi1: 1.57, self.ch1: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_sched = pulse.Schedule()
        ref_sched += pulse.ShiftPhase(1.57, pulse.DriveChannel(2))

        with self.assertWarns(DeprecationWarning):
            ref_obj = pulse.Call(subroutine=ref_sched)

        self.assertEqual(assigned, ref_obj)

    def test_with_function(self):
        """Test ParameterExpressions formed trivially in a function."""

        def get_shift(variable):
            return variable - 1

        test_obj = pulse.ShiftPhase(get_shift(self.phi1), self.d1)

        value_dict = {self.phi1: 2.0, self.ch1: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.ShiftPhase(1.0, pulse.DriveChannel(2))

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_alignment_context(self):
        """Test get parameters from alignment context."""
        test_obj = AlignEquispaced(duration=self.context_dur + self.dur1)

        value_dict = {self.context_dur: 1000, self.dur1: 100}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = AlignEquispaced(duration=1100)

        self.assertEqual(assigned, ref_obj)

    def test_nested_assignment_partial_bind(self):
        """Test nested schedule with call instruction.
        Inline the schedule and partially bind parameters."""
        context = AlignEquispaced(duration=self.context_dur)
        subroutine = pulse.ScheduleBlock(alignment_context=context)
        subroutine += pulse.Play(self.parametric_waveform1, self.d1)

        nested_block = pulse.ScheduleBlock()
        with self.assertWarns(DeprecationWarning):
            nested_block += pulse.Call(subroutine=subroutine)

        test_obj = pulse.ScheduleBlock()
        test_obj += nested_block

        test_obj = inline_subroutines(test_obj)

        value_dict = {self.context_dur: 1000, self.dur1: 200, self.ch1: 1}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_context = AlignEquispaced(duration=1000)
        ref_subroutine = pulse.ScheduleBlock(alignment_context=ref_context)
        ref_subroutine += pulse.Play(
            pulse.Gaussian(200, self.amp1_1 + self.amp1_2, 50), pulse.DriveChannel(1)
        )

        ref_nested_block = pulse.ScheduleBlock()
        ref_nested_block += ref_subroutine

        ref_obj = pulse.ScheduleBlock()
        ref_obj += ref_nested_block

        self.assertEqual(assigned, ref_obj)

    def test_complex_valued_parameter(self):
        """Test complex valued parameter can be casted to a complex value,
        but raises PendingDeprecationWarning.."""
        amp = Parameter("amp")
        test_obj = pulse.Constant(duration=160, amp=1j * amp)

        value_dict = {amp: 0.1}

        visitor = ParameterSetter(param_map=value_dict)
        with self.assertWarns(PendingDeprecationWarning):
            assigned = visitor.visit(test_obj)

        with self.assertWarns(DeprecationWarning):
            ref_obj = pulse.Constant(duration=160, amp=1j * 0.1)

        self.assertEqual(assigned, ref_obj)

    def test_complex_value_to_parameter(self):
        """Test complex value can be assigned to parameter object,
        but raises PendingDeprecationWarning."""
        amp = Parameter("amp")
        test_obj = pulse.Constant(duration=160, amp=amp)

        value_dict = {amp: 0.1j}

        visitor = ParameterSetter(param_map=value_dict)
        with self.assertWarns(PendingDeprecationWarning):
            assigned = visitor.visit(test_obj)

        with self.assertWarns(DeprecationWarning):
            ref_obj = pulse.Constant(duration=160, amp=1j * 0.1)

        self.assertEqual(assigned, ref_obj)

    def test_complex_parameter_expression(self):
        """Test assignment of complex-valued parameter expression to parameter,
        but raises PendingDeprecationWarning."""
        amp = Parameter("amp")

        mag = Parameter("A")
        phi = Parameter("phi")

        test_obj = pulse.Constant(duration=160, amp=amp)
        test_obj_copy = deepcopy(test_obj)
        # generate parameter expression
        value_dict = {amp: mag * np.exp(1j * phi)}
        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        # generate complex value
        value_dict = {mag: 0.1, phi: 0.5}
        visitor = ParameterSetter(param_map=value_dict)
        with self.assertWarns(PendingDeprecationWarning):
            assigned = visitor.visit(assigned)

        # evaluated parameter expression: 0.0877582561890373 + 0.0479425538604203*I
        value_dict = {amp: 0.1 * np.exp(0.5j)}

        visitor = ParameterSetter(param_map=value_dict)
        with self.assertWarns(PendingDeprecationWarning):
            ref_obj = visitor.visit(test_obj_copy)
        self.assertEqual(assigned, ref_obj)

    def test_invalid_pulse_amplitude(self):
        """Test that invalid parameters are still checked upon assignment."""
        amp = Parameter("amp")

        test_sched = pulse.ScheduleBlock()
        test_sched.append(
            pulse.Play(
                pulse.Constant(160, amp=2 * amp),
                pulse.DriveChannel(0),
            ),
            inplace=True,
        )
        with self.assertRaises(PulseError):
            test_sched.assign_parameters({amp: 0.6}, inplace=False)

    def test_set_parameter_to_complex_schedule(self):
        """Test get parameters from complicated schedule."""
        test_block = deepcopy(self.test_sched)

        value_dict = {
            self.amp1_1: 0.1,
            self.amp1_2: 0.2,
            self.amp2: 0.3,
            self.amp3: 0.4,
            self.dur1: 100,
            self.dur2: 125,
            self.dur3: 150,
            self.ch1: 0,
            self.ch2: 2,
            self.ch3: 4,
            self.phi1: 1.0,
            self.phi2: 2.0,
            self.phi3: 3.0,
            self.meas_dur: 300,
            self.mem1: 3,
            self.reg1: 0,
            self.context_dur: 1000,
        }

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_block)

        # create ref schedule
        subroutine = pulse.ScheduleBlock(alignment_context=AlignLeft())
        subroutine += pulse.ShiftPhase(1.0, pulse.DriveChannel(0))
        subroutine += pulse.Play(pulse.Gaussian(100, 0.3, 25), pulse.DriveChannel(0))

        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(3.0, pulse.DriveChannel(4))

        ref_obj = pulse.ScheduleBlock(alignment_context=AlignEquispaced(1000), name="long_schedule")

        ref_obj += subroutine
        ref_obj += pulse.ShiftPhase(2.0, pulse.DriveChannel(2))
        ref_obj += pulse.Play(pulse.Gaussian(125, 0.3, 25), pulse.DriveChannel(2))
        with self.assertWarns(DeprecationWarning):
            ref_obj += pulse.Call(sched)
        ref_obj += pulse.Play(pulse.Gaussian(150, 0.4, 25), pulse.DriveChannel(4))

        ref_obj += pulse.Acquire(
            300, pulse.AcquireChannel(0), pulse.MemorySlot(3), pulse.RegisterSlot(0)
        )

        self.assertEqual(assigned, ref_obj)


class TestAssignFromProgram(QiskitTestCase):
    """Test managing parameters from programs. Parameter manager is implicitly called."""

    def test_attribute_parameters(self):
        """Test the ``parameter`` attributes."""
        sigma = Parameter("sigma")
        amp = Parameter("amp")

        waveform = pulse.library.Gaussian(duration=128, sigma=sigma, amp=amp)

        block = pulse.ScheduleBlock()
        block += pulse.Play(waveform, pulse.DriveChannel(10))

        ref_set = {amp, sigma}

        self.assertSetEqual(set(block.parameters), ref_set)

    def test_parametric_pulses(self):
        """Test Parametric Pulses with parameters determined by ParameterExpressions
        in the Play instruction."""
        sigma = Parameter("sigma")
        amp = Parameter("amp")

        waveform = pulse.library.Gaussian(duration=128, sigma=sigma, amp=amp)

        block = pulse.ScheduleBlock()
        block += pulse.Play(waveform, pulse.DriveChannel(10))
        block.assign_parameters({amp: 0.2, sigma: 4}, inplace=True)

        self.assertEqual(block.blocks[0].pulse.amp, 0.2)
        self.assertEqual(block.blocks[0].pulse.sigma, 4.0)

    def test_parameters_from_subroutine(self):
        """Test that get parameter objects from subroutines."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        program_layer0 = pulse.Schedule()
        program_layer0 += pulse.Play(waveform, pulse.DriveChannel(0))

        # from call instruction
        program_layer1 = pulse.Schedule()
        with self.assertWarns(DeprecationWarning):
            program_layer1 += pulse.instructions.Call(program_layer0)
        self.assertEqual(program_layer1.get_parameters("amp")[0], param1)

        # from nested call instruction
        program_layer2 = pulse.Schedule()
        with self.assertWarns(DeprecationWarning):
            program_layer2 += pulse.instructions.Call(program_layer1)
        self.assertEqual(program_layer2.get_parameters("amp")[0], param1)

    def test_assign_parameter_to_subroutine(self):
        """Test that assign parameter objects to subroutines."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        program_layer0 = pulse.Schedule()
        program_layer0 += pulse.Play(waveform, pulse.DriveChannel(0))
        reference = program_layer0.assign_parameters({param1: 0.1}, inplace=False)

        # to call instruction
        program_layer1 = pulse.Schedule()
        with self.assertWarns(DeprecationWarning):
            program_layer1 += pulse.instructions.Call(program_layer0)
        target = program_layer1.assign_parameters({param1: 0.1}, inplace=False)
        self.assertEqual(inline_subroutines(target), reference)

        # to nested call instruction
        program_layer2 = pulse.Schedule()
        with self.assertWarns(DeprecationWarning):
            program_layer2 += pulse.instructions.Call(program_layer1)
        target = program_layer2.assign_parameters({param1: 0.1}, inplace=False)
        self.assertEqual(inline_subroutines(target), reference)

    def test_assign_parameter_to_subroutine_parameter(self):
        """Test that assign parameter objects to parameter of subroutine."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        param_sub1 = Parameter("p1")
        param_sub2 = Parameter("p2")

        subroutine = pulse.Schedule()
        subroutine += pulse.Play(waveform, pulse.DriveChannel(0))
        reference = subroutine.assign_parameters({param1: 0.6}, inplace=False)

        main_prog = pulse.Schedule()
        pdict = {param1: param_sub1 + param_sub2}
        with self.assertWarns(DeprecationWarning):
            main_prog += pulse.instructions.Call(subroutine, value_dict=pdict)

        # parameter is overwritten by parameters
        self.assertEqual(len(main_prog.parameters), 2)
        target = main_prog.assign_parameters({param_sub1: 0.1, param_sub2: 0.5}, inplace=False)
        result = inline_subroutines(target)

        self.assertEqual(result, reference)


class TestScheduleTimeslots(QiskitTestCase):
    """Test for edge cases of timing overlap on parametrized channels.

    Note that this test is dedicated to `Schedule` since `ScheduleBlock` implicitly
    assigns instruction time t0 that doesn't overlap with existing instructions.
    """

    def test_overlapping_pulses(self):
        """Test that an error is still raised when overlapping instructions are assigned."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        with self.assertRaises(PulseError):
            schedule |= pulse.Play(
                pulse.Waveform([0.5, 0.5, 0.5, 0.5]), pulse.DriveChannel(param_idx)
            )

    def test_overlapping_on_assignment(self):
        """Test that assignment will catch against existing instructions."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(1))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx: 1})

    def test_overlapping_on_expression_assigment_to_zero(self):
        """Test constant*zero expression conflict."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(2 * param_idx))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx: 0})

    def test_merging_upon_assignment(self):
        """Test that schedule can match instructions on a channel."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(1))
        schedule = schedule.insert(
            4, pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        )
        schedule.assign_parameters({param_idx: 1})

        self.assertEqual(schedule.ch_duration(pulse.DriveChannel(1)), 8)
        self.assertEqual(schedule.channels, (pulse.DriveChannel(1),))

    def test_overlapping_on_multiple_assignment(self):
        """Test that assigning one qubit then another raises error when overlapping."""
        param_idx1 = Parameter("q1")
        param_idx2 = Parameter("q2")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx1))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx2))
        schedule.assign_parameters({param_idx1: 2})

        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx2: 2})

    def test_cannot_build_schedule_with_unassigned_duration(self):
        """Test we cannot build schedule with parameterized instructions"""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_play = pulse.Play(pulse.Gaussian(dur, 0.1, dur / 4), ch)

        sched = pulse.Schedule()
        with self.assertRaises(UnassignedDurationError):
            sched.insert(0, test_play)
