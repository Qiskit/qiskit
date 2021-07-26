# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for parameters used in Schedules."""
import unittest
import cmath
from copy import deepcopy

import numpy as np

from qiskit import pulse, assemble
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import HAS_SYMENGINE
from qiskit.pulse import PulseError
from qiskit.pulse.channels import DriveChannel, AcquireChannel, MemorySlot
from qiskit.pulse.transforms import inline_subroutines
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAlmaden


class TestPulseParameters(QiskitTestCase):
    """Tests usage of Parameters in qiskit.pulse; specifically in Schedules,
    Instructions, and Pulses.
    """

    def setUp(self):
        """Just some useful, reusable Parameters and constants."""
        super().setUp()
        self.alpha = Parameter("⍺")
        self.beta = Parameter("beta")
        self.gamma = Parameter("γ")
        self.phi = Parameter("ϕ")
        self.theta = Parameter("ϑ")
        self.amp = Parameter("amp")
        self.sigma = Parameter("sigma")
        self.qubit = Parameter("q")
        self.dur = Parameter("dur")

        self.freq = 4.5e9
        self.shift = 0.2e9
        self.phase = 3.1415 / 4

        self.backend = FakeAlmaden()

    def test_parameter_attribute_channel(self):
        """Test the ``parameter`` attributes."""
        chan = DriveChannel(self.qubit * self.alpha)
        self.assertTrue(chan.is_parameterized())
        self.assertEqual(chan.parameters, {self.qubit, self.alpha})
        chan = chan.assign(self.qubit, self.alpha)
        self.assertEqual(chan.parameters, {self.alpha})
        chan = chan.assign(self.alpha, self.beta)
        self.assertEqual(chan.parameters, {self.beta})
        chan = chan.assign(self.beta, 1)
        self.assertFalse(chan.is_parameterized())

    def test_parameter_attribute_instruction(self):
        """Test the ``parameter`` attributes."""
        inst = pulse.ShiftFrequency(self.alpha * self.qubit, DriveChannel(self.qubit))
        self.assertTrue(inst.is_parameterized())
        self.assertEqual(inst.parameters, {self.alpha, self.qubit})
        inst.assign_parameters({self.alpha: self.qubit})
        self.assertEqual(inst.parameters, {self.qubit})
        inst.assign_parameters({self.qubit: 1})
        self.assertFalse(inst.is_parameterized())
        self.assertEqual(inst.parameters, set())

    def test_parameter_attribute_play(self):
        """Test the ``parameter`` attributes."""
        inst = pulse.Play(
            pulse.Gaussian(self.dur, self.amp, self.sigma), pulse.DriveChannel(self.qubit)
        )
        self.assertTrue(inst.is_parameterized())
        self.assertSetEqual(inst.parameters, {self.dur, self.amp, self.sigma, self.qubit})

        inst = pulse.Play(pulse.Gaussian(self.dur, 0.1, self.sigma), pulse.DriveChannel(self.qubit))
        self.assertTrue(inst.is_parameterized())
        self.assertSetEqual(inst.parameters, {self.dur, self.sigma, self.qubit})

    def test_parameter_attribute_schedule(self):
        """Test the ``parameter`` attributes."""
        schedule = pulse.Schedule()
        self.assertFalse(schedule.is_parameterized())
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))
        self.assertEqual(schedule.parameters, {self.alpha})
        schedule += pulse.ShiftFrequency(self.gamma, DriveChannel(0))
        self.assertEqual(schedule.parameters, {self.alpha, self.gamma})
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))
        self.assertTrue(schedule.is_parameterized())
        self.assertEqual(schedule.parameters, {self.alpha, self.gamma, self.phi})
        schedule.assign_parameters({self.phi: self.alpha, self.gamma: self.shift})
        self.assertEqual(schedule.parameters, {self.alpha})
        schedule.assign_parameters({self.alpha: self.beta})
        self.assertEqual(schedule.parameters, {self.beta})
        schedule.assign_parameters({self.beta: 10})
        self.assertFalse(schedule.is_parameterized())

    def test_straight_schedule_bind(self):
        """Nothing fancy, 1:1 mapping."""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))
        schedule += pulse.ShiftFrequency(self.gamma, DriveChannel(0))
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))
        schedule += pulse.ShiftPhase(self.theta, DriveChannel(1))

        schedule.assign_parameters(
            {
                self.alpha: self.freq,
                self.gamma: self.shift,
                self.phi: self.phase,
                self.theta: -self.phase,
            }
        )

        insts = assemble(schedule, self.backend).experiments[0].instructions
        GHz = 1e9  # pylint: disable=invalid-name
        self.assertEqual(float(insts[0].frequency * GHz), self.freq)
        self.assertEqual(float(insts[1].frequency * GHz), self.shift)
        self.assertEqual(float(insts[2].phase), self.phase)
        self.assertEqual(float(insts[3].phase), -self.phase)

    def test_multiple_parameters(self):
        """Expressions of parameters with partial assignment."""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha + self.beta, DriveChannel(0))
        schedule += pulse.ShiftFrequency(self.gamma + self.beta, DriveChannel(0))
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))

        # Partial bind
        delta = 1e9
        schedule.assign_parameters({self.alpha: self.freq - delta})
        schedule.assign_parameters({self.beta: delta})
        schedule.assign_parameters({self.gamma: self.shift - delta})
        schedule.assign_parameters({self.phi: self.phase})

        insts = schedule.instructions
        self.assertEqual(float(insts[0][1].frequency), self.freq)
        self.assertEqual(float(insts[1][1].frequency), self.shift)
        self.assertEqual(float(insts[2][1].phase), self.phase)

    def test_with_function(self):
        """Test ParameterExpressions formed trivially in a function."""

        def get_frequency(variable):
            return 2 * variable

        def get_shift(variable):
            return variable - 1

        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(get_frequency(self.alpha), DriveChannel(0))
        schedule += pulse.ShiftFrequency(get_shift(self.gamma), DriveChannel(0))

        schedule.assign_parameters({self.alpha: self.freq / 2, self.gamma: self.shift + 1})

        insts = schedule.instructions
        self.assertEqual(float(insts[0][1].frequency), self.freq)
        self.assertEqual(float(insts[1][1].frequency), self.shift)

    def test_substitution(self):
        """Test Parameter substitution (vs bind)."""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))

        schedule.assign_parameters({self.alpha: 2 * self.beta})
        self.assertEqual(schedule.instructions[0][1].frequency, 2 * self.beta)
        schedule.assign_parameters({self.beta: self.freq / 2})
        self.assertEqual(float(schedule.instructions[0][1].frequency), self.freq)

    def test_substitution_with_existing(self):
        """Test that substituting one parameter with an existing parameter works."""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(self.qubit))

        schedule.assign_parameters({self.alpha: 1e9 * self.qubit})
        self.assertEqual(schedule.instructions[0][1].frequency, 1e9 * self.qubit)
        schedule.assign_parameters({self.qubit: 2})
        self.assertEqual(float(schedule.instructions[0][1].frequency), 2e9)

    def test_channels(self):
        """Test that channel indices can also be parameterized and assigned."""
        schedule = pulse.Schedule()
        schedule += pulse.ShiftPhase(self.phase, DriveChannel(2 * self.qubit))

        schedule.assign_parameters({self.qubit: 4})
        self.assertEqual(schedule.instructions[0][1].channel, DriveChannel(8))

    def test_acquire_channels(self):
        """Test Acquire instruction with multiple channels parameterized."""
        schedule = pulse.Schedule()
        schedule += pulse.Acquire(16000, AcquireChannel(self.qubit), MemorySlot(self.qubit))
        schedule.assign_parameters({self.qubit: 1})
        self.assertEqual(schedule.instructions[0][1].channel, AcquireChannel(1))
        self.assertEqual(schedule.instructions[0][1].mem_slot, MemorySlot(1))

    def test_overlapping_pulses(self):
        """Test that an error is still raised when overlapping instructions are assigned."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(self.qubit))
        with self.assertRaises(PulseError):
            schedule |= pulse.Play(pulse.Waveform([0.5, 0.5, 0.5, 0.5]), DriveChannel(self.qubit))

    def test_overlapping_on_assignment(self):
        """Test that assignment will catch against existing instructions."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(1))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(self.qubit))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({self.qubit: 1})

    def test_overlapping_on_expression_assigment_to_zero(self):
        """Test constant*zero expression conflict."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(self.qubit))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(2 * self.qubit))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({self.qubit: 0})

    def test_merging_upon_assignment(self):
        """Test that schedule can match instructions on a channel."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(1))
        schedule = schedule.insert(
            4, pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(self.qubit))
        )
        schedule.assign_parameters({self.qubit: 1})
        self.assertEqual(schedule.ch_duration(DriveChannel(1)), 8)
        self.assertEqual(schedule.channels, (DriveChannel(1),))

    def test_overlapping_on_multiple_assignment(self):
        """Test that assigning one qubit then another raises error when overlapping."""
        qubit2 = Parameter("q2")
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(self.qubit))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), DriveChannel(qubit2))
        schedule.assign_parameters({qubit2: 2})
        with self.assertRaises(PulseError):
            schedule.assign_parameters({self.qubit: 2})

    def test_play_with_parametricpulse(self):
        """Test Parametric Pulses with parameters determined by ParameterExpressions
        in the Play instruction."""
        waveform = pulse.library.Gaussian(duration=128, sigma=self.sigma, amp=self.amp)

        schedule = pulse.Schedule()
        schedule += pulse.Play(waveform, DriveChannel(10))
        schedule.assign_parameters({self.amp: 0.2, self.sigma: 4})

        self.backend.configuration().parametric_pulses = ["gaussian", "drag"]
        insts = schedule.instructions
        self.assertEqual(insts[0][1].pulse.amp, 0.2)
        self.assertEqual(insts[0][1].pulse.sigma, 4.0)

    def test_parametric_pulses_parameter_assignment(self):
        """Test Parametric Pulses with parameters determined by ParameterExpressions."""
        waveform = pulse.library.GaussianSquare(
            duration=1280, sigma=self.sigma, amp=self.amp, width=1000
        )
        waveform = waveform.assign_parameters({self.amp: 0.3, self.sigma: 12})
        self.assertEqual(waveform.amp, 0.3)
        self.assertEqual(waveform.sigma, 12)

        waveform = pulse.library.Drag(duration=1280, sigma=self.sigma, amp=self.amp, beta=2)
        waveform = waveform.assign_parameters({self.sigma: 12.7})
        self.assertEqual(waveform.amp, self.amp)
        self.assertEqual(waveform.sigma, 12.7)

    @unittest.skip("Not yet supported by ParameterExpression")
    def test_complex_value_assignment(self):
        """Test that complex values can be assigned to Parameters."""
        waveform = pulse.library.Constant(duration=1280, amp=self.amp)
        waveform.assign_parameters({self.amp: 0.2j})
        self.assertEqual(waveform.amp, 0.2j)

    def test_invalid_parametric_pulses(self):
        """Test that invalid parameters are still checked upon assignment."""
        schedule = pulse.Schedule()
        waveform = pulse.library.Constant(duration=1280, amp=2 * self.amp)
        schedule += pulse.Play(waveform, DriveChannel(0))
        with self.assertRaises(PulseError):
            waveform.assign_parameters({self.amp: 0.6})

    def test_get_parameter(self):
        """Test that get parameter by name."""
        param1 = Parameter("amp")
        param2 = Parameter("amp")

        schedule = pulse.Schedule()
        waveform1 = pulse.library.Constant(duration=1280, amp=param1)
        waveform2 = pulse.library.Constant(duration=1280, amp=param2)
        schedule += pulse.Play(waveform1, DriveChannel(0))
        schedule += pulse.Play(waveform2, DriveChannel(1))

        self.assertEqual(len(schedule.get_parameters("amp")), 2)

    def test_reference_to_subroutine_params(self):
        """Test that get parameter objects from subroutines."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        program_layer0 = pulse.Schedule()
        program_layer0 += pulse.Play(waveform, DriveChannel(0))

        # from call instruction
        program_layer1 = pulse.Schedule()
        program_layer1 += pulse.instructions.Call(program_layer0)
        self.assertEqual(program_layer1.get_parameters("amp")[0], param1)

        # from nested call instruction
        program_layer2 = pulse.Schedule()
        program_layer2 += pulse.instructions.Call(program_layer1)
        self.assertEqual(program_layer2.get_parameters("amp")[0], param1)

    def test_assign_parameter_to_subroutine(self):
        """Test that assign parameter objects to subroutines."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        program_layer0 = pulse.Schedule()
        program_layer0 += pulse.Play(waveform, DriveChannel(0))
        reference = deepcopy(program_layer0).assign_parameters({param1: 0.1})

        # to call instruction
        program_layer1 = pulse.Schedule()
        program_layer1 += pulse.instructions.Call(program_layer0)
        target = deepcopy(program_layer1).assign_parameters({param1: 0.1})
        self.assertEqual(inline_subroutines(target), reference)

        # to nested call instruction
        program_layer2 = pulse.Schedule()
        program_layer2 += pulse.instructions.Call(program_layer1)
        target = deepcopy(program_layer2).assign_parameters({param1: 0.1})
        self.assertEqual(inline_subroutines(target), reference)

    def test_assign_parameter_to_subroutine_parameter(self):
        """Test that assign parameter objects to parameter of subroutine."""
        param1 = Parameter("amp")
        waveform = pulse.library.Constant(duration=100, amp=param1)

        param_sub1 = Parameter("amp")
        param_sub2 = Parameter("phase")

        subroutine = pulse.Schedule()
        subroutine += pulse.Play(waveform, DriveChannel(0))
        reference = deepcopy(subroutine).assign_parameters({param1: 0.1 * np.exp(0.5j)})

        main_prog = pulse.Schedule()
        pdict = {param1: param_sub1 * np.exp(1j * param_sub2)}
        main_prog += pulse.instructions.Call(subroutine, value_dict=pdict)

        # parameter is overwritten by parameters
        self.assertEqual(len(main_prog.parameters), 2)
        target = deepcopy(main_prog).assign_parameters({param_sub1: 0.1, param_sub2: 0.5})
        result = inline_subroutines(target)
        if not HAS_SYMENGINE:
            self.assertEqual(result, reference)
        else:
            # Because of simplification differences between sympy and symengine when
            # symengine is used we get 0.1*exp(0.5*I) instead of the evaluated
            # 0.0877582562 + 0.0479425539*I resulting in a failure. When
            # symengine is installed manually build the amplitude as a complex to
            # avoid this.
            reference = pulse.Schedule()
            waveform = pulse.library.Constant(duration=100, amp=0.1 * cmath.exp(0.5j))
            reference += pulse.Play(waveform, DriveChannel(0))
            self.assertEqual(result, reference)


class TestParameterDuration(QiskitTestCase):
    """Tests parametrization of instruction duration."""

    def test_pulse_duration(self):
        """Test parametrization of pulse duration."""
        dur = Parameter("dur")

        test_pulse = pulse.Gaussian(dur, 0.1, dur / 4)
        ref_pulse = pulse.Gaussian(160, 0.1, 40)

        self.assertEqual(test_pulse.assign_parameters({dur: 160}), ref_pulse)

    def test_play_duration(self):
        """Test parametrization of play instruction duration."""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_play = pulse.Play(pulse.Gaussian(dur, 0.1, dur / 4), ch)
        test_play.assign_parameters({dur: 160})

        self.assertEqual(test_play.duration, 160)

    def test_delay_duration(self):
        """Test parametrization of delay duration."""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_delay = pulse.Delay(dur, ch)
        test_delay.assign_parameters({dur: 300})

        self.assertEqual(test_delay.duration, 300)

    def test_acquire_duration(self):
        """Test parametrization of acquire duration."""
        dur = Parameter("dur")
        ch = pulse.AcquireChannel(0)
        mem_slot = pulse.MemorySlot(0)

        test_acquire = pulse.Acquire(dur, ch, mem_slot=mem_slot)
        test_acquire.assign_parameters({dur: 300})

        self.assertEqual(test_acquire.duration, 300)

    def test_is_parameterized(self):
        """Test is parameterized method for parameter duration."""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_play = pulse.Play(pulse.Gaussian(dur, 0.1, dur / 4), ch)

        self.assertEqual(test_play.is_parameterized(), True)

    def test_cannot_build_schedule(self):
        """Test we cannot build schedule with parameterized instructions"""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_play = pulse.Play(pulse.Gaussian(dur, 0.1, dur / 4), ch)

        sched = pulse.Schedule()
        with self.assertRaises(pulse.exceptions.UnassignedDurationError):
            sched.insert(0, test_play)
