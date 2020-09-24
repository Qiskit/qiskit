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
from qiskit.test import QiskitTestCase

from qiskit import pulse, assemble
from qiskit.circuit import Parameter
from qiskit.pulse import PulseError
from qiskit.pulse.channels import DriveChannel, AcquireChannel, MemorySlot
from qiskit.test.mock import FakeAlmaden


class TestPulseParameters(QiskitTestCase):
    """Tests usage of Parameters in qiskit.pulse; specifically in Schedules,
    Instructions, and Pulses.
    """

    def setUp(self):
        """Just some useful, reusable Parameters and constants."""
        super().setUp()
        self.alpha = Parameter('⍺')
        self.beta = Parameter('beta')
        self.gamma = Parameter('γ')
        self.phi = Parameter('ϕ')
        self.theta = Parameter('ϑ')
        self.amp = Parameter('amp')
        self.sigma = Parameter('sigma')
        self.qubit = Parameter('q')

        self.freq = 4.5e9
        self.shift = 0.2e9
        self.phase = 3.1415 / 4

        self.backend = FakeAlmaden()

    def test_straight_schedule_bind(self):
        """Nothing fancy, 1:1 mapping."""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))
        schedule += pulse.ShiftFrequency(self.gamma, DriveChannel(0))
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))
        schedule += pulse.ShiftPhase(self.theta, DriveChannel(1))

        schedule.assign_parameters({self.alpha: self.freq, self.gamma: self.shift,
                                    self.phi: self.phase, self.theta: -self.phase})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        GHz = 1e9  # pylint: disable=invalid-name
        self.assertEqual(float(insts[0].frequency*GHz), self.freq)
        self.assertEqual(float(insts[1].frequency*GHz), self.shift)
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
            return 2*variable

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

        schedule.assign_parameters({self.alpha: 2*self.beta})
        self.assertEqual(schedule.instructions[0][1].frequency, 2*self.beta)
        schedule.assign_parameters({self.beta: self.freq / 2})
        self.assertEqual(float(schedule.instructions[0][1].frequency), self.freq)

    def test_channels(self):
        """Test that channel indices can also be parameterized and assigned."""
        schedule = pulse.Schedule()
        schedule += pulse.ShiftPhase(self.phase, DriveChannel(2*self.qubit))

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
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(self.qubit))
        with self.assertRaises(PulseError):
            schedule |= pulse.Play(pulse.SamplePulse([0.5, 0.5, 0.5, 0.5]),
                                   DriveChannel(self.qubit))

    def test_overlapping_on_assignment(self):
        """Test that assignment will catch against existing instructions."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(1))
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(self.qubit))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({self.qubit: 1})

    def test_overlapping_on_expression_assigment_to_zero(self):
        """Test constant*zero expression conflict."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(self.qubit))
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(2*self.qubit))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({self.qubit: 0})

    def test_merging_upon_assignment(self):
        """Test that schedule can match instructions on a channel."""
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(1))
        schedule = schedule.insert(4, pulse.Play(pulse.SamplePulse([1, 1, 1, 1]),
                                                 DriveChannel(self.qubit)))
        schedule.assign_parameters({self.qubit: 1})
        self.assertEqual(schedule.ch_duration(DriveChannel(1)), 8)
        self.assertEqual(schedule.channels, (DriveChannel(1),))

    def test_overlapping_on_multiple_assignment(self):
        """Test that assigning one qubit then another raises error when overlapping."""
        qubit2 = Parameter('q2')
        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(self.qubit))
        schedule |= pulse.Play(pulse.SamplePulse([1, 1, 1, 1]), DriveChannel(qubit2))
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

        self.backend.configuration().parametric_pulses = ['gaussian', 'drag']
        insts = schedule.instructions
        self.assertEqual(insts[0][1].pulse.amp, 0.2)
        self.assertEqual(insts[0][1].pulse.sigma, 4.)

    def test_parametric_pulses_parameter_assignment(self):
        """Test Parametric Pulses with parameters determined by ParameterExpressions."""
        waveform = pulse.library.GaussianSquare(duration=1280, sigma=self.sigma,
                                                amp=self.amp, width=1000)
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
        waveform = pulse.library.Constant(duration=1280, amp=2*self.amp)
        schedule += pulse.Play(waveform, DriveChannel(0))
        with self.assertRaises(PulseError):
            waveform.assign_parameters({self.amp: 0.6})
