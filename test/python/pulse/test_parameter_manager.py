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

"""Test cases for parameter manager."""

from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.pulse.parameter_manager import ParameterGetter, ParameterSetter
from qiskit.pulse.transforms import AlignEquispaced, AlignLeft, inline_subroutines
from qiskit.test import QiskitTestCase


class ParameterTestBase(QiskitTestCase):
    """A base class for parameter manager unittest, providing test schedule."""

    def setUp(self):
        """Just some useful, reusable Parameters, constants, schedules."""
        super().setUp()

        self.amp1_1 = Parameter('amp1_1')
        self.amp1_2 = Parameter('amp1_2')
        self.amp2 = Parameter('amp2')
        self.amp3 = Parameter('amp3')

        self.dur1 = Parameter('dur1')
        self.dur2 = Parameter('dur2')
        self.dur3 = Parameter('dur3')

        self.parametric_waveform1 = pulse.Gaussian(
            duration=self.dur1,
            amp=self.amp1_1 + self.amp1_2,
            sigma=self.dur1/4
        )

        self.parametric_waveform2 = pulse.Gaussian(
            duration=self.dur2,
            amp=self.amp2,
            sigma=self.dur2/5
        )

        self.parametric_waveform3 = pulse.Gaussian(
            duration=self.dur3,
            amp=self.amp3,
            sigma=self.dur3/6
        )

        self.ch1 = Parameter('ch1')
        self.ch2 = Parameter('ch2')
        self.ch3 = Parameter('ch3')

        self.d1 = pulse.DriveChannel(self.ch1)
        self.d2 = pulse.DriveChannel(self.ch2)
        self.d3 = pulse.DriveChannel(self.ch3)

        self.phi1 = Parameter('phi1')
        self.phi2 = Parameter('phi2')
        self.phi3 = Parameter('phi3')

        self.context_dur = Parameter('context_dur')

    def create_fully_parametric_schedule(self):
        """A helper function to generate a long schedule."""
        subroutine = pulse.ScheduleBlock(
            alignment_context=AlignLeft()
        )
        subroutine += pulse.ShiftPhase(self.phi1, self.d1)
        subroutine += pulse.Play(self.parametric_waveform1, self.d1)

        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(self.phi3, self.d3)

        long_schedule = pulse.ScheduleBlock(
            alignment_context=AlignEquispaced(self.context_dur),
            name='long_schedule'
        )

        long_schedule += subroutine
        long_schedule += pulse.ShiftPhase(self.phi2, self.d2)
        long_schedule += pulse.Play(self.parametric_waveform2, self.d2)
        long_schedule += pulse.Call(sched)
        long_schedule += pulse.Play(self.parametric_waveform3, self.d3)

        return long_schedule


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

        test_obj = pulse.Call(subroutine=sched)

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
        test_block = self.create_fully_parametric_schedule()

        visitor = ParameterGetter()
        visitor.visit(test_block)

        self.assertEqual(len(visitor.parameters), 14)


class TestParameterSetter(ParameterTestBase):
    """Test setting parameters."""

    def test_set_parameter_to_channel(self):
        """Test get parameters from channel."""
        test_obj = pulse.DriveChannel(self.ch1 + self.ch2)

        value_dict = {self.ch1: 1, self.ch2: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.DriveChannel(3)

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_pulse(self):
        """Test get parameters from pulse instruction."""
        test_obj = self.parametric_waveform1

        value_dict = {self.amp1_1: 0.1, self.amp1_2: 0.2, self.dur1: 160}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.Gaussian(duration=160, amp=0.3, sigma=40)

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

        test_obj = pulse.Schedule()
        test_obj += pulse.Call(subroutine=sched)

        value_dict = {self.phi1: 1.57, self.ch1: 2}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = pulse.Schedule()
        ref_obj += pulse.ShiftPhase(1.57, pulse.DriveChannel(2))

        self.assertEqual(inline_subroutines(assigned), ref_obj)

    def test_set_parameter_to_alignment_context(self):
        """Test get parameters from alignment context."""
        test_obj = AlignEquispaced(duration=self.context_dur + self.dur1)

        value_dict = {self.context_dur: 1000, self.dur1: 100}

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_obj)

        ref_obj = AlignEquispaced(duration=1100)

        self.assertEqual(assigned, ref_obj)

    def test_set_parameter_to_complex_schedule(self):
        """Test get parameters from complicated schedule."""
        test_block = self.create_fully_parametric_schedule()

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
            self.phi1: 1.,
            self.phi2: 2.,
            self.phi3: 3.,
            self.context_dur: 1000
        }

        visitor = ParameterSetter(param_map=value_dict)
        assigned = visitor.visit(test_block)

        # create ref schedule
        subroutine = pulse.ScheduleBlock(
            alignment_context=AlignLeft()
        )
        subroutine += pulse.ShiftPhase(1., pulse.DriveChannel(0))
        subroutine += pulse.Play(pulse.Gaussian(100, 0.3, 25), pulse.DriveChannel(0))

        sched = pulse.Schedule()
        sched += pulse.ShiftPhase(3., pulse.DriveChannel(4))

        ref_obj = pulse.ScheduleBlock(
            alignment_context=AlignEquispaced(1000),
            name='long_schedule'
        )

        ref_obj += subroutine
        ref_obj += pulse.ShiftPhase(2., pulse.DriveChannel(2))
        ref_obj += pulse.Play(pulse.Gaussian(125, 0.3, 25), pulse.DriveChannel(2))
        ref_obj += pulse.Call(sched)
        ref_obj += pulse.Play(pulse.Gaussian(150, 0.4, 25), pulse.DriveChannel(4))

        self.assertEqual(assigned, ref_obj)
