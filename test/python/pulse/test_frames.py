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

"""Test cases for frames in Qiskit pulse."""

import numpy as np

from qiskit.circuit import Parameter
from qiskit.compiler.assembler import frames_configuration
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit.pulse.transforms import resolve_frames
from qiskit.pulse.transforms.resolved_frame import ResolvedFrame
from qiskit.pulse.parameter_manager import ParameterManager


class TestFrame(QiskitTestCase):
    """Basic frame tests."""

    def basic(self):
        """Test basic functionality of frames."""

        frame = pulse.Frame(123)
        self.assertEqual(frame.index, 123)
        self.assertEqual(frame.name, 'f123')

    def test_parameter(self):
        """Test parameter assignment."""
        parameter_manager = ParameterManager()
        param = Parameter('a')
        frame = pulse.Frame(param)
        self.assertTrue(isinstance(frame.index, Parameter))

        parameter_manager.update_parameter_table(frame)
        new_frame = parameter_manager.assign_parameters(frame, {param: 123})

        self.assertEqual(new_frame, pulse.Frame(123))
        self.assertEqual(frame, pulse.Frame(param))


class TestResolvedFrames(QiskitTestCase):
    """Test that resolved frames behave properly."""

    def setUp(self):
        super().setUp()
        self.dt_ = 2.2222222222222221e-10
        self.freq0 = 0.093e9  # Frequency of frame 0
        self.freq1 = 0.125e9  # Frequency of frame 1

    def test_phase_advance(self):
        """Test that phases are properly set when frames are resolved."""

        d0 = pulse.DriveChannel(0)
        sig0 = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame(0))
        sig1 = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame(1))

        with pulse.build() as sched:
            pulse.play(sig0, d0)
            pulse.play(sig1, d0)
            pulse.play(sig0, d0)
            pulse.play(sig1, d0)

        frames_config = frames_configuration([[d0], [d0]], [self.freq0, self.freq1], self.dt_)

        frame0_ = ResolvedFrame(pulse.Frame(1), self.freq0, 0.0, self.dt_, [])
        frame1_ = ResolvedFrame(pulse.Frame(2), self.freq1, 0.0, self.dt_, [])

        # Check that the resolved frames are tracking phases properly
        for time in [0, 160, 320, 480]:
            phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * time * self.dt_)) % (2 * np.pi)
            self.assertAlmostEqual(frame0_.phase(time), phase, places=8)

            phase = np.angle(np.exp(2.0j * np.pi * self.freq1 * time * self.dt_)) % (2 * np.pi)
            self.assertAlmostEqual(frame1_.phase(time), phase, places=8)

        # Check that the proper phase instructions are added to the frame resolved schedules.
        resolved = resolve_frames(sched, frames_config).instructions

        params = [(0, 0, self.freq0, 1), (160, 160, self.freq1 - self.freq0, 4),
                  (320, 160, self.freq0 - self.freq1, 7), (480, 160, self.freq1 - self.freq0, 10)]

        for time, delta, frame_frequency, index in params:
            phase = np.angle(np.exp(2.0j * np.pi * frame_frequency * delta * self.dt_)) % (2 * np.pi)
            self.assertEqual(resolved[index][0], time)
            self.assertAlmostEqual(resolved[index][1].phase % (2*np.pi), phase, places=8)

    def test_phase_advance_with_instructions(self):
        """Test that the phase advances are properly computed with frame instructions."""

        sig = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame(0))

        with pulse.build() as sched:
            pulse.play(sig, pulse.DriveChannel(0))
            pulse.shift_phase(1.0, pulse.Frame(0))

        frame = ResolvedFrame(pulse.Frame(0), self.freq0, 0.0, self.dt_, [])
        frame.set_frame_instructions(sched)

        self.assertAlmostEqual(frame.phase(0), 0.0, places=8)

        # Test the phase right before the shift phase instruction
        phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * 159 * self.dt_)) % (2 * np.pi)
        self.assertAlmostEqual(frame.phase(159), phase, places=8)

        # Test the phase at and after the shift phase instruction
        phase = (np.angle(np.exp(2.0j * np.pi * self.freq0 * 160 * self.dt_)) + 1.0) % (2 * np.pi)
        self.assertAlmostEqual(frame.phase(160), phase, places=8)

        phase = (np.angle(np.exp(2.0j * np.pi * self.freq0 * 161 * self.dt_)) + 1.0) % (2 * np.pi)
        self.assertAlmostEqual(frame.phase(161), phase, places=8)

    def test_set_frequency(self):
        """Test setting the frequency of the resolved frame."""

        frame = ResolvedFrame(pulse.Frame(0), self.freq1, 0.0, self.dt_, [])
        frame.set_frequency(16, self.freq0)
        frame.set_frequency(10, self.freq1)
        frame.set_frequency(10, self.freq1)

        self.assertAlmostEqual(frame.frequency(0), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(10), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(15), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(16), self.freq0, places=8)

    def test_broadcasting(self):
        """Test that resolved frames broadcast to control channels."""

        sig = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame(0))

        with pulse.build() as sched:
            with pulse.align_left():
                pulse.play(sig, pulse.DriveChannel(3))
                pulse.shift_phase(1.23, pulse.Frame(0))
                with pulse.align_left(ignore_frames=True):
                    pulse.play(sig, pulse.DriveChannel(3))
                    pulse.play(sig, pulse.ControlChannel(0))

        frames_config = frames_configuration([[pulse.DriveChannel(3), pulse.ControlChannel(0)]],
                                             [self.freq0], self.dt_)

        resolved = resolve_frames(sched, frames_config).instructions

        phase = (np.angle(np.exp(2.0j * np.pi * self.freq0 * 160 * self.dt_)) + 1.23) % (2 * np.pi)

        # Check that the frame resolved schedule has the correct phase and frequency instructions
        # at the right place.
        # First, ensure that resolved starts with a SetFrequency and SetPhase.
        self.assertEquals(resolved[0][0], 0)
        set_freq = resolved[0][1]
        self.assertTrue(isinstance(set_freq, pulse.SetFrequency))
        self.assertAlmostEqual(set_freq.frequency, self.freq0, places=8)
        
        self.assertEquals(resolved[1][0], 0)
        set_phase = resolved[1][1]
        self.assertTrue(isinstance(set_phase, pulse.SetPhase))
        self.assertAlmostEqual(set_phase.phase, 0.0, places=8)

        # Next, check that we do phase shifts on the DriveChannel after the first Gaussian.
        self.assertEquals(resolved[3][0], 160)
        shift_phase = resolved[3][1]
        self.assertTrue(isinstance(shift_phase, pulse.ShiftPhase))
        self.assertAlmostEqual(shift_phase.phase, 1.23, places=8)

        # Up to now, no pulse has been applied on the ControlChannel so we should
        # encounter a Set instructions at time 160 which is when the first pulse
        # is played on ControlChannel(0)
        self.assertEquals(resolved[4][0], 160)
        set_freq = resolved[4][1]
        self.assertTrue(isinstance(set_freq, pulse.SetFrequency))
        self.assertAlmostEqual(set_freq.frequency, self.freq0, places=8)

        self.assertEquals(resolved[5][0], 160)
        set_phase = resolved[5][1]
        self.assertTrue(isinstance(set_phase, pulse.SetPhase))
        self.assertAlmostEqual(set_phase.phase, phase, places=8)
