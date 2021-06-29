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

from qiskit import schedule, assemble
from qiskit.circuit import Parameter, QuantumCircuit, Gate
import qiskit.pulse as pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.pulse.transforms.frames import requires_frame_mapping
from qiskit.test import QiskitTestCase
from qiskit.pulse.transforms import map_frames, block_to_schedule
from qiskit.pulse.transforms.resolved_frame import ResolvedFrame
from qiskit.pulse.parameter_manager import ParameterManager
from qiskit.pulse.frame import Frame, FramesConfiguration, FrameDefinition
from qiskit.test.mock import FakeAthens
from qiskit.exceptions import QiskitError


class TestFrame(QiskitTestCase):
    """Basic frame tests."""

    def basic(self):
        """Test basic functionality of frames."""

        frame = pulse.Frame("Q", 13)
        self.assertEqual(frame.identifier, "Q13")
        self.assertEqual(frame.name, "Q13")

    def test_parameter(self):
        """Test parameter assignment."""
        parameter_manager = ParameterManager()
        param = Parameter("a")
        frame = pulse.Frame("Q", param)
        self.assertEqual(frame.identifier, ("Q", param))
        self.assertEqual(frame.prefix, "Q")

        parameter_manager.update_parameter_table(frame)
        new_frame = parameter_manager.assign_parameters(frame, {param: 123})

        self.assertEqual(new_frame, pulse.Frame("Q", 123))
        self.assertEqual(new_frame.identifier, ("Q", 123))
        self.assertEqual(frame, pulse.Frame("Q", param))


class TestFramesConfiguration(QiskitTestCase):
    """The the frames config object."""

    def test_frame_config(self):
        """Test that frame configs can be properly created."""

        config = {
            Frame("Q", 0): {"frequency": 5.5e9, "purpose": "Frame of qubit 0"},
            Frame("Q", 1): {
                "frequency": 5.2e9,
            },
            Frame("Q", 2): {
                "frequency": 5.2e9,
            },
        }

        frames_config = FramesConfiguration.from_dict(config)
        frames_config.sample_duration = 0.222e-9

        self.assertEqual(frames_config[Frame("Q", 0)].frequency, 5.5e9)
        self.assertEqual(frames_config.sample_duration, 0.222e-9)
        self.assertEqual(frames_config[Frame("Q", 1)].frequency, 5.2e9)

        for frame_def in frames_config.definitions:
            frame_def.sample_duration = 0.1e-9

        for idx in range(3):
            self.assertEqual(frames_config[Frame("Q", idx)].sample_duration, 0.1e-9)
            self.assertEqual(frames_config[Frame("Q", idx)].phase, 0.0)

    def test_merge_two_configs(self):
        """Test to see if we can merge two configs."""

        config1 = FramesConfiguration.from_dict(
            {
                Frame("Q", 0): {"frequency": 5.5e9},
                Frame("Q", 1): {"frequency": 5.2e9},
            }
        )

        config2 = FramesConfiguration.from_dict(
            {
                Frame("Q", 1): {"frequency": 4.5e9},
                Frame("Q", 2): {"frequency": 4.2e9},
            }
        )

        for frame, frame_def in config2.items():
            config1[frame] = frame_def

        for idx, freq in enumerate([5.5e9, 4.5e9, 4.2e9]):
            self.assertEqual(config1[Frame("Q", idx)].frequency, freq)


class TestResolvedFrames(QiskitTestCase):
    """Test that resolved frames behave properly."""

    def setUp(self):
        super().setUp()
        self.dt_ = 2.2222222222222221e-10
        self.freq0 = 0.093e9  # Frequency of frame 0
        self.freq1 = 0.125e9  # Frequency of frame 1

    def test_phase(self):
        """Test the phase of the resolved frames."""
        two_pi_dt = 2.0j * np.pi * self.dt_

        r_frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(5.5e9), self.dt_)
        r_frame.set_frequency(99, 5.2e9)

        for time in [0, 55, 98]:
            phase = np.angle(np.exp(two_pi_dt * 5.5e9 * time)) % (2 * np.pi)
            self.assertAlmostEqual(r_frame.phase(time) % (2 * np.pi), phase, places=8)
            self.assertEqual(r_frame.frequency(time), 5.5e9)

        for time in [100, 112]:
            phase = np.angle(np.exp(two_pi_dt * (5.5e9 * 99 + 5.2e9 * (time - 99)))) % (2 * np.pi)
            self.assertAlmostEqual(r_frame.phase(time) % (2 * np.pi), phase, places=8)

    def test_get_phase(self):
        """Test that we get the correct phase as function of time."""

        r_frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(0.0), self.dt_)
        r_frame.set_phase(4, 1.0)
        r_frame.set_phase(8, 2.0)

        expected = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
        for time, phase in enumerate(expected):
            self.assertEqual(r_frame.phase(time), phase)

    def test_get_frequency(self):
        """Test that we get the correct phase as function of time."""

        r_frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(1.0), self.dt_)
        r_frame.set_frequency(4, 2.0)
        r_frame.set_frequency(8, 3.0)

        expected = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
        for time, phase in enumerate(expected):
            self.assertEqual(r_frame.frequency(time), phase)

    def test_set_phase_frequency(self):
        """Test the set frequency and phase methods of the channel trackers."""

        sample_duration = 0.25

        r_frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(1.0), sample_duration)
        r_frame.set_frequency_phase(2, 2.0, 0.5)
        r_frame.set_frequency_phase(4, 3.0, 0.5)
        r_frame.set_frequency_phase(6, 3.0, 1.0)

        # Frequency and phases at the time they are set.
        expected = [(0, 1, 0), (2, 2, 0.5), (4, 3, 0.5), (6, 3, 1)]
        for time, freq, phase in expected:
            self.assertEqual(r_frame.frequency(time), freq)
            self.assertEqual(r_frame.phase(time), phase)

        # Frequency and phases in between setting times i.e. with phase advance.
        # As we have one sample in between times when phases are set, the total
        # phase is the phase at the setting time pulse the phase advance during
        # one sample.
        expected = [(1, 1, 0), (3, 2, 0.5), (5, 3, 0.5), (7, 3, 1)]
        time_step = 1

        for time, freq, phase in expected:
            self.assertEqual(r_frame.frequency(time), freq)

            total_phase = phase + 2 * np.pi * time_step * sample_duration * freq
            self.assertEqual(r_frame.phase(time), total_phase)

    def test_phase_advance(self):
        """Test that phases are properly set when frames are resolved.

        Here we apply four pulses in two frame and alternate between them.
        The resolved schedule will have shift-phase instructions.
        """

        d0 = pulse.DriveChannel(0)
        sig0 = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame("Q", 0))
        sig1 = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame("Q", 1))

        with pulse.build() as sched:
            pulse.play(sig0, d0)
            pulse.play(sig1, d0)
            pulse.play(sig0, d0)
            pulse.play(sig1, d0)

        frames_config = FramesConfiguration.from_dict(
            {
                pulse.Frame("Q", 0): {
                    "frequency": self.freq0,
                    "purpose": "Frame of qubit 0.",
                },
                pulse.Frame("Q", 1): {
                    "frequency": self.freq1,
                    "purpose": "Frame of qubit 1.",
                },
            }
        )
        frames_config.sample_duration = self.dt_

        frame0_ = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(self.freq0), self.dt_)
        frame1_ = ResolvedFrame(pulse.Frame("Q", 1), FrameDefinition(self.freq1), self.dt_)

        # Check that the resolved frames are tracking phases properly
        for time in [0, 160, 320, 480]:
            phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * time * self.dt_))
            self.assertAlmostEqual(frame0_.phase(time) % (2 * np.pi), phase % (2 * np.pi), places=8)

            phase = np.angle(np.exp(2.0j * np.pi * self.freq1 * time * self.dt_)) % (2 * np.pi)
            self.assertAlmostEqual(frame1_.phase(time) % (2 * np.pi), phase % (2 * np.pi), places=8)

        # Check that the proper phase instructions are added to the frame resolved schedules.
        resolved = map_frames(sched, frames_config).instructions

        params = [
            (160, 160, self.freq1, self.freq0, 3),
            (320, 160, self.freq0, self.freq1, 6),
            (480, 160, self.freq1, self.freq0, 9),
        ]

        for time, delta, frame_freq, prev_freq, index in params:
            desired_phase = np.angle(np.exp(2.0j * np.pi * frame_freq * time * self.dt_))
            channel_phase = np.angle(np.exp(2.0j * np.pi * prev_freq * time * self.dt_))

            phase_diff = (desired_phase - channel_phase) % (2 * np.pi)

            self.assertTrue(isinstance(resolved[index][1], pulse.ShiftPhase))

            self.assertEqual(resolved[index][0], time)
            self.assertAlmostEqual(resolved[index][1].phase % (2 * np.pi), phase_diff, places=8)

    def test_phase_advance_with_instructions(self):
        """Test that the phase advances are properly computed with frame instructions."""

        sig = pulse.Signal(pulse.Gaussian(160, 0.1, 40), pulse.Frame("Q", 0))

        with pulse.build() as sched:
            with pulse.align_sequential():
                pulse.play(sig, pulse.DriveChannel(0))
                pulse.shift_phase(1.0, pulse.Frame("Q", 0))

        frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(self.freq0), self.dt_)
        frame.set_frame_instructions(block_to_schedule(sched))

        self.assertAlmostEqual(frame.phase(0), 0.0, places=8)

        # Test the phase right before the shift phase instruction
        phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * 159 * self.dt_))
        self.assertAlmostEqual(frame.phase(159) % (2 * np.pi), phase % (2 * np.pi), places=8)

        # Test the phase at and after the shift phase instruction
        phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * 160 * self.dt_)) + 1.0
        self.assertAlmostEqual(frame.phase(160) % (2 * np.pi), phase % (2 * np.pi), places=8)

        phase = np.angle(np.exp(2.0j * np.pi * self.freq0 * 161 * self.dt_)) + 1.0
        self.assertAlmostEqual(frame.phase(161) % (2 * np.pi), phase % (2 * np.pi), places=8)

    def test_set_frequency(self):
        """Test setting the frequency of the resolved frame."""

        frame = ResolvedFrame(pulse.Frame("Q", 0), FrameDefinition(self.freq1), self.dt_)
        frame.set_frequency(16, self.freq0)
        frame.set_frequency(10, self.freq1)
        frame.set_frequency(10, self.freq1)

        self.assertAlmostEqual(frame.frequency(0), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(10), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(15), self.freq1, places=8)
        self.assertAlmostEqual(frame.frequency(16), self.freq0, places=8)

    def test_implicit_frame(self):
        """If a frame is not specified then the frame of the channel is assumed."""

        qt = 0
        frame01 = Frame("d", qt)
        frame12 = Frame("t", qt)
        xp01 = pulse.Gaussian(160, 0.2, 40)
        xp12 = pulse.Gaussian(160, 0.2, 40)
        d_chan = pulse.DriveChannel(qt)

        # Create a schedule in which frames are specified for all pulses.
        with pulse.build() as rabi_sched_explicit:
            pulse.play(pulse.Signal(xp01, frame01), d_chan)
            pulse.play(pulse.Signal(xp12, frame12), d_chan)
            pulse.play(pulse.Signal(xp01, frame01), d_chan)

        # Create a schedule in which frames are implicitly specified for qubit transitions.
        with pulse.build() as rabi_sched_implicit:
            pulse.play(xp01, d_chan)
            pulse.play(pulse.Signal(xp12, frame12), d_chan)
            pulse.play(xp01, d_chan)

        frames_config = FramesConfiguration.from_dict(
            {
                frame01: {"frequency": 5.5e9, "purpose": "Frame of 0 <-> 1."},
                frame12: {"frequency": 5.2e9, "purpose": "Frame of 1 <-> 2."},
            }
        )
        frames_config.sample_duration = 0.222e-9

        transform = lambda sched: target_qobj_transform(sched, frames_config=frames_config)

        resolved_explicit = transform(rabi_sched_explicit)
        resolved_implicit = transform(rabi_sched_implicit)

        self.assertEqual(resolved_explicit, resolved_implicit)


class TestFrameAssembly(QiskitTestCase):
    """Test that the assembler resolves the frames."""

    def setUp(self):
        """Setup the tests."""

        self.backend = FakeAthens()
        self.defaults = self.backend.defaults()
        self.qubit = 0

        # Setup the frames
        freq12 = self.defaults.qubit_freq_est[self.qubit] - 330e6
        frames_config = self.backend.defaults().frames
        frames_config.add_frame(Frame("t", self.qubit), freq12, purpose="Frame of qutrit.")
        frames_config.sample_duration = self.backend.configuration().dt

        self.xp01 = (
            self.defaults.instruction_schedule_map.get("x", qubits=self.qubit)
            .instructions[0][1]
            .pulse
        )

        self.xp12 = pulse.Gaussian(160, 0.1, 40)

        # Create the expected schedule with all the explicit frames
        with pulse.build(backend=self.backend, default_alignment="sequential") as expected_schedule:
            pulse.play(
                pulse.Signal(self.xp01, pulse.Frame("d", self.qubit)),
                pulse.drive_channel(self.qubit),
            )
            pulse.play(
                pulse.Signal(self.xp12, pulse.Frame("t", self.qubit)),
                pulse.drive_channel(self.qubit),
            )
            pulse.play(
                pulse.Signal(self.xp01, pulse.Frame("d", self.qubit)),
                pulse.drive_channel(self.qubit),
            )
            pulse.measure(qubits=[self.qubit])

        self.expected_schedule = expected_schedule

    def test_frames_in_circuit(self):
        """
        Test that we can add schedules with frames in the calibrations of a circuit
        and that it will compile to the expected outcome.
        """

        with pulse.build(backend=self.backend, default_alignment="sequential") as xp12_schedule:
            pulse.play(
                pulse.Signal(self.xp12, Frame("t", self.qubit)), pulse.drive_channel(self.qubit)
            )

        # Create a quantum circuit and attach the pulse to it.
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.append(Gate("xp12", num_qubits=1, params=[]), (0,))
        circ.x(0)
        circ.measure_active()
        circ.add_calibration("xp12", schedule=xp12_schedule, qubits=[0])

        transform = lambda sched: target_qobj_transform(
            sched, frames_config=self.backend.defaults().frames
        )

        sched_resolved = transform(schedule(circ, self.backend))

        expected_resolved = transform(self.expected_schedule)

        self.assertEqual(sched_resolved, expected_resolved)

        try:
            assemble(circ, self.backend)
        except QiskitError as error:
            self.assertEqual(error.message, "Calibrations with frames are not yet supported.")

    def test_transforms_frame_resolution(self):
        """Test that resolving frames multiple times does not change the schedule."""

        transform = lambda sched: target_qobj_transform(
            sched, frames_config=self.backend.defaults().frames
        )

        resolved1 = transform(self.expected_schedule)
        resolved2 = transform(resolved1)

        self.assertEqual(resolved1, resolved2)

    def test_assemble_frames(self):
        """Test that the assembler resolves the frames of schedule experiments."""

        # Assemble a schedule with frames.
        qobj = assemble(self.expected_schedule, self.backend)

        # Manually resolve frames and assemble.
        transform = lambda sched: target_qobj_transform(
            sched, frames_config=self.backend.defaults().frames
        )
        qobj_resolved = assemble(transform(self.expected_schedule), self.backend)

        self.assertEqual(qobj.experiments, qobj_resolved.experiments)


class TestRequiresFrameMapping(QiskitTestCase):
    """Test the condition on whether a Schedule requires mapping."""

    def test_requires_mapping(self):
        """Test a few simple schedules."""

        # No frame mapping needed.
        with pulse.build() as sched:
            pulse.play(pulse.Gaussian(160, 0.2, 40), pulse.DriveChannel(1))
            pulse.shift_phase(1.57, Frame("d", 0))

        self.assertFalse(requires_frame_mapping(block_to_schedule(sched)))

        # Frame(Q0) needs frame mapping
        with pulse.build() as sched:
            pulse.play(pulse.Gaussian(160, 0.2, 40), pulse.DriveChannel(1))
            pulse.shift_phase(1.57, Frame("Q", 0))

        self.assertTrue(requires_frame_mapping(block_to_schedule(sched)))

        # The signal needs frame mapping
        signal = pulse.Signal(pulse.Gaussian(160, 0.2, 40), Frame("d", 0))
        with pulse.build() as sched:
            pulse.play(signal, pulse.DriveChannel(1))

        self.assertTrue(requires_frame_mapping(block_to_schedule(sched)))
