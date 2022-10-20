# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the RZXCalibrationBuilderNoEcho."""

from math import erf, pi

import numpy as np
from ddt import data, ddt

from qiskit import circuit, schedule
from qiskit.pulse import (
    ControlChannel,
    Delay,
    DriveChannel,
    GaussianSquare,
    Waveform,
    Play,
    ShiftPhase,
    InstructionScheduleMap,
    Schedule,
)
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeAthens
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.calibration.builders import (
    RZXCalibrationBuilder,
    RZXCalibrationBuilderNoEcho,
)


class TestCalibrationBuilder(QiskitTestCase):
    """Test the Calibration Builder."""

    def setUp(self):
        super().setUp()
        self.backend = FakeAthens()
        self.inst_map = self.backend.defaults().instruction_schedule_map


@ddt
class TestRZXCalibrationBuilder(TestCalibrationBuilder):
    """Test RZXCalibrationBuilder."""

    @data(np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_builder_duration(self, theta: float):
        """Test that pulse durations are computed correctly."""
        width = 512.00000001
        sigma = 64
        n_sigmas = 4
        duration = width + n_sigmas * sigma
        sample_mult = 16
        amp = 1.0
        pulse = GaussianSquare(duration=duration, amp=amp, sigma=sigma, width=width)
        instruction = Play(pulse, ControlChannel(1))
        scaled = RZXCalibrationBuilder.rescale_cr_inst(instruction, theta, sample_mult=sample_mult)
        gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * erf(n_sigmas)
        area = gaussian_area + abs(amp) * width
        target_area = abs(theta) / (np.pi / 2.0) * area
        width = (target_area - gaussian_area) / abs(amp)
        expected_duration = round((width + n_sigmas * sigma) / sample_mult) * sample_mult
        self.assertEqual(scaled.duration, expected_duration)

    def test_pass_alive_with_dcx_ish(self):
        """Test if the pass is not terminated by error with direct CX input."""
        cx_sched = Schedule()
        # Fake direct cr
        cx_sched.insert(0, Play(GaussianSquare(800, 0.2, 64, 544), ControlChannel(1)), inplace=True)
        # Fake direct compensation tone
        # Compensation tone doesn't have dedicated pulse class.
        # So it's reported as a waveform now.
        compensation_tone = Waveform(0.1 * np.ones(800, dtype=complex))
        cx_sched.insert(0, Play(compensation_tone, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("cx", (1, 0), schedule=cx_sched)

        theta = pi / 3
        rzx_qc = circuit.QuantumCircuit(2)
        rzx_qc.rzx(theta, 1, 0)

        pass_ = RZXCalibrationBuilder(instruction_schedule_map=inst_map)
        with self.assertWarns(UserWarning):
            # User warning that says q0 q1 is invalid
            cal_qc = PassManager(pass_).run(rzx_qc)
        self.assertEqual(cal_qc, rzx_qc)


class TestRZXCalibrationBuilderNoEcho(TestCalibrationBuilder):
    """Test RZXCalibrationBuilderNoEcho."""

    def test_rzx_calibration_builder(self):
        """Test whether RZXCalibrationBuilderNoEcho scales pulses correctly."""

        # Define a circuit with one RZX gate and an angle theta.
        theta = pi / 3
        rzx_qc = circuit.QuantumCircuit(2)
        rzx_qc.rzx(theta / 2, 1, 0)

        # Verify that there are no calibrations for this circuit yet.
        self.assertEqual(rzx_qc.calibrations, {})

        # apply the RZXCalibrationBuilderNoEcho.
        pass_ = RZXCalibrationBuilderNoEcho(
            instruction_schedule_map=self.backend.defaults().instruction_schedule_map
        )
        cal_qc = PassManager(pass_).run(rzx_qc)
        rzx_qc_duration = schedule(cal_qc, self.backend).duration

        # Check that the calibrations contain the correct instructions
        # and pulses on the correct channels.
        rzx_qc_instructions = cal_qc.calibrations["rzx"][((1, 0), (theta / 2,))].instructions
        self.assertEqual(rzx_qc_instructions[0][1].channel, DriveChannel(0))
        self.assertTrue(isinstance(rzx_qc_instructions[0][1], Play))
        self.assertEqual(rzx_qc_instructions[0][1].pulse.pulse_type, "GaussianSquare")
        self.assertEqual(rzx_qc_instructions[1][1].channel, DriveChannel(1))
        self.assertTrue(isinstance(rzx_qc_instructions[1][1], Delay))
        self.assertEqual(rzx_qc_instructions[2][1].channel, ControlChannel(1))
        self.assertTrue(isinstance(rzx_qc_instructions[2][1], Play))
        self.assertEqual(rzx_qc_instructions[2][1].pulse.pulse_type, "GaussianSquare")

        # Calculate the duration of one scaled Gaussian square pulse from the CX gate.
        cx_sched = self.inst_map.get("cx", qubits=(1, 0))

        crs = []
        for time, inst in cx_sched.instructions:

            # Identify the CR pulses.
            if isinstance(inst, Play) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.channel, ControlChannel):
                    crs.append((time, inst))

        pulse_ = crs[0][1].pulse
        amp = pulse_.amp
        width = pulse_.width
        sigma = pulse_.sigma
        n_sigmas = (pulse_.duration - width) / sigma
        sample_mult = 16

        gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * erf(n_sigmas)
        area = gaussian_area + abs(amp) * width
        target_area = abs(theta) / (np.pi / 2.0) * area
        width = (target_area - gaussian_area) / abs(amp)
        duration = round((width + n_sigmas * sigma) / sample_mult) * sample_mult

        # Check whether the durations of the RZX pulse and
        # the scaled CR pulse from the CX gate match.
        self.assertEqual(rzx_qc_duration, duration)

    def test_pulse_amp_typecasted(self):
        """Test if scaled pulse amplitude is complex type."""
        fake_play = Play(
            GaussianSquare(duration=800, amp=0.1, sigma=64, risefall_sigma_ratio=2),
            ControlChannel(0),
        )
        fake_theta = circuit.Parameter("theta")
        assigned_theta = fake_theta.assign(fake_theta, 0.01)

        scaled = RZXCalibrationBuilderNoEcho.rescale_cr_inst(
            instruction=fake_play, theta=assigned_theta
        )
        scaled_pulse = scaled.pulse

        self.assertIsInstance(scaled_pulse.amp, complex)

    def test_pass_alive_with_dcx_ish(self):
        """Test if the pass is not terminated by error with direct CX input."""
        cx_sched = Schedule()
        # Fake direct cr
        cx_sched.insert(0, Play(GaussianSquare(800, 0.2, 64, 544), ControlChannel(1)), inplace=True)
        # Fake direct compensation tone
        # Compensation tone doesn't have dedicated pulse class.
        # So it's reported as a waveform now.
        compensation_tone = Waveform(0.1 * np.ones(800, dtype=complex))
        cx_sched.insert(0, Play(compensation_tone, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("cx", (1, 0), schedule=cx_sched)

        theta = pi / 3
        rzx_qc = circuit.QuantumCircuit(2)
        rzx_qc.rzx(theta, 1, 0)

        pass_ = RZXCalibrationBuilderNoEcho(instruction_schedule_map=inst_map)
        with self.assertWarns(UserWarning):
            # User warning that says q0 q1 is invalid
            cal_qc = PassManager(pass_).run(rzx_qc)
        self.assertEqual(cal_qc, rzx_qc)
