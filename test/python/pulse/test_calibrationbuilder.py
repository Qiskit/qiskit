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

from math import pi, erf, ceil

import numpy as np

from qiskit import circuit, schedule
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.pulse import (
    Play,
    Delay,
    ShiftPhase,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
)
from qiskit.transpiler.passes.scheduling.calibration_creators import (
    RZXCalibrationBuilderNoEcho,
)
from qiskit.test.mock import FakeAthens


class TestCalibrationBuilder(QiskitTestCase):
    """Test the Calibration Builder."""

    def setUp(self):
        super().setUp()
        self.backend = FakeAthens()
        self.inst_map = self.backend.defaults().instruction_schedule_map


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
        pass_ = RZXCalibrationBuilderNoEcho(self.backend)
        cal_qc = PassManager(pass_).run(rzx_qc)
        rzx_qc_duration = schedule(cal_qc, self.backend).duration

        # Check that the calibrations contain the correct instructions
        # and pulses on the correct channels.
        # pylint: disable=no-member
        rzx_qc_instructions = cal_qc.calibrations["rzx"][((1, 0), (theta / 2,))].instructions
        self.assertEqual(rzx_qc_instructions[0][1].channel, DriveChannel(0))
        self.assertTrue(isinstance(rzx_qc_instructions[0][1], Play))
        self.assertTrue(isinstance(rzx_qc_instructions[0][1].pulse, GaussianSquare))
        self.assertEqual(rzx_qc_instructions[1][1].channel, DriveChannel(1))
        self.assertTrue(isinstance(rzx_qc_instructions[1][1], Delay))
        self.assertEqual(rzx_qc_instructions[2][1].channel, ControlChannel(1))
        self.assertTrue(isinstance(rzx_qc_instructions[2][1], Play))
        self.assertTrue(isinstance(rzx_qc_instructions[2][1].pulse, GaussianSquare))

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
        duration = ceil((width + n_sigmas * sigma) / sample_mult) * sample_mult

        # Check whether the durations of the RZX pulse and
        # the scaled CR pulse from the CX gate match.
        self.assertEqual(rzx_qc_duration, duration)
