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

from math import pi, erf  # pylint: disable=no-name-in-module

import numpy as np
from ddt import data, ddt

from qiskit import circuit, schedule
from qiskit.circuit.library.standard_gates import SXGate, RZGate
from qiskit.providers.fake_provider import FakeHanoi
from qiskit.pulse import (
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Waveform,
    Play,
    InstructionScheduleMap,
    Schedule,
)
from qiskit.pulse import builder
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.calibration.builders import (
    RZXCalibrationBuilder,
    RZXCalibrationBuilderNoEcho,
)


class TestCalibrationBuilder(QiskitTestCase):
    """Test the Calibration Builder."""

    # CR parameters
    __risefall = 4
    __angle = np.pi / 2
    __granularity = 16

    def setUp(self):
        super().setUp()
        # This backend provides CX(0,1) with native ECR direction.
        self.backend = FakeHanoi()
        self.inst_map = self.backend.defaults().instruction_schedule_map

        # Get positive CR tone
        def _get_cr(time_inst, name):
            return isinstance(time_inst[1], Play) and time_inst[1].pulse.name.startswith(name)

        cx_sched = self.inst_map.get("cx", (0, 1))

        # CR tone
        self.u0p_play = cx_sched.filter(lambda elm: _get_cr(elm, "CR90p_u")).instructions[0][1]
        self.u0m_play = cx_sched.filter(lambda elm: _get_cr(elm, "CR90m_u")).instructions[0][1]

        # Rotary tone
        self.d1p_play = cx_sched.filter(lambda elm: _get_cr(elm, "CR90p_d")).instructions[0][1]
        self.d1m_play = cx_sched.filter(lambda elm: _get_cr(elm, "CR90m_d")).instructions[0][1]

    def compute_stretch_duration(self, play_gaussian_square_pulse, theta):
        """Compute duration of stretched Gaussian Square pulse."""
        pulse = play_gaussian_square_pulse.pulse
        sigma = pulse.sigma
        width = self.compute_stretch_width(play_gaussian_square_pulse, theta)

        duration = width + sigma * self.__risefall
        return round(duration / self.__granularity) * self.__granularity

    def compute_stretch_width(self, play_gaussian_square_pulse, theta):
        """Compute width of stretched Gaussian Square pulse."""
        pulse = play_gaussian_square_pulse.pulse
        sigma = pulse.sigma
        width = pulse.width

        risefall_area = sigma * np.sqrt(2 * np.pi) * erf(self.__risefall)
        full_area = risefall_area + width

        target_area = abs(theta) / self.__angle * full_area
        return max(0, target_area - risefall_area)


@ddt
class TestRZXCalibrationBuilder(TestCalibrationBuilder):
    """Test RZXCalibrationBuilder."""

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_cr_pulse_stretch(self, theta: float):
        """Test that cross resonance pulse durations are computed correctly."""
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.u0p_play, theta)

        self.assertEqual(test_sched.duration, self.compute_stretch_duration(self.u0p_play, theta))

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_rotary_pulse_stretch(self, theta: float):
        """Test that rotary pulse durations are computed correctly."""
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.d1p_play, theta)

        self.assertEqual(test_sched.duration, self.compute_stretch_duration(self.d1p_play, theta))

    def test_native_cr(self):
        """Test that correct pulse sequence is generated for native CR pair."""
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        _pass = RZXCalibrationBuilder(self.inst_map)
        test_qc = PassManager(_pass).run(qc)

        duration = self.compute_stretch_duration(self.u0p_play, theta)
        with builder.build(
            self.backend,
            default_alignment="sequential",
            default_transpiler_settings={"optimization_level": 0},
        ) as ref_sched:
            with builder.align_left():
                # Positive CRs
                u0p_params = self.u0p_play.pulse.parameters
                u0p_params["duration"] = duration
                u0p_params["width"] = self.compute_stretch_width(self.u0p_play, theta)
                builder.play(
                    GaussianSquare(**u0p_params),
                    ControlChannel(0),
                )
                d1p_params = self.d1p_play.pulse.parameters
                d1p_params["duration"] = duration
                d1p_params["width"] = self.compute_stretch_width(self.d1p_play, theta)
                builder.play(
                    GaussianSquare(**d1p_params),
                    DriveChannel(1),
                )
            builder.x(0)
            with builder.align_left():
                # Negative CRs
                u0m_params = self.u0m_play.pulse.parameters
                u0m_params["duration"] = duration
                u0m_params["width"] = self.compute_stretch_width(self.u0m_play, theta)
                builder.play(
                    GaussianSquare(**u0m_params),
                    ControlChannel(0),
                )
                d1m_params = self.d1m_play.pulse.parameters
                d1m_params["duration"] = duration
                d1m_params["width"] = self.compute_stretch_width(self.d1m_play, theta)
                builder.play(
                    GaussianSquare(**d1m_params),
                    DriveChannel(1),
                )
            builder.x(0)

        self.assertEqual(schedule(test_qc, self.backend), target_qobj_transform(ref_sched))

    def test_non_native_cr(self):
        """Test that correct pulse sequence is generated for non-native CR pair."""
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 1, 0)

        _pass = RZXCalibrationBuilder(self.inst_map)
        test_qc = PassManager(_pass).run(qc)

        duration = self.compute_stretch_duration(self.u0p_play, theta)
        with builder.build(
            self.backend,
            default_alignment="sequential",
            default_transpiler_settings={"optimization_level": 0},
        ) as ref_sched:
            # Hadamard gates
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(SXGate(), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            builder.call_gate(SXGate(), qubits=(1,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            with builder.align_left():
                # Positive CRs
                u0p_params = self.u0p_play.pulse.parameters
                u0p_params["duration"] = duration
                u0p_params["width"] = self.compute_stretch_width(self.u0p_play, theta)
                builder.play(
                    GaussianSquare(**u0p_params),
                    ControlChannel(0),
                )
                d1p_params = self.d1p_play.pulse.parameters
                d1p_params["duration"] = duration
                d1p_params["width"] = self.compute_stretch_width(self.d1p_play, theta)
                builder.play(
                    GaussianSquare(**d1p_params),
                    DriveChannel(1),
                )
            builder.x(0)
            with builder.align_left():
                # Negative CRs
                u0m_params = self.u0m_play.pulse.parameters
                u0m_params["duration"] = duration
                u0m_params["width"] = self.compute_stretch_width(self.u0m_play, theta)
                builder.play(
                    GaussianSquare(**u0m_params),
                    ControlChannel(0),
                )
                d1m_params = self.d1m_play.pulse.parameters
                d1m_params["duration"] = duration
                d1m_params["width"] = self.compute_stretch_width(self.d1m_play, theta)
                builder.play(
                    GaussianSquare(**d1m_params),
                    DriveChannel(1),
                )
            builder.x(0)
            # Hadamard gates
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(SXGate(), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(0,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))
            builder.call_gate(SXGate(), qubits=(1,))
            builder.call_gate(RZGate(np.pi / 2), qubits=(1,))

        self.assertEqual(schedule(test_qc, self.backend), target_qobj_transform(ref_sched))

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

    def test_native_cr(self):
        """Test that correct pulse sequence is generated for native CR pair.

        .. notes::
            No echo builder only supports native direction.
        """
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        _pass = RZXCalibrationBuilderNoEcho(self.inst_map)
        test_qc = PassManager(_pass).run(qc)

        duration = self.compute_stretch_duration(self.u0p_play, 2.0 * theta)
        with builder.build() as ref_sched:
            # Positive CRs
            u0p_params = self.u0p_play.pulse.parameters
            u0p_params["duration"] = duration
            u0p_params["width"] = self.compute_stretch_width(self.u0p_play, 2.0 * theta)
            builder.play(
                GaussianSquare(**u0p_params),
                ControlChannel(0),
            )
            d1p_params = self.d1p_play.pulse.parameters
            d1p_params["duration"] = duration
            d1p_params["width"] = self.compute_stretch_width(self.d1p_play, 2.0 * theta)
            builder.play(
                GaussianSquare(**d1p_params),
                DriveChannel(1),
            )
            builder.delay(duration, DriveChannel(0))

        self.assertEqual(schedule(test_qc, self.backend), target_qobj_transform(ref_sched))

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
