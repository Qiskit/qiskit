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

from math import pi, erf

import numpy as np
from ddt import data, ddt
from qiskit.converters import circuit_to_dag

from qiskit import circuit, schedule, QiskitError
from qiskit.circuit.library.standard_gates import SXGate, RZGate
from qiskit.providers.fake_provider import FakeHanoi  # TODO - include FakeHanoiV2, FakeSherbrooke
from qiskit.providers.fake_provider import FakeArmonk
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

    @staticmethod
    def get_cr_play(cr_schedule, name):
        """A helper function to filter CR pulses."""

        def _filter_func(time_inst):
            return isinstance(time_inst[1], Play) and time_inst[1].pulse.name.startswith(name)

        return cr_schedule.filter(_filter_func).instructions[0][1]

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

    def u0p_play(self, cr_schedule):
        """Returns the positive CR pulse from cr_schedule."""
        return self.get_cr_play(cr_schedule, "CR90p_u")

    def u0m_play(self, cr_schedule):
        """Returns the negative CR pulse from cr_schedule."""
        return self.get_cr_play(cr_schedule, "CR90m_u")

    def d1p_play(self, cr_schedule):
        """Returns the positive rotary echo pulse from cr_schedule."""
        return self.get_cr_play(cr_schedule, "CR90p_d")

    def d1m_play(self, cr_schedule):
        """Returns the negative rotary echo pulse from cr_schedule."""
        return self.get_cr_play(cr_schedule, "CR90m_d")


@ddt
class TestRZXCalibrationBuilder(TestCalibrationBuilder):
    """Test RZXCalibrationBuilder."""

    def build_forward(
        self,
        backend,
        theta,
        u0p_play,
        d1p_play,
        u0m_play,
        d1m_play,
    ):
        """A helper function to generate reference pulse schedule for forward direction."""
        duration = self.compute_stretch_duration(u0p_play, theta)
        width = self.compute_stretch_width(u0p_play, theta)

        with builder.build(
            backend,
            default_alignment="sequential",
            default_transpiler_settings={"optimization_level": 0},
        ) as ref_sched:
            with builder.align_left():
                # Positive CRs
                u0p_params = u0p_play.pulse.parameters
                u0p_params["duration"] = duration
                u0p_params["width"] = width
                builder.play(
                    GaussianSquare(**u0p_params),
                    ControlChannel(0),
                )
                d1p_params = d1p_play.pulse.parameters
                d1p_params["duration"] = duration
                d1p_params["width"] = width
                builder.play(
                    GaussianSquare(**d1p_params),
                    DriveChannel(1),
                )
            builder.x(0)
            with builder.align_left():
                # Negative CRs
                u0m_params = u0m_play.pulse.parameters
                u0m_params["duration"] = duration
                u0m_params["width"] = width
                builder.play(
                    GaussianSquare(**u0m_params),
                    ControlChannel(0),
                )
                d1m_params = d1m_play.pulse.parameters
                d1m_params["duration"] = duration
                d1m_params["width"] = width
                builder.play(
                    GaussianSquare(**d1m_params),
                    DriveChannel(1),
                )
            builder.x(0)
        return ref_sched

    def build_reverse(
        self,
        backend,
        theta,
        u0p_play,
        d1p_play,
        u0m_play,
        d1m_play,
    ):
        """A helper function to generate reference pulse schedule for backward direction."""
        duration = self.compute_stretch_duration(u0p_play, theta)
        width = self.compute_stretch_width(u0p_play, theta)

        with builder.build(
            backend,
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
                u0p_params = u0p_play.pulse.parameters
                u0p_params["duration"] = duration
                u0p_params["width"] = width
                builder.play(
                    GaussianSquare(**u0p_params),
                    ControlChannel(0),
                )
                d1p_params = d1p_play.pulse.parameters
                d1p_params["duration"] = duration
                d1p_params["width"] = width
                builder.play(
                    GaussianSquare(**d1p_params),
                    DriveChannel(1),
                )
            builder.x(0)
            with builder.align_left():
                # Negative CRs
                u0m_params = u0m_play.pulse.parameters
                u0m_params["duration"] = duration
                u0m_params["width"] = width
                builder.play(
                    GaussianSquare(**u0m_params),
                    ControlChannel(0),
                )
                d1m_params = d1m_play.pulse.parameters
                d1m_params["duration"] = duration
                d1m_params["width"] = width
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

        return ref_sched

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_cr_pulse_stretch(self, theta: float):
        """Test that cross resonance pulse durations are computed correctly."""
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        cr_schedule = inst_map.get("cx", (0, 1))
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.u0p_play(cr_schedule), theta)

        self.assertEqual(
            test_sched.duration, self.compute_stretch_duration(self.u0p_play(cr_schedule), theta)
        )

    @data(-np.pi / 4, 0.1, np.pi / 4, np.pi / 2, np.pi)
    def test_rzx_calibration_rotary_pulse_stretch(self, theta: float):
        """Test that rotary pulse durations are computed correctly."""
        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        cr_schedule = inst_map.get("cx", (0, 1))
        with builder.build() as test_sched:
            RZXCalibrationBuilder.rescale_cr_inst(self.d1p_play(cr_schedule), theta)

        self.assertEqual(
            test_sched.duration, self.compute_stretch_duration(self.d1p_play(cr_schedule), theta)
        )

    def test_raise(self):
        """Test that the correct error is raised."""
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)
        dag = circuit_to_dag(qc)

        backend = FakeArmonk()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)

        qubit_map = {qubit: i for i, qubit in enumerate(dag.qubits)}
        with self.assertRaises(QiskitError):
            for node in dag.gate_nodes():
                qubits = [qubit_map[q] for q in node.qargs]
                _pass.get_calibration(node.op, qubits)

    def test_ecr_cx_forward(self):
        """Test that correct pulse sequence is generated for native CR pair."""
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)
        test_qc = PassManager(_pass).run(qc)

        cr_schedule = inst_map.get("cx", (0, 1))
        ref_sched = self.build_forward(
            backend,
            theta,
            self.u0p_play(cr_schedule),
            self.d1p_play(cr_schedule),
            self.u0m_play(cr_schedule),
            self.d1m_play(cr_schedule),
        )

        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

    def test_ecr_cx_reverse(self):
        """Test that correct pulse sequence is generated for non-native CR pair."""
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 1, 0)

        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map
        _pass = RZXCalibrationBuilder(inst_map)
        test_qc = PassManager(_pass).run(qc)

        cr_schedule = inst_map.get("cx", (0, 1))
        ref_sched = self.build_reverse(
            backend,
            theta,
            self.u0p_play(cr_schedule),
            self.d1p_play(cr_schedule),
            self.u0m_play(cr_schedule),
            self.d1m_play(cr_schedule),
        )

        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

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

    def build_forward(
        self,
        theta,
        u0p_play,
        d1p_play,
    ):
        """A helper function to generate reference pulse schedule for forward direction."""
        duration = self.compute_stretch_duration(u0p_play, 2.0 * theta)
        width = self.compute_stretch_width(u0p_play, 2.0 * theta)

        with builder.build() as ref_sched:
            # Positive CRs
            u0p_params = u0p_play.pulse.parameters
            u0p_params["duration"] = duration
            u0p_params["width"] = width
            builder.play(
                GaussianSquare(**u0p_params),
                ControlChannel(0),
            )
            d1p_params = d1p_play.pulse.parameters
            d1p_params["duration"] = duration
            d1p_params["width"] = width
            builder.play(
                GaussianSquare(**d1p_params),
                DriveChannel(1),
            )
            builder.delay(duration, DriveChannel(0))

        return ref_sched

    def test_ecr_cx_forward(self):
        """Test that correct pulse sequence is generated for native CR pair.

        .. notes::
            No echo builder only supports native direction.
        """
        # Sufficiently large angle to avoid minimum duration, i.e. amplitude rescaling
        theta = np.pi / 4

        qc = circuit.QuantumCircuit(2)
        qc.rzx(theta, 0, 1)

        backend = FakeHanoi()
        inst_map = backend.defaults().instruction_schedule_map

        _pass = RZXCalibrationBuilderNoEcho(inst_map)
        test_qc = PassManager(_pass).run(qc)

        cr_schedule = inst_map.get("cx", (0, 1))
        ref_sched = self.build_forward(
            theta,
            self.u0p_play(cr_schedule),
            self.d1p_play(cr_schedule),
        )

        self.assertEqual(schedule(test_qc, backend), target_qobj_transform(ref_sched))

    # # TODO - write test for forward ECR native pulse
    # def test_ecr_forward(self):

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
