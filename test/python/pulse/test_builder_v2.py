# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse builder with backendV2 context utilities."""

import numpy as np

from qiskit import circuit, pulse
from qiskit.pulse import builder, macros

from qiskit.pulse.instructions import directives
from qiskit.providers.fake_provider import FakeMumbaiV2
from qiskit.pulse import instructions
from qiskit.test import QiskitTestCase
from qiskit.pulse import library, instructions


class TestBuilderV2(QiskitTestCase):
    """Test the pulse builder context with backendV1."""

    def setUp(self):
        super().setUp()
        self.backend = FakeMumbaiV2()


class TestBuilderBaseV2(TestBuilderV2):
    """Test builder base."""

    def test_schedule_supplied(self):
        """Test that schedule is used if it is supplied to the builder."""
        d0 = pulse.DriveChannel(0)
        with pulse.build(name="reference") as reference:
            with pulse.align_sequential():
                pulse.delay(10, d0)

        with pulse.build(schedule=reference) as schedule:
            pass

        self.assertScheduleEqual(schedule, reference)
        self.assertEqual(schedule.name, "reference")

    def test_default_alignment_left(self):
        """Test default left alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="left") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build(self.backend) as reference:
            with pulse.align_left():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_default_alignment_right(self):
        """Test default right alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="right") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build() as reference:
            with pulse.align_right():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_default_alignment_sequential(self):
        """Test default sequential alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="sequential") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build() as reference:
            with pulse.align_sequential():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)


class TestContextsV2(TestBuilderV2):
    """Test builder contexts."""

    def test_align_sequential(self):
        """Test the sequential alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.delay(3, d0)
                pulse.delay(5, d1)
                pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(8, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_align_left(self):
        """Test the left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(11, d2)
                pulse.delay(3, d0)
                with pulse.align_left():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_align_right(self):
        """Test the right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                with pulse.align_right():
                    pulse.delay(11, d2)
                    pulse.delay(3, d0)
                pulse.delay(13, d0)
                pulse.delay(5, d1)

        reference = pulse.Schedule()
        # d0
        reference.insert(8, instructions.Delay(3, d0), inplace=True)
        reference.insert(11, instructions.Delay(13, d0), inplace=True)
        # d1
        reference.insert(19, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_transpiler_settings(self):
        """Test the transpiler settings context.

        Tests that two cx gates are optimized away with higher optimization level.
        """
        twice_cx_qc = circuit.QuantumCircuit(2)
        twice_cx_qc.cx(0, 1)
        twice_cx_qc.cx(0, 1)

        with pulse.build(self.backend) as schedule:
            with pulse.transpiler_settings(optimization_level=0):
                builder.call(twice_cx_qc)
        self.assertNotEqual(len(schedule.instructions), 0)

        with pulse.build(self.backend) as schedule:
            with pulse.transpiler_settings(optimization_level=3):
                builder.call(twice_cx_qc)
        self.assertEqual(len(schedule.instructions), 0)

    def test_scheduler_settings(self):
        """Test the circuit scheduler settings context."""
        inst_map = pulse.InstructionScheduleMap()
        d0 = pulse.DriveChannel(0)
        test_x_sched = pulse.Schedule()
        test_x_sched += instructions.Delay(10, d0)
        inst_map.add("x", (0,), test_x_sched)

        ref_sched = pulse.Schedule()
        ref_sched += pulse.instructions.Call(test_x_sched)

        x_qc = circuit.QuantumCircuit(2)
        x_qc.x(0)

        with pulse.build(backend=self.backend) as schedule:
            with pulse.transpiler_settings(basis_gates=["x"]):
                with pulse.circuit_scheduler_settings(inst_map=inst_map):
                    builder.call(x_qc)

        self.assertScheduleEqual(schedule, ref_sched)

    def test_phase_offset(self):
        """Test the phase offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.phase_offset(3.14, d0):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(-3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_frequency_offset(self):
        """Test the frequency offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.frequency_offset(1e9, d0):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftFrequency(-1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_phase_compensated_frequency_offset(self):
        """Test that the phase offset context properly compensates for phase
        accumulation with backendV2."""
        d0 = pulse.DriveChannel(0)
        with pulse.build(self.backend) as schedule:
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(
            -2 * np.pi * ((1e9 * 10 * self.backend.target.dt) % 1), d0
        )
        reference += instructions.ShiftFrequency(-1e9, d0)
        self.assertScheduleEqual(schedule, reference)


class TestChannelsV2(TestBuilderV2):
    """Test builder channels."""

    def test_drive_channel(self):
        """Text context builder drive channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.drive_channel(0), pulse.DriveChannel(0))

    def test_measure_channel(self):
        """Text context builder measure channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.measure_channel(0), pulse.MeasureChannel(0))

    def test_acquire_channel(self):
        """Text context builder acquire channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.acquire_channel(0), pulse.AcquireChannel(0))

    def test_control_channel(self):
        """Text context builder control channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.control_channels(0, 1)[0], pulse.ControlChannel(0))


class TestInstructionsV2(TestBuilderV2):
    """Test builder instructions."""

    def test_delay(self):
        """Test delay instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_parametric_pulse(self):
        """Test play instruction with parametric pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Constant(10, 1.0)

        with pulse.build() as schedule:
            pulse.play(test_pulse, d0)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_sample_pulse(self):
        """Test play instruction with sample pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Waveform([0.0, 0.0])

        with pulse.build() as schedule:
            pulse.play(test_pulse, d0)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_array_pulse(self):
        """Test play instruction on an array directly."""
        d0 = pulse.DriveChannel(0)
        test_array = np.array([0.0, 0.0], dtype=np.complex_)

        with pulse.build() as schedule:
            pulse.play(test_array, d0)

        reference = pulse.Schedule()
        test_pulse = pulse.Waveform(test_array)
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_name_argument(self):
        """Test name argument for play instruction."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Constant(10, 1.0)

        with pulse.build() as schedule:
            pulse.play(test_pulse, channel=d0, name="new_name")

        self.assertEqual(schedule.instructions[0][1].name, "new_name")

    def test_acquire_memory_slot(self):
        """Test acquire instruction into memory slot."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, acquire0, mem0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertScheduleEqual(schedule, reference)

    def test_acquire_register_slot(self):
        """Test acquire instruction into register slot."""
        acquire0 = pulse.AcquireChannel(0)
        reg0 = pulse.RegisterSlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, acquire0, reg0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, reg_slot=reg0)

        self.assertScheduleEqual(schedule, reference)

    def test_acquire_qubit(self):
        """Test acquire instruction on qubit."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, 0, mem0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertScheduleEqual(schedule, reference)

    def test_instruction_name_argument(self):
        """Test setting the name of an instruction."""
        d0 = pulse.DriveChannel(0)

        for instruction_method in [
            pulse.delay,
            pulse.set_frequency,
            pulse.set_phase,
            pulse.shift_frequency,
            pulse.shift_phase,
        ]:
            with pulse.build() as schedule:
                instruction_method(0, d0, name="instruction_name")
            self.assertEqual(schedule.instructions[0][1].name, "instruction_name")

    def test_set_frequency(self):
        """Test set frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_frequency(1e9, d0)

        reference = pulse.Schedule()
        reference += instructions.SetFrequency(1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_shift_frequency(self):
        """Test shift frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_frequency(0.1e9, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(0.1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_set_phase(self):
        """Test set phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_phase(3.14, d0)

        reference = pulse.Schedule()
        reference += instructions.SetPhase(3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_shift_phase(self):
        """Test shift phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_phase(3.14, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_snapshot(self):
        """Test snapshot instruction."""
        with pulse.build() as schedule:
            pulse.snapshot("test", "state")

        reference = pulse.Schedule()
        reference += instructions.Snapshot("test", "state")

        self.assertScheduleEqual(schedule, reference)


class TestDirectivesV2(TestBuilderV2):
    """Test builder directives."""

    def test_barrier_with_align_right(self):
        """Test barrier directive with right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.delay(3, d0)
                pulse.barrier(d0, d1, d2)
                pulse.delay(11, d2)
                with pulse.align_right():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(7, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(9, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(3, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_barrier_with_align_left(self):
        """Test barrier directive with left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(3, d0)
                pulse.barrier(d0, d1, d2)
                pulse.delay(11, d2)
                with pulse.align_left():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(3, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_barrier_on_qubits(self):
        """Test barrier directive on qubits with backendV2.
        A part of qubits map of Mumbai
            0 -- 1 -- 4 --
                |
                |
                2
        """
        with pulse.build(self.backend) as schedule:
            pulse.barrier(0, 1)
        reference = pulse.ScheduleBlock()
        reference += directives.RelativeBarrier(
            pulse.DriveChannel(0),
            pulse.DriveChannel(1),
            pulse.MeasureChannel(0),
            pulse.MeasureChannel(1),
            pulse.ControlChannel(0),
            pulse.ControlChannel(1),
            pulse.ControlChannel(2),
            pulse.ControlChannel(3),
            pulse.ControlChannel(4),
            pulse.ControlChannel(8),
            pulse.AcquireChannel(0),
            pulse.AcquireChannel(1),
        )
        self.assertEqual(schedule, reference)


class TestUtilitiesV2(TestBuilderV2):
    """Test builder utilities."""

    def test_active_backend(self):
        """Test getting active builder backend."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.active_backend(), self.backend)

    def test_append_schedule(self):
        """Test appending a schedule to the active builder."""
        d0 = pulse.DriveChannel(0)
        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        with pulse.build() as schedule:
            builder.call(reference)

        self.assertScheduleEqual(schedule, reference)

    def test_append_instruction(self):
        """Test appending an instruction to the active builder."""
        d0 = pulse.DriveChannel(0)
        instruction = instructions.Delay(10, d0)

        with pulse.build() as schedule:
            builder.append_instruction(instruction)

        self.assertScheduleEqual(schedule, (0, instruction))

    def test_qubit_channels(self):
        """Test getting the qubit channels of the active builder's backend."""
        with pulse.build(self.backend):
            qubit_channels = pulse.qubit_channels(0)

        self.assertEqual(
            qubit_channels,
            {
                pulse.DriveChannel(0),
                pulse.MeasureChannel(0),
                pulse.AcquireChannel(0),
                pulse.ControlChannel(0),
                pulse.ControlChannel(1),
            },
        )

    def test_active_transpiler_settings(self):
        """Test setting settings of active builder's transpiler."""
        with pulse.build(self.backend):
            self.assertFalse(pulse.active_transpiler_settings())
            with pulse.transpiler_settings(test_setting=1):
                self.assertEqual(pulse.active_transpiler_settings()["test_setting"], 1)

    def test_active_circuit_scheduler_settings(self):
        """Test setting settings of active builder's circuit scheduler."""
        with pulse.build(self.backend):
            self.assertFalse(pulse.active_circuit_scheduler_settings())
            with pulse.circuit_scheduler_settings(test_setting=1):
                self.assertEqual(pulse.active_circuit_scheduler_settings()["test_setting"], 1)

    def test_num_qubits(self):
        """Test builder utility to get number of qubits with backendV2."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.num_qubits(), 27)

    def test_samples_to_seconds(self):
        """Test samples to time with backendV2"""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            time = pulse.samples_to_seconds(100)
            self.assertTrue(isinstance(time, float))
            self.assertEqual(pulse.samples_to_seconds(100), 10)

    def test_samples_to_seconds_array(self):
        """Test samples to time (array format) with backendV2."""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            samples = np.array([100, 200, 300])
            times = pulse.samples_to_seconds(samples)
            self.assertTrue(np.issubdtype(times.dtype, np.floating))
            np.testing.assert_allclose(times, np.array([10, 20, 30]))

    def test_seconds_to_samples(self):
        """Test time to samples with backendV2"""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            samples = pulse.seconds_to_samples(10)
            self.assertTrue(isinstance(samples, int))
            self.assertEqual(pulse.seconds_to_samples(10), 100)

    def test_seconds_to_samples_array(self):
        """Test time to samples (array format) with backendV2."""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            times = np.array([10, 20, 30])
            samples = pulse.seconds_to_samples(times)
            self.assertTrue(np.issubdtype(samples.dtype, np.integer))
            np.testing.assert_allclose(pulse.seconds_to_samples(times), np.array([100, 200, 300]))


class TestMacrosV2(TestBuilderV2):
    """Test builder macros with backendV2."""

    def test_macro(self):
        """Test builder macro decorator."""

        @pulse.macro
        def nested(a):
            pulse.play(pulse.Gaussian(100, a, 20), pulse.drive_channel(0))
            return a * 2

        @pulse.macro
        def test():
            pulse.play(pulse.Constant(100, 1.0), pulse.drive_channel(0))
            output = nested(0.5)
            return output

        with pulse.build(self.backend) as schedule:
            output = test()
            self.assertEqual(output, 0.5 * 2)

        reference = pulse.Schedule()
        reference += pulse.Play(pulse.Constant(100, 1.0), pulse.DriveChannel(0))
        reference += pulse.Play(pulse.Gaussian(100, 0.5, 20), pulse.DriveChannel(0))

        self.assertScheduleEqual(schedule, reference)

    def test_measure(self):
        """Test utility function - measure with backendV2."""
        with pulse.build(self.backend) as schedule:
            reg = pulse.measure(0)

        self.assertEqual(reg, pulse.MemorySlot(0))

        reference = macros.measure(qubits=[0], backend=self.backend, meas_map=self.backend.meas_map)

        self.assertScheduleEqual(schedule, reference)

    def test_measure_multi_qubits(self):
        """Test utility function - measure with multi qubits with backendV2."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure([0, 1])

        self.assertListEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])

        reference = macros.measure(
            qubits=[0, 1], backend=self.backend, meas_map=self.backend.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_all(self):
        """Test utility function - measure with backendV2.."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure_all()

        self.assertEqual(regs, [pulse.MemorySlot(i) for i in range(self.backend.num_qubits)])
        reference = macros.measure_all(self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_delay_qubit(self):
        """Test delaying on a qubit macro."""
        with pulse.build(self.backend) as schedule:
            pulse.delay_qubits(10, 0)

        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        a0 = pulse.AcquireChannel(0)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)

        self.assertScheduleEqual(schedule, reference)

    def test_delay_qubits(self):
        """Test delaying on multiple qubits with backendV2 to make sure we don't insert delays twice."""
        with pulse.build(self.backend) as schedule:
            pulse.delay_qubits(10, 0, 1)

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        m0 = pulse.MeasureChannel(0)
        m1 = pulse.MeasureChannel(1)
        a0 = pulse.AcquireChannel(0)
        a1 = pulse.AcquireChannel(1)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)
        u2 = pulse.ControlChannel(2)
        u3 = pulse.ControlChannel(3)
        u4 = pulse.ControlChannel(4)
        u8 = pulse.ControlChannel(8)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, d1)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, m1)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, a1)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)
        reference += instructions.Delay(10, u2)
        reference += instructions.Delay(10, u3)
        reference += instructions.Delay(10, u4)
        reference += instructions.Delay(10, u8)

        self.assertScheduleEqual(schedule, reference)
