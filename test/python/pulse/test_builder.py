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

"""Test pulse builder context utilities."""

from math import pi

import numpy as np

from qiskit import circuit, compiler, pulse
from qiskit.pulse import builder, exceptions, macros
from qiskit.pulse.instructions import directives
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.test.mock.utils import ConfigurableFakeBackend as ConfigurableBackend
from qiskit.pulse import library, instructions


class TestBuilder(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        super().setUp()
        self.backend = FakeOpenPulse2Q()
        self.configuration = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.inst_map = self.defaults.instruction_schedule_map

    def assertScheduleEqual(self, program, target):
        """Assert an error when two pulse programs are not equal.

        .. note:: Two programs are converted into standard execution format then compared.
        """
        self.assertEqual(target_qobj_transform(program), target_qobj_transform(target))


class TestBuilderBase(TestBuilder):
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


class TestContexts(TestBuilder):
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

    def test_inline(self):
        """Test the inlining context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            pulse.delay(3, d0)
            with pulse.inline():
                # this alignment will be ignored due to inlining.
                with pulse.align_right():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        reference += instructions.Delay(7, d0)
        # d1
        reference += instructions.Delay(5, d1)

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
        accumulation."""
        d0 = pulse.DriveChannel(0)

        with pulse.build(self.backend) as schedule:
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(
            -2 * np.pi * ((1e9 * 10 * self.configuration.dt) % 1), d0
        )
        reference += instructions.ShiftFrequency(-1e9, d0)

        self.assertScheduleEqual(schedule, reference)


class TestChannels(TestBuilder):
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


class TestInstructions(TestBuilder):
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


class TestDirectives(TestBuilder):
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
        """Test barrier directive on qubits."""
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
            pulse.AcquireChannel(0),
            pulse.AcquireChannel(1),
        )
        self.assertEqual(schedule, reference)

    def test_trivial_barrier(self):
        """Test that trivial barrier is not added."""
        with pulse.build() as schedule:
            pulse.barrier(pulse.DriveChannel(0))

        self.assertEqual(schedule, pulse.ScheduleBlock())


class TestUtilities(TestBuilder):
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
        """Test builder utility to get number of qubits."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.num_qubits(), 2)

    def test_samples_to_seconds(self):
        """Test samples to time"""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            time = pulse.samples_to_seconds(100)
            self.assertTrue(isinstance(time, float))
            self.assertEqual(pulse.samples_to_seconds(100), 10)

    def test_samples_to_seconds_array(self):
        """Test samples to time (array format)."""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            samples = np.array([100, 200, 300])
            times = pulse.samples_to_seconds(samples)
            self.assertTrue(np.issubdtype(times.dtype, np.floating))
            np.testing.assert_allclose(times, np.array([10, 20, 30]))

    def test_seconds_to_samples(self):
        """Test time to samples"""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            samples = pulse.seconds_to_samples(10)
            self.assertTrue(isinstance(samples, int))
            self.assertEqual(pulse.seconds_to_samples(10), 100)

    def test_seconds_to_samples_array(self):
        """Test time to samples (array format)."""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            times = np.array([10, 20, 30])
            samples = pulse.seconds_to_samples(times)
            self.assertTrue(np.issubdtype(samples.dtype, np.integer))
            np.testing.assert_allclose(pulse.seconds_to_samples(times), np.array([100, 200, 300]))


class TestMacros(TestBuilder):
    """Test builder macros."""

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
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            reg = pulse.measure(0)

        self.assertEqual(reg, pulse.MemorySlot(0))

        reference = macros.measure(
            qubits=[0], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_multi_qubits(self):
        """Test utility function - measure with multi qubits."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure([0, 1])

        self.assertListEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])

        reference = macros.measure(
            qubits=[0, 1], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_all(self):
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure_all()

        self.assertEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])
        reference = macros.measure_all(self.backend)

        self.assertScheduleEqual(schedule, reference)

        backend_100q = ConfigurableBackend("100q", 100)
        with pulse.build(backend_100q) as schedule:
            regs = pulse.measure_all()

        reference = backend_100q.defaults().instruction_schedule_map.get(
            "measure", list(range(100))
        )

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
        """Test delaying on multiple qubits to make sure we don't insert delays twice."""
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

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, d1)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, m1)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, a1)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)

        self.assertScheduleEqual(schedule, reference)


class TestGates(TestBuilder):
    """Test builder gates."""

    def test_cx(self):
        """Test cx gate."""
        with pulse.build(self.backend) as schedule:
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference_qc.cx(0, 1)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_u1(self):
        """Test u1 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u1(np.pi / 2, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U1Gate(np.pi / 2), [0])
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_u2(self):
        """Test u2 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u2(np.pi / 2, 0, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U2Gate(np.pi / 2, 0), [0])
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_u3(self):
        """Test u3 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u3(np.pi / 8, np.pi / 16, np.pi / 4, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U3Gate(np.pi / 8, np.pi / 16, np.pi / 4), [0])
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_x(self):
        """Test x gate."""
        with pulse.build(self.backend) as schedule:
            pulse.x(0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.x(0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_lazy_evaluation_with_transpiler(self):
        """Test that the two cx gates are optimizied away by the transpiler."""
        with pulse.build(self.backend) as schedule:
            pulse.cx(0, 1)
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_measure(self):
        """Test pulse measurement macro against circuit measurement and
        ensure agreement."""
        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.x(0)
                pulse.measure(0)

        reference_qc = circuit.QuantumCircuit(1, 1)
        reference_qc.x(0)
        reference_qc.measure(0, 0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_backend_require(self):
        """Test that a backend is required to use a gate."""
        with self.assertRaises(exceptions.BackendNotSet):
            with pulse.build():
                pulse.x(0)


class TestBuilderComposition(TestBuilder):
    """Test more sophisticated composite builder examples."""

    def test_complex_build(self):
        """Test a general program build with nested contexts,
        circuits and macros."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        delay_dur = 30
        short_dur = 20
        long_dur = 49

        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.delay(delay_dur, d0)
                pulse.u2(0, pi / 2, 1)
            with pulse.align_right():
                pulse.play(library.Constant(short_dur, 0.1), d1)
                pulse.play(library.Constant(long_dur, 0.1), d2)
                pulse.u2(0, pi / 2, 1)
            with pulse.align_left():
                pulse.u2(0, pi / 2, 0)
                pulse.u2(0, pi / 2, 1)
                pulse.u2(0, pi / 2, 0)
            pulse.measure(0)

        # prepare and schedule circuits that will be used.
        single_u2_qc = circuit.QuantumCircuit(2)
        single_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [1])
        single_u2_qc = compiler.transpile(single_u2_qc, self.backend)
        single_u2_sched = compiler.schedule(single_u2_qc, self.backend)

        # sequential context
        sequential_reference = pulse.Schedule()
        sequential_reference += instructions.Delay(delay_dur, d0)
        sequential_reference.insert(delay_dur, single_u2_sched, inplace=True)

        # align right
        align_right_reference = pulse.Schedule()
        align_right_reference += pulse.Play(library.Constant(long_dur, 0.1), d2)
        align_right_reference.insert(
            long_dur - single_u2_sched.duration, single_u2_sched, inplace=True
        )
        align_right_reference.insert(
            long_dur - single_u2_sched.duration - short_dur,
            pulse.Play(library.Constant(short_dur, 0.1), d1),
            inplace=True,
        )

        # align left
        triple_u2_qc = circuit.QuantumCircuit(2)
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [0])
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [1])
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [0])
        triple_u2_qc = compiler.transpile(triple_u2_qc, self.backend)
        align_left_reference = compiler.schedule(triple_u2_qc, self.backend, method="alap")

        # measurement
        measure_reference = macros.measure(
            qubits=[0], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )
        reference = pulse.Schedule()
        reference += sequential_reference
        # Insert so that the long pulse on d2 occurs as early as possible
        # without an overval on d1.
        insert_time = reference.ch_stop_time(d1) - align_right_reference.ch_start_time(d1)
        reference.insert(insert_time, align_right_reference, inplace=True)
        reference.insert(reference.ch_stop_time(d0, d1), align_left_reference, inplace=True)
        reference += measure_reference

        self.assertScheduleEqual(schedule, reference)


class TestSubroutineCall(TestBuilder):
    """Test for calling subroutine."""

    def test_call(self):
        """Test calling schedule instruction."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        reference = pulse.Schedule()
        reference = reference.insert(10, instructions.Delay(10, d0))
        reference += instructions.Delay(20, d1)

        ref_sched = pulse.Schedule()
        ref_sched += pulse.instructions.Call(reference)

        with pulse.build() as schedule:
            with pulse.align_right():
                builder.call(reference)

        self.assertScheduleEqual(schedule, ref_sched)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.call(reference)

        self.assertScheduleEqual(schedule, ref_sched)

    def test_call_circuit(self):
        """Test calling circuit instruction."""
        inst_map = self.inst_map
        reference = inst_map.get("u1", (0,), 0.0)

        ref_sched = pulse.Schedule()
        ref_sched += pulse.instructions.Call(reference)

        u1_qc = circuit.QuantumCircuit(2)
        u1_qc.append(circuit.library.U1Gate(0.0), [0])

        transpiler_settings = {"optimization_level": 0}

        with pulse.build(self.backend, default_transpiler_settings=transpiler_settings) as schedule:
            with pulse.align_right():
                builder.call(u1_qc)

        self.assertScheduleEqual(schedule, ref_sched)

    def test_call_circuit_with_cregs(self):
        """Test calling of circuit wiht classical registers."""

        qc = circuit.QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        with pulse.build(self.backend) as schedule:
            pulse.call(qc)

        reference_qc = compiler.transpile(qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        ref_sched = pulse.Schedule()
        ref_sched += pulse.instructions.Call(reference)

        self.assertScheduleEqual(schedule, ref_sched)

    def test_call_gate_and_circuit(self):
        """Test calling circuit with gates."""
        h_control = circuit.QuantumCircuit(2)
        h_control.h(0)

        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                # this is circuit, a subroutine stored as Call instruction
                pulse.call(h_control)
                # this is instruction, not subroutine
                pulse.cx(0, 1)
                # this is macro, not subroutine
                pulse.measure([0, 1])

        # subroutine
        h_reference = compiler.schedule(compiler.transpile(h_control, self.backend), self.backend)

        # gate
        cx_circ = circuit.QuantumCircuit(2)
        cx_circ.cx(0, 1)
        cx_reference = compiler.schedule(compiler.transpile(cx_circ, self.backend), self.backend)

        # measurement
        measure_reference = macros.measure(
            qubits=[0, 1], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )

        reference = pulse.Schedule()
        reference += pulse.instructions.Call(h_reference)
        reference += cx_reference
        reference += measure_reference << reference.duration

        self.assertScheduleEqual(schedule, reference)

    def test_subroutine_not_transpiled(self):
        """Test called circuit is frozen as a subroutine."""
        subprogram = circuit.QuantumCircuit(1)
        subprogram.x(0)

        transpiler_settings = {"optimization_level": 2}

        with pulse.build(self.backend, default_transpiler_settings=transpiler_settings) as schedule:
            pulse.call(subprogram)
            pulse.call(subprogram)

        self.assertNotEqual(len(target_qobj_transform(schedule).instructions), 0)

    def test_subroutine_not_transformed(self):
        """Test called schedule is not transformed."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        subprogram = pulse.Schedule()
        subprogram.insert(0, pulse.Delay(30, d0), inplace=True)
        subprogram.insert(10, pulse.Delay(10, d1), inplace=True)

        with pulse.build() as target:
            with pulse.align_right():
                pulse.delay(10, d1)
                pulse.call(subprogram)

        reference = pulse.Schedule()
        reference.insert(0, pulse.Delay(10, d1), inplace=True)
        reference.insert(10, pulse.Delay(30, d0), inplace=True)
        reference.insert(20, pulse.Delay(10, d1), inplace=True)

        self.assertScheduleEqual(target, reference)

    def test_deepcopying_subroutine(self):
        """Test if deepcopying the schedule can copy inline subroutine."""
        from copy import deepcopy

        with pulse.build() as subroutine:
            pulse.delay(10, pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine)

        copied_prog = deepcopy(main_prog)

        main_call = main_prog.instructions[0]
        copy_call = copied_prog.instructions[0]

        self.assertNotEqual(id(main_call), id(copy_call))

    def test_call_with_parameters(self):
        """Test call subroutine with parameters."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)
            pulse.call(subroutine, amp=0.3)

        self.assertEqual(main_prog.is_parameterized(), False)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.3)

    def test_call_partly_with_parameters(self):
        """Test multiple calls partly with parameters then assign."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)
            pulse.call(subroutine)

        self.assertEqual(main_prog.is_parameterized(), True)

        main_prog.assign_parameters({amp: 0.5})
        self.assertEqual(main_prog.is_parameterized(), False)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.5)

    def test_call_with_not_existing_parameter(self):
        """Test call subroutine with parameter not defined."""
        amp = circuit.Parameter("amp1")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with self.assertRaises(exceptions.PulseError):
            with pulse.build():
                pulse.call(subroutine, amp=0.1)

    def test_call_with_common_parameter(self):
        """Test call subroutine with parameter that is defined multiple times."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(320, amp, 80), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.1)

    def test_call_with_parameter_name_collision(self):
        """Test call subroutine with duplicated parameter names."""
        amp1 = circuit.Parameter("amp")
        amp2 = circuit.Parameter("amp")
        sigma = circuit.Parameter("sigma")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp1, sigma), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, amp2, sigma), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, value_dict={amp1: 0.1, amp2: 0.2}, sigma=40)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_0.pulse.sigma, 40)
        self.assertEqual(play_1.pulse.amp, 0.2)
        self.assertEqual(play_1.pulse.sigma, 40)

    def test_call_subroutine_with_parametrized_duration(self):
        """Test call subroutine containing a parametrized duration."""
        dur = circuit.Parameter("dur")

        with pulse.build() as subroutine:
            pulse.play(pulse.Constant(dur, 0.1), pulse.DriveChannel(0))
            pulse.play(pulse.Constant(dur, 0.2), pulse.DriveChannel(0))

        with pulse.build() as main:
            pulse.call(subroutine)

        self.assertEqual(len(main.blocks), 1)
