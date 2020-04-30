# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
import unittest

import numpy as np

from qiskit import circuit
from qiskit import compiler
from qiskit import pulse
from qiskit.pulse import builder
from qiskit.pulse import macros
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse import pulse_lib, instructions

# pylint: disable=invalid-name


class TestBuilder(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.configuration = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.inst_map = self.defaults.instruction_schedule_map


class TestContexts(TestBuilder):
    """test context builder contexts."""
    def test_align_sequential(self):
        """Test the sequential alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.delay(d0, 3)
                pulse.delay(d1, 5)
                pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), mutate=True)
        reference.insert(8, instructions.Delay(7, d0), mutate=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), mutate=True)

        self.assertEqual(schedule, reference)

    def test_align_left(self):
        """Test the left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(d2, 11)
                pulse.delay(d0, 3)
                with pulse.align_left():
                    pulse.delay(d1, 5)
                    pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        reference += instructions.Delay(7, d0)
        # d1
        reference = reference.insert(3, instructions.Delay(5, d1))
        # d2
        reference += instructions.Delay(11, d2)
        self.assertEqual(schedule, reference)

    def test_align_right(self):
        """Test the right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.delay(d2, 11)
                pulse.delay(d0, 3)
                with pulse.align_right():
                    pulse.delay(d1, 5)
                    pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference = reference.insert(1, instructions.Delay(3, d0))
        reference = reference.insert(4, instructions.Delay(7, d0))
        # d1
        reference = reference.insert(6, instructions.Delay(5, d1))
        # d2
        reference += instructions.Delay(11, d2)

        self.assertEqual(schedule, reference)

    def test_group(self):
        """Test the grouping context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            pulse.delay(d0, 3)
            with pulse.group():
                pulse.delay(d1, 5)
                pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        reference += instructions.Delay(7, d0)
        # d1
        reference = reference.insert(3, instructions.Delay(5, d1))
        self.assertEqual(schedule, reference)

    def test_inline(self):
        """Test the inlining context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            pulse.delay(d0, 3)
            with pulse.inline():
                pulse.delay(d1, 5)
                pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        reference += instructions.Delay(7, d0)
        # d1
        reference += instructions.Delay(5, d1)

        self.assertEqual(schedule, reference)

    def test_transpiler_settings(self):
        """Test the transpiler settings context.

        Tests that two cx gates are optimized away with higher optimization level.
        """
        twice_cx_qc = circuit.QuantumCircuit(2)
        twice_cx_qc.cx(0, 1)
        twice_cx_qc.cx(0, 1)

        with pulse.build(self.backend) as schedule:
            with pulse.transpiler_settings(optimization_level=0):
                pulse.call_circuit(twice_cx_qc)
        self.assertNotEqual(len(schedule.instructions), 0)

        with pulse.build(self.backend) as schedule:
            with pulse.transpiler_settings(optimization_level=3):
                pulse.call_circuit(twice_cx_qc)
        self.assertEqual(len(schedule.instructions), 0)

    def test_scheduler_settings(self):
        """Test the circuit scheduler settings context."""
        inst_map = pulse.InstructionScheduleMap()
        d0 = pulse.DriveChannel(0)
        test_x_sched = pulse.Schedule()
        test_x_sched += instructions.Delay(10, d0)
        inst_map.add('x', (0,), test_x_sched)

        x_qc = circuit.QuantumCircuit(2)
        x_qc.x(0)

        with pulse.build(backend=self.backend) as schedule:
            with pulse.transpiler_settings(basis_gates=['x']):
                with pulse.circuit_scheduler_settings(inst_map=inst_map):
                    pulse.call_circuit(x_qc)

        self.assertEqual(schedule, test_x_sched)

    def test_phase_offset(self):
        """Test the phase offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.phase_offset(d0, 3.14):
                pulse.delay(d0, 10)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(-3.14, d0)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_frequency_offset(self):
        """Test the frequency offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.frequency_offset(d0, 1e9):
                pulse.delay(d0, 10)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)  # pylint: disable=no-member
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftFrequency(-1e9, d0)  # pylint: disable=no-member

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_phase_compensated_frequency_offset(self):
        """Test that the phase offset context properly compensates for phase accumulation."""
        d0 = pulse.DriveChannel(0)

        with pulse.build(self.backend) as schedule:
            with pulse.frequency_offset(d0, 1e9, compensate_phase=True):
                pulse.delay(d0, 10)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)  # pylint: disable=no-member
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(-1e9*10/self.configuration.dt, d0)
        reference += instructions.ShiftFrequency(-1e9, d0)  # pylint: disable=no-member

        self.assertEqual(schedule, reference)


class TestTypes(TestBuilder):
    """test context builder types."""

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
            self.assertEqual(pulse.control_channel((0, 1))[0],
                             pulse.ControlChannel(0))


class TestInstructions(TestBuilder):
    """test context builder instructions."""

    def test_delay(self):
        """Test delay instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.delay(d0, 10)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        self.assertEqual(schedule, reference)

    def test_play_parametric_pulse(self):
        """Test play instruction with parametric pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        with pulse.build() as schedule:
            pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_play_sample_pulse(self):
        """Test play instruction with sample pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = pulse_lib.SamplePulse([0.0, 0.0])

        with pulse.build() as schedule:
            pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_play_array_pulse(self):
        """Test play instruction on an array directly."""
        d0 = pulse.DriveChannel(0)
        test_array = np.array([0., 0.], dtype=np.complex_)

        with pulse.build() as schedule:
            pulse.play(d0, test_array)

        reference = pulse.Schedule()
        test_pulse = pulse.SamplePulse(test_array)
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_acquire_memory_slot(self):
        """Test acquire instruction into memory slot."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(acquire0, mem0, 10)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertEqual(schedule, reference)

    def test_acquire_register_slot(self):
        """Test acquire instruction into register slot."""
        acquire0 = pulse.AcquireChannel(0)
        reg0 = pulse.RegisterSlot(0)

        with pulse.build() as schedule:
            pulse.acquire(acquire0, reg0, 10)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, reg_slot=reg0)

        self.assertEqual(schedule, reference)

    def test_acquire_qubit(self):
        """Test acquire instruction on qubit."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(0, mem0, 10)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertEqual(schedule, reference)

    def test_set_frequency(self):
        """Test set frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_frequency(d0, 1e9)

        reference = pulse.Schedule()
        reference += instructions.SetFrequency(1e9, d0)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_shift_frequency(self):  # pylint: disable=no-member
        """Test shift frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_frequency(d0, 0.1e9)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(0.1e9, d0)  # pylint: disable=no-member

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_set_phase(self):  # pylint: disable=no-member
        """Test set phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_phase(d0, 3.14)

        reference = pulse.Schedule()
        reference += instructions.SetPhase(3.14, d0)  # pylint: disable=no-member

        self.assertEqual(schedule, reference)

    def test_shift_phase(self):
        """Test shift phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_phase(d0, 3.14)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)

        self.assertEqual(schedule, reference)

    def test_snapshot(self):
        """Test snapshot instruction."""
        with pulse.build() as schedule:
            pulse.snapshot('test', 'state')

        reference = pulse.Schedule()
        reference += instructions.Snapshot('test', 'state')

        self.assertEqual(schedule, reference)

    def test_call_schedule(self):
        """Test calling schedule instruction."""
        schedule = pulse.Schedule()
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        reference = pulse.Schedule()
        reference = reference.insert(10, instructions.Delay(10, d0))
        reference += instructions.Delay(20, d1)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.call_schedule(reference)

        self.assertEqual(schedule, reference)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.call(reference)

        self.assertEqual(schedule, reference)

    def test_call_circuit(self):
        """Test calling circuit instruction."""
        inst_map = self.inst_map
        reference = inst_map.get('u1', (0,), 0.0)

        u1_qc = circuit.QuantumCircuit(2)
        u1_qc.u1(0.0, 0)

        transpiler_settings = {'optimization_level': 0}

        with pulse.build(self.backend,
                         default_transpiler_settings=transpiler_settings
                         ) as schedule:
            with pulse.align_right():
                pulse.call_circuit(u1_qc)

        self.assertEqual(schedule, reference)


class TestDirectives(TestBuilder):
    """Test context builder directives."""

    def test_barrier_with_align_right(self):
        """Test barrier directive with right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.delay(d0, 3)
                pulse.barrier(d0, d1, d2)
                pulse.delay(d2, 11)
                with pulse.align_right():
                    pulse.delay(d1, 5)
                    pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference = reference.insert(0, instructions.Delay(3, d0))
        reference = reference.insert(7, instructions.Delay(7, d0))
        # d1
        reference = reference.insert(9, instructions.Delay(5, d1))
        # d2
        reference = reference.insert(3, instructions.Delay(11, d2))

        self.assertEqual(schedule, reference)

    def test_barrier_with_align_left(self):
        """Test barrier directive with left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(d0, 3)
                pulse.barrier(d0, d1, d2)
                pulse.delay(d2, 11)
                with pulse.align_left():
                    pulse.delay(d1, 5)
                    pulse.delay(d0, 7)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        reference += instructions.Delay(7, d0)
        # d1
        reference = reference.insert(3, instructions.Delay(5, d1))
        # d2
        reference = reference.insert(3, instructions.Delay(11, d2))

        self.assertEqual(schedule, reference)


class TestUtilities(TestBuilder):
    """test context builder utilities."""
    def test_active_backend(self):
        """Test getting active builder backend."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.active_backend(), self.backend)

    def test_append_block(self):
        """Test appending a block to the active builder."""
        d0 = pulse.DriveChannel(0)
        block = pulse.Schedule()
        block += instructions.Delay(10, d0)

        schedule = pulse.Schedule()
        with pulse.build() as schedule:
            pulse.append_block(block)

        self.assertEqual(schedule, block)

    def test_append_instruction(self):
        """Test appending an instruction to the active builder."""
        d0 = pulse.DriveChannel(0)
        instruction = instructions.Delay(10, d0)

        with pulse.build() as schedule:
            pulse.append_instruction(instruction)

        self.assertEqual(schedule, instruction)

    def test_qubit_channels(self):
        """Test getting the qubit channels of the active builder's backend."""
        with pulse.build(self.backend):
            qubit_channels = pulse.qubit_channels(0)

        self.assertEqual(qubit_channels,
                         {pulse.DriveChannel(0),
                          pulse.MeasureChannel(0),
                          pulse.AcquireChannel(0),
                          pulse.ControlChannel(0),
                          pulse.ControlChannel(1)})

    def test_active_transpiler_settings(self):
        """Test setting settings of active builder's transpiler."""
        with pulse.build(self.backend):
            self.assertFalse(pulse.active_transpiler_settings())
            with pulse.transpiler_settings(test_setting=1):
                self.assertEqual(
                    pulse.active_transpiler_settings()['test_setting'], 1)

    def test_active_circuit_scheduler_settings(self):
        """Test setting settings of active builder's circuit scheduler."""
        with pulse.build(self.backend):
            self.assertFalse(pulse.active_circuit_scheduler_settings())
            with pulse.circuit_scheduler_settings(test_setting=1):
                self.assertEqual(
                    pulse.active_circuit_scheduler_settings()['test_setting'], 1)

    def test_num_qubits(self):
        """Test builder utility to get number of qubits."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.num_qubits(), 2)


class TestMacros(TestBuilder):
    """test context builder macros."""

    def test_measure(self):
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            reg = pulse.measure(0)

        self.assertEqual(reg, pulse.MemorySlot(0))

        reference = macros.measure(
            qubits=[0],
            inst_map=self.inst_map,
            meas_map=self.configuration.meas_map)

        self.assertEqual(schedule, reference)

    def test_measure_all(self):
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure_all()

        self.assertEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])
        reference = macros.measure_all(self.backend)

        self.assertEqual(schedule, reference)

    def test_delay_qubit(self):
        """Test delaying on a qubit macro."""
        with pulse.build(self.backend) as schedule:
            pulse.delay_qubits(0, 10)

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

        self.assertEqual(schedule, reference)

    def test_delay_qubits(self):
        """Test delaying on multiple qubits to make sure we don't insert delays twice."""
        with pulse.build(self.backend) as schedule:
            pulse.delay_qubits([0, 1], 10)

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

        self.assertEqual(schedule, reference)


class TestGates(TestBuilder):
    """test context builder gates."""

    def test_cx(self):
        """Test cx gate."""
        with pulse.build(self.backend) as schedule:
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference_qc.cx(0, 1)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u1(self):
        """Test u1 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u1(0, np.pi)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u1(np.pi, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u2(self):
        """Test u2 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u2(0, np.pi, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u2(np.pi, 0, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u3(self):
        """Test u3 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u3(0, np.pi, 0, np.pi/2)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u3(np.pi, 0, np.pi/2, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_x(self):

        """Test x gate."""
        with pulse.build(self.backend) as schedule:
            pulse.x(0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.x(0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_lazy_evaluation_with_transpiler(self):
        """Test that the two cx gates are optimizied away by the transpiler."""
        with pulse.build(self.backend) as schedule:
            pulse.cx(0, 1)
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_measure(self):
        """Test pulse measurement macro against circuit measurement and ensure agreement."""
        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.x(0)
                pulse.measure(0)

        reference_qc = circuit.QuantumCircuit(1, 1)
        reference_qc.x(0)
        reference_qc.measure(0, 0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_backend_require(self):
        """Test that a backend is required to use a gate."""
        with self.assertRaises(builder.BackendNotSet):
            with pulse.build():
                pulse.x(0)


class TestBuilderComposition(TestBuilder):
    """Test more sophisticated composite builder examples."""

    def test_complex_build(self):
        """Test a general program build with nested contexts, circuits and macros."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        delay_dur = 19
        short_dur = 31
        long_dur = 101

        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.delay(d0, delay_dur)
                pulse.u2(1, 0, pi/2)
            with pulse.align_right():
                pulse.play(d1, pulse_lib.ConstantPulse(short_dur, 0.1))
                pulse.play(d2, pulse_lib.ConstantPulse(long_dur, 0.1))
                pulse.u2(1, 0, pi/2)
            with pulse.align_left():
                pulse.u2(0, 0, pi/2)
                pulse.u2(1, 0, pi/2)
                pulse.u2(0, 0, pi/2)
            pulse.measure(0)

        single_u2_qc = circuit.QuantumCircuit(2)
        single_u2_qc.u2(0, pi/2, 1)
        single_u2_qc = compiler.transpile(single_u2_qc, self.backend)
        single_u2_sched = compiler.schedule(single_u2_qc, self.backend)

        triple_u2_qc = circuit.QuantumCircuit(2)
        triple_u2_qc.u2(0, pi/2, 0)
        triple_u2_qc.u2(0, pi/2, 1)
        triple_u2_qc.u2(0, pi/2, 0)
        triple_u2_qc = compiler.transpile(triple_u2_qc, self.backend)
        triple_u2_sched = compiler.schedule(triple_u2_qc, self.backend, method='alap')

        # outer sequential context
        outer_reference = pulse.Schedule()
        outer_reference += instructions.Delay(delay_dur, d0)
        outer_reference.insert(delay_dur, single_u2_sched, mutate=True)

        # inner align right
        align_right_reference = pulse.Schedule()
        align_right_reference += pulse.Play(pulse_lib.ConstantPulse(long_dur, 0.1), d2)
        align_right_reference.insert(long_dur-single_u2_sched.duration,
                                     single_u2_sched,
                                     mutate=True)
        align_right_reference.insert(long_dur-single_u2_sched.duration-short_dur,
                                     pulse.Play(pulse_lib.ConstantPulse(short_dur, 0.1), d1),
                                     mutate=True)
        # inner align left
        align_left_reference = triple_u2_sched

        # measurement
        measure_reference = macros.measure(qubits=[0],
                                           inst_map=self.inst_map,
                                           meas_map=self.configuration.meas_map)
        reference = pulse.Schedule()
        reference += outer_reference
        # Insert so that the long pulse on d2 occurs as early as possible
        # without an overval on d1.
        insert_time = (reference.ch_stop_time(d1) -
                       align_right_reference.ch_start_time(d1))
        reference.insert(insert_time,
                         align_right_reference,
                         mutate=True)
        reference.insert(reference.ch_stop_time(d0, d1),
                         align_left_reference,
                         mutate=True)
        reference += measure_reference
        self.assertEqual(schedule, reference)

    def test_default_alignment_left(self):
        """Test default left alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment='left') as schedule:
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        with pulse.build(self.backend) as reference:
            with pulse.align_left():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)

    def test_default_alignment_right(self):
        """Test default right alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment='right') as schedule:
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        with pulse.build() as reference:
            with pulse.align_right():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)

    def test_default_alignment_sequential(self):
        """Test default sequential alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment='sequential') as schedule:
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        reference = pulse.Schedule()
        with pulse.build(reference) as reference:
            with pulse.align_sequential():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)

    def test_schedule_supplied(self):
        """Test that schedule is used if it is supplied to the builder."""
        d0 = pulse.DriveChannel(0)
        reference = pulse.Schedule()
        with pulse.build(reference) as reference:
            with pulse.align_sequential():
                pulse.delay(d0, 10)

        with pulse.build(schedule=reference) as schedule:
            pass

        self.assertEqual(schedule, reference)
