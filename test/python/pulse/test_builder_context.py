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

import qiskit.circuit as circuit
import qiskit.compiler as compiler
import qiskit.pulse as pulse
import qiskit.pulse.macros as macros
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse import pulse_lib, instructions


class TestBuilderContext(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.configuration = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.inst_map = self.defaults.instruction_schedule_map


class TestContexts(TestBuilderContext):
    """test context builder contexts."""
    def test_align_sequential(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_sequential():
                pulse.delay(d0, 3)
                pulse.delay(d1, 5)
                pulse.delay(d0, 7)

        reference = pulse.Schedule()

        reference.insert(0, instructions.Delay(3, d0), mutate=True)
        reference.insert(8, instructions.Delay(7, d0), mutate=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), mutate=True)

        self.assertEqual(schedule, reference)

    def test_align_left(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
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
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
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
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
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

    def test_transpiler_settings(self):
        """Test that two cx gates are optimized away with higher optimization level"""
        twice_cx_qc = circuit.QuantumCircuit(2)
        twice_cx_qc.cx(0, 1)
        twice_cx_qc.cx(0, 1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.transpiler_settings(optimization_level=0):
                pulse.call_circuit(twice_cx_qc)
        self.assertNotEqual(len(schedule.instructions), 0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.transpiler_settings(optimization_level=3):
                pulse.call_circuit(twice_cx_qc)
        self.assertEqual(len(schedule.instructions), 0)

    def test_scheduler_settings(self):
        inst_map = pulse.InstructionScheduleMap()
        d0 = pulse.DriveChannel(0)
        test_x_sched = pulse.Schedule()
        test_x_sched += instructions.Delay(10, d0)
        inst_map.add('x', (0,), test_x_sched)

        x_qc = circuit.QuantumCircuit(2)
        x_qc.x(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.transpiler_settings(basis_gates=['x']):
                with pulse.circuit_scheduler_settings(inst_map=inst_map):
                    pulse.call_circuit(x_qc)

        self.assertEqual(schedule, test_x_sched)


class TestInstructions(TestBuilderContext):
    """test context builder instructions."""

    def test_delay(self):
        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.delay(d0, 10)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        self.assertEqual(schedule, reference)

    def test_play_parametric_pulse(self):
        d0 = pulse.DriveChannel(0)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_play_sample_pulse(self):
        d0 = pulse.DriveChannel(0)
        test_pulse = pulse_lib.SamplePulse([0.0, 0.0])

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_play_array_pulse(self):
        d0 = pulse.DriveChannel(0)
        test_array = np.array([0., 0.], dtype=np.complex_)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.play(d0, test_array)

        reference = pulse.Schedule()
        test_pulse = pulse.SamplePulse(test_array)
        reference += instructions.Play(test_pulse, d0)

        self.assertEqual(schedule, reference)

    def test_acquire_memory_slot(self):
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.acquire(acquire0, mem0, 10)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertEqual(schedule, reference)

    def test_acquire_register_slot(self):
        acquire0 = pulse.AcquireChannel(0)
        reg0 = pulse.RegisterSlot(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.acquire(acquire0, reg0, 10)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, reg_slot=reg0)

        self.assertEqual(schedule, reference)

    def test_set_frequency(self):
        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.set_frequency(d0, 1e9)

        reference = pulse.Schedule()
        reference += instructions.SetFrequency(1e9, d0)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_shift_frequency(self):
        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.shift_frequency(d0, 0.1e9)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(0.1e9, d0)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_set_phase(self):
        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.set_phase(d0, 3.14)

        reference = pulse.Schedule()
        reference += instructions.SetPhase(3.14, d0)

        self.assertEqual(schedule, reference)

    def test_shift_phase(self):
        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.shift_phase(d0, 3.14)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)

        self.assertEqual(schedule, reference)

    def test_snapshot(self):

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.snapshot('test', 'state')

        reference = pulse.Schedule()
        reference += instructions.Snapshot('test', 'state')

        self.assertEqual(schedule, reference)

    def test_call_schedule(self):
        schedule = pulse.Schedule()
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        reference = pulse.Schedule()
        reference = reference.insert(10, instructions.Delay(10, d0))
        reference += instructions.Delay(20, d1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_right():
                pulse.call_schedule(reference)

        self.assertEqual(schedule, reference)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_right():
                pulse.call(reference)

        self.assertEqual(schedule, reference)

    def test_call_circuit(self):
        inst_map = self.inst_map
        reference = inst_map.get('u1', (0,), 0.0)

        u1_qc = circuit.QuantumCircuit(2)
        u1_qc.u1(0.0, 0)

        transpiler_settings = {'optimization_level': 0}

        schedule = pulse.Schedule()
        with pulse.build(self.backend,
                         schedule,
                         transpiler_settings=transpiler_settings):
            with pulse.align_right():
                pulse.call_circuit(u1_qc)

        schedule = pulse.Schedule()
        with pulse.build(self.backend,
                         schedule,
                         transpiler_settings=transpiler_settings):
            with pulse.align_right():
                pulse.call(u1_qc)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_barrier(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_left():
                pulse.play(d0, test_pulse)
                pulse.barrier(d0, d1)
                pulse.play(d1, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)

        # test barrier on qubits
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_left():
                pulse.play(d0, test_pulse)
                pulse.barrier(0, 1)
                pulse.play(d1, test_pulse)

        self.assertEqual(schedule, reference)


class TestUtilities(TestBuilderContext):
    """test context builder utilities."""
    def test_current_backend(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            self.assertEqual(pulse.current_backend(), self.backend)

    def test_append_block(self):
        d0 = pulse.DriveChannel(0)
        block = pulse.Schedule()
        block += instructions.Delay(10, d0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.append_block(block)

        self.assertEqual(schedule, block)

    def test_append_instruction(self):
        d0 = pulse.DriveChannel(0)
        instruction = instructions.Delay(10, d0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.append_instruction(instruction)

        self.assertEqual(schedule, instruction)

    @unittest.expectedFailure
    def test_qubit_channels(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            qubit_channels = pulse.qubit_channels(0)

        self.assertEqual(qubit_channels, set())

    def test_current_transpiler_settings(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            self.assertFalse(pulse.current_transpiler_settings())
            with pulse.transpiler_settings(test_setting=1):
                self.assertEqual(
                    pulse.current_transpiler_settings()['test_setting'], 1)

    def test_current_circuit_scheduler_settings(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            self.assertFalse(pulse.current_circuit_scheduler_settings())
            with pulse.circuit_scheduler_settings(test_setting=1):
                self.assertEqual(
                    pulse.current_circuit_scheduler_settings()['test_setting'], 1)


class TestMacros(TestBuilderContext):
    """test context builder macros."""

    def test_measure(self):
        """Test utility function - measure."""
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.measure(0)

        reference = macros.measure(
            qubits=[0],
            inst_map=self.inst_map,
            meas_map=self.configuration.meas_map)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_delay_qubit(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.delay_qubit(0, 10)

        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        a0 = pulse.AcquireChannel(0)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(0)

        reference = pulse.Schedule()
        reference += instructions.Delay(d0)
        reference += instructions.Delay(m0)
        reference += instructions.Delay(a0)
        reference += instructions.Delay(u0)
        reference += instructions.Delay(u1)

        self.assertEqual(schedule, reference)


class TestGates(TestBuilderContext):
    """test context builder gates."""

    def test_cx(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference_qc.cx(0, 1)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u1(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.u1(0, np.pi)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u1(np.pi, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u2(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.u2(0, np.pi, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u2(np.pi, 0, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_u3(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.u3(0, np.pi, 0, np.pi/2)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.u3(np.pi, 0, np.pi/2, 0)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_x(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.x(0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.x(0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_lazy_evaluation_with_transpiler(self):
        """Test that the two cx gates are optimizied away by the transpiler."""
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.cx(0, 1)
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)

    def test_measure(self):
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.align_sequential():
                pulse.x(0)
                pulse.measure(0)

        reference_qc = circuit.QuantumCircuit(1, 1)
        reference_qc.x(0)
        reference_qc.measure(0, 0)
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertEqual(schedule, reference)


class TestBuilderComposition(TestBuilderContext):
    """Test more sophisticated composite builder examples."""

    def test_complex_build(self):
        """Test a general program build."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        delay_dur = 19
        short_dur = 31
        long_dur = 101
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
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
        # Insert so that the long pulse on d2 ocurrs as early as possible
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
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule,
                         alignment='left'):
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        reference = pulse.Schedule()
        with pulse.build(self.backend, reference):
            with pulse.align_left():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)

    def test_default_alignment_right(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule,
                         alignment='right'):
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        reference = pulse.Schedule()
        with pulse.build(self.backend, reference):
            with pulse.align_right():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)

    def test_default_alignment_sequential(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule,
                         alignment='sequential'):
            pulse.delay(d0, 10)
            pulse.delay(d1, 20)

        reference = pulse.Schedule()
        with pulse.build(self.backend, reference):
            with pulse.align_sequential():
                pulse.delay(d0, 10)
                pulse.delay(d1, 20)

        self.assertEqual(schedule, reference)
