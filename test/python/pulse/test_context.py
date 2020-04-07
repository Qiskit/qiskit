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
import qiskit.pulse as pulse
from qiskit.pulse.pulse_lib import gaussian
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse import pulse_lib, instructions


class TestBuilderContext(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()

    def test_context(self):
        """Test a general program build."""
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.delay(0, 1000)
            pulse.u2(0, 0, pi/2)
            pulse.delay(0, 1000)
            pulse.u2(0, 0, pi)
            with pulse.left_align():
                pulse.play(pulse.DriveChannel(0), gaussian(500, 0.1, 125))
                pulse.shift_phase(pulse.DriveChannel(0), pi/2)
                pulse.play(pulse.DriveChannel(0), gaussian(500, 0.1, 125))
                pulse.u2(1, 0, pi/2)
            with pulse.sequential():
                pulse.u2(0, 0, pi/2)
                pulse.u2(1, 0, pi/2)
                pulse.u2(0, 0, pi/2)
            pulse.measure(0)


class TestContexts(TestBuilderContext):
    """Test builder contexts."""
    def test_parallel(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.parallel():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference += instructions.Play(test_pulse, d1)

        self.assertEqual(schedule, reference)

    def test_sequential(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.sequential():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Play(test_pulse, d0), mutate=True)
        reference.insert(20, instructions.Play(test_pulse, d0), mutate=True)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)

    def test_left_align(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.left_align():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference += instructions.Play(test_pulse, d1)

        self.assertEqual(schedule, reference)

    def test_right_align(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)

    def test_group(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.play(d0, test_pulse)
            with pulse.group():
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference = reference.insert(10, instructions.Play(test_pulse, d1))
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
    """Test builder instructions."""

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
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(20, d1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.call_schedule(reference)

        self.assertEqual(schedule, reference)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.call(reference)

        self.assertEqual(schedule, reference)

    def test_call_circuit(self):
        inst_map = self.backend.defaults().instruction_schedule_map
        reference = inst_map.get('cx', (0, 1))

        cx_qc = circuit.QuantumCircuit(2)
        cx_qc.cx(0, 1)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.call_circuit(cx_qc)

        self.assertEqual(schedule, reference)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.call(cx_qc)

        self.assertEqual(schedule, reference)

    @unittest.expectedFailure
    def test_barrier(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.left_align():
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
            with pulse.left_align():
                pulse.play(d0, test_pulse)
                pulse.barrier(0, 1)
                pulse.play(d1, test_pulse)

        self.assertEqual(schedule, reference)


class TestUtilities(TestBuilderContext):
    """Test builder utilities."""
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
    """Test builder macros."""


class TestGates(TestBuilderContext):
    """Test builder gates."""
