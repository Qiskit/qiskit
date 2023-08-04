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

from qiskit import circuit, compiler, pulse
from qiskit.pulse import builder, macros

from qiskit.pulse import library, instructions
from qiskit.pulse.instructions import directives
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.providers.fake_provider import FakeMumbaiV2
from qiskit.test import QiskitTestCase


class TestBuilderV2(QiskitTestCase):
    """Test the pulse builder context with backendV2."""

    def setUp(self):
        super().setUp()
        self.backend = FakeMumbaiV2()

    def assertScheduleEqual(self, program, target):
        """Assert an error when two pulse programs are not equal.

        .. note:: Two programs are converted into standard execution format then compared.
        """
        self.assertEqual(target_qobj_transform(program), target_qobj_transform(target))


class TestContextsV2(TestBuilderV2):
    """Test builder contexts."""

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


class TestDirectivesV2(TestBuilderV2):
    """Test builder directives."""

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


class TestGatesV2(TestBuilderV2):
    """Test builder gates."""

    def test_cx(self):
        """Test cx gate."""
        with pulse.build(self.backend) as schedule:
            pulse.cx(0, 1)

        reference_qc = circuit.QuantumCircuit(2)
        reference_qc.cx(0, 1)
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

    def test_u1(self):
        """Test u1 gate."""
        with pulse.build(self.backend) as schedule:
            with pulse.transpiler_settings(layout_method="trivial"):
                pulse.u1(np.pi / 2, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U1Gate(np.pi / 2), [0])
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_u2(self):
        """Test u2 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u2(np.pi / 2, 0, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U2Gate(np.pi / 2, 0), [0])
        reference_qc = compiler.transpile(reference_qc, self.backend)
        reference = compiler.schedule(reference_qc, self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_u3(self):
        """Test u3 gate."""
        with pulse.build(self.backend) as schedule:
            pulse.u3(np.pi / 8, np.pi / 16, np.pi / 4, 0)

        reference_qc = circuit.QuantumCircuit(1)
        reference_qc.append(circuit.library.U3Gate(np.pi / 8, np.pi / 16, np.pi / 4), [0])
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


class TestBuilderCompositionV2(TestBuilderV2):
    """Test more sophisticated composite builder examples."""

    def test_complex_build(self):
        """Test a general program build with nested contexts,
        circuits and macros."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        delay_dur = 300
        short_dur = 200
        long_dur = 810

        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.delay(delay_dur, d0)
                pulse.u2(0, np.pi / 2, 1)
            with pulse.align_right():
                pulse.play(library.Constant(short_dur, 0.1), d1)
                pulse.play(library.Constant(long_dur, 0.1), d2)
                pulse.u2(0, np.pi / 2, 1)
            with pulse.align_left():
                pulse.u2(0, np.pi / 2, 0)
                pulse.u2(0, np.pi / 2, 1)
                pulse.u2(0, np.pi / 2, 0)
            pulse.measure(0)
        # prepare and schedule circuits that will be used.
        single_u2_qc = circuit.QuantumCircuit(2)
        single_u2_qc.append(circuit.library.U2Gate(0, np.pi / 2), [1])
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
        triple_u2_qc.append(circuit.library.U2Gate(0, np.pi / 2), [0])
        triple_u2_qc.append(circuit.library.U2Gate(0, np.pi / 2), [1])
        triple_u2_qc.append(circuit.library.U2Gate(0, np.pi / 2), [0])
        triple_u2_qc = compiler.transpile(triple_u2_qc, self.backend)
        align_left_reference = compiler.schedule(triple_u2_qc, self.backend, method="alap")

        # measurement
        measure_reference = macros.measure(
            qubits=[0], backend=self.backend, meas_map=self.backend.meas_map
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


class TestSubroutineCallV2(TestBuilderV2):
    """Test for calling subroutine."""

    def test_call_circuit(self):
        """Test calling circuit instruction."""
        inst_map = self.backend.target.instruction_schedule_map()
        reference = inst_map.get("x", (0,))

        ref_sched = pulse.Schedule()
        ref_sched += pulse.instructions.Call(reference)

        u1_qc = circuit.QuantumCircuit(2)
        u1_qc.append(circuit.library.XGate(), [0])

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
        measure_reference = macros.measure(qubits=[0, 1], backend=self.backend)

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
