# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assembler Test."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from qiskit import pulse
from qiskit.assembler.disassemble import disassemble
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.compiler.assembler import assemble
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
import qiskit.quantum_info as qi


class TestQuantumCircuitDisassembler(QiskitTestCase):
    """Tests for disassembling circuits to qobj."""

    def test_disassemble_single_circuit(self):
        """Test disassembling a single circuit."""
        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(2, name="c")
        circ = QuantumCircuit(qr, cr, name="circ")
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)

        qubit_lo_freq = [5e9, 5e9]
        meas_lo_freq = [6.7e9, 6.7e9]
        qobj = assemble(
            circ,
            shots=2000,
            memory=True,
            qubit_lo_freq=qubit_lo_freq,
            meas_lo_freq=meas_lo_freq,
        )
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, True)
        self.assertEqual(run_config_out.qubit_lo_freq, qubit_lo_freq)
        self.assertEqual(run_config_out.meas_lo_freq, meas_lo_freq)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disassemble_multiple_circuits(self):
        """Test disassembling multiple circuits, all should have the same config."""
        qr0 = QuantumRegister(2, name="q0")
        qc0 = ClassicalRegister(2, name="c0")
        circ0 = QuantumCircuit(qr0, qc0, name="circ0")
        circ0.h(qr0[0])
        circ0.cx(qr0[0], qr0[1])
        circ0.measure(qr0, qc0)

        qr1 = QuantumRegister(3, name="q1")
        qc1 = ClassicalRegister(3, name="c1")
        circ1 = QuantumCircuit(qr1, qc1, name="circ0")
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr1[1])
        circ1.cx(qr1[0], qr1[2])
        circ1.measure(qr1, qc1)

        qobj = assemble([circ0, circ1], shots=100, memory=False, seed=6)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(run_config_out.shots, 100)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.seed, 6)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertIn(circuit, [circ0, circ1])
        self.assertEqual({}, headers)

    def test_disassemble_no_run_config(self):
        """Test disassembling with no run_config, relying on default."""
        qr = QuantumRegister(2, name="q")
        qc = ClassicalRegister(2, name="c")
        circ = QuantumCircuit(qr, qc, name="circ")
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        qobj = assemble(circ)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disassemble_initialize(self):
        """Test disassembling a circuit with an initialize."""
        q = QuantumRegister(2, name="q")
        circ = QuantumCircuit(q, name="circ")
        circ.initialize([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], q[:])

        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_disassemble_isometry(self):
        """Test disassembling a circuit with an isometry."""
        q = QuantumRegister(2, name="q")
        circ = QuantumCircuit(q, name="circ")
        circ.iso(qi.random_unitary(4).data, circ.qubits, [])
        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        # params array
        assert_allclose(circuits[0]._data[0][0].params[0], circ._data[0][0].params[0])
        # all other data
        self.assertEqual(circuits[0]._data[0][0].params[1:], circ._data[0][0].params[1:])
        self.assertEqual(circuits[0]._data[0][1:], circ._data[0][1:])
        self.assertEqual(circuits[0]._data[1:], circ._data[1:])
        self.assertEqual({}, header)

    def test_opaque_instruction(self):
        """Test the disassembler handles opaque instructions correctly."""
        opaque_inst = Instruction(name="my_inst", num_qubits=4, num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name="q")
        c = ClassicalRegister(4, name="c")
        circ = QuantumCircuit(q, c, name="circ")
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])
        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 6)
        self.assertEqual(run_config_out.memory_slots, 4)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_circuit_with_conditionals(self):
        """Verify disassemble sets conditionals correctly."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr1, cr2)
        qc.measure(qr[0], cr1)  # Measure not required for a later conditional
        qc.measure(qr[1], cr2[1])  # Measure required for a later conditional
        qc.h(qr[1]).c_if(cr2, 3)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_simple_conditional(self):
        """Verify disassemble handles a simple conditional on the only bits."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0]).c_if(cr, 1)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 1)
        self.assertEqual(run_config_out.memory_slots, 1)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_mcx(self):
        """Verify disassemble handles mcx gate - #6271."""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.mcx([0, 1, 2], 4)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 5)
        self.assertEqual(run_config_out.memory_slots, 5)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_multiple_conditionals_multiple_registers(self):
        """Verify disassemble handles multiple conditionals and registers."""
        qr = QuantumRegister(3)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(5)
        cr3 = ClassicalRegister(6)
        cr4 = ClassicalRegister(1)

        qc = QuantumCircuit(qr, cr1, cr2, cr3, cr4)
        qc.x(qr[1])
        qc.h(qr)
        qc.cx(qr[1], qr[0]).c_if(cr3, 14)
        qc.ccx(qr[0], qr[2], qr[1]).c_if(cr4, 1)
        qc.h(qr).c_if(cr1, 3)
        qobj = assemble(qc)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 15)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)


class TestPulseScheduleDisassembler(QiskitTestCase):
    """Tests for disassembling pulse schedules to qobj."""

    def setUp(self):
        super().setUp()
        self.backend = FakeOpenPulse2Q()
        self.backend_config = self.backend.configuration()
        self.backend_config.parametric_pulses = ["constant", "gaussian", "gaussian_square", "drag"]

    def test_disassemble_single_schedule(self):
        """Test disassembling a single schedule."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        with pulse.build(self.backend) as sched:
            with pulse.align_right():
                pulse.play(pulse.library.Constant(10, 1.0), d0)
                pulse.set_phase(1.0, d0)
                pulse.shift_phase(3.11, d0)
                pulse.set_frequency(1e9, d0)
                pulse.shift_frequency(1e7, d0)
                pulse.delay(20, d0)
                pulse.delay(10, d1)
                pulse.play(pulse.library.Constant(8, 0.1), d1)
                pulse.measure_all()

        qobj = assemble(sched, backend=self.backend, shots=2000)
        scheds, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.meas_level, 2)
        self.assertEqual(run_config_out.meas_lo_freq, self.backend.defaults().meas_freq_est)
        self.assertEqual(run_config_out.qubit_lo_freq, self.backend.defaults().qubit_freq_est)
        self.assertEqual(run_config_out.rep_time, 99)
        self.assertEqual(len(scheds), 1)
        self.assertEqual(scheds[0], target_qobj_transform(sched))

    def test_disassemble_multiple_schedules(self):
        """Test disassembling multiple schedules, all should have the same config."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        with pulse.build(self.backend) as sched0:
            with pulse.align_right():
                pulse.play(pulse.library.Constant(10, 1.0), d0)
                pulse.set_phase(1.0, d0)
                pulse.shift_phase(3.11, d0)
                pulse.set_frequency(1e9, d0)
                pulse.shift_frequency(1e7, d0)
                pulse.delay(20, d0)
                pulse.delay(10, d1)
                pulse.play(pulse.library.Constant(8, 0.1), d1)
                pulse.measure_all()

        with pulse.build(self.backend) as sched1:
            with pulse.align_right():
                pulse.play(pulse.library.Constant(8, 0.1), d0)
                pulse.play(pulse.library.Waveform([0.0, 1.0]), d1)
                pulse.set_phase(1.1, d0)
                pulse.shift_phase(3.5, d0)
                pulse.set_frequency(2e9, d0)
                pulse.shift_frequency(3e7, d1)
                pulse.delay(20, d1)
                pulse.delay(10, d0)
                pulse.play(pulse.library.Constant(8, 0.4), d1)
                pulse.measure_all()

        qobj = assemble([sched0, sched1], backend=self.backend, shots=2000)
        scheds, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(len(scheds), 2)
        self.assertEqual(scheds[0], target_qobj_transform(sched0))
        self.assertEqual(scheds[1], target_qobj_transform(sched1))

    def test_disassemble_parametric_pulses(self):
        """Test disassembling multiple schedules all should have the same config."""
        d0 = pulse.DriveChannel(0)
        with pulse.build(self.backend) as sched:
            with pulse.align_right():
                pulse.play(pulse.library.Constant(10, 1.0), d0)
                pulse.play(pulse.library.Gaussian(10, 1.0, 2.0), d0)
                pulse.play(pulse.library.GaussianSquare(10, 1.0, 2.0, 3), d0)
                pulse.play(pulse.library.Drag(10, 1.0, 2.0, 0.1), d0)

        qobj = assemble(sched, backend=self.backend, shots=2000)
        scheds, _, _ = disassemble(qobj)
        self.assertEqual(scheds[0], target_qobj_transform(sched))

    def test_disassemble_schedule_los(self):
        """Test disassembling schedule los."""
        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        d1 = pulse.DriveChannel(1)
        m1 = pulse.MeasureChannel(1)

        sched0 = pulse.Schedule()
        sched1 = pulse.Schedule()

        schedule_los = [
            {d0: 4.5e9, d1: 5e9, m0: 6e9, m1: 7e9},
            {d0: 5e9, d1: 4.5e9, m0: 7e9, m1: 6e9},
        ]
        qobj = assemble([sched0, sched1], backend=self.backend, schedule_los=schedule_los)
        _, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)

        self.assertEqual(run_config_out.schedule_los, schedule_los)


if __name__ == "__main__":
    unittest.main(verbosity=2)
