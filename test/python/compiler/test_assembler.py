# -*- coding: utf-8 -*-

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

import qiskit.pulse as pulse
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler.assemble import assemble
from qiskit.exceptions import QiskitError
from qiskit.qobj import QasmQobj
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class TestCircuitAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def test_assemble_single_circuit(self):
        """Test assembling a single circuit.
        """
        qr = QuantumRegister(2, name='q')
        cr = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, cr, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)

        qobj = assemble(circ, shots=2000, memory=True)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 2000)
        self.assertEqual(qobj.config.memory, True)
        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(qobj.experiments[0].instructions[1].name, 'cx')

    def test_assemble_multiple_circuits(self):
        """Test assembling multiple circuits, all should have the same config.
        """
        qr0 = QuantumRegister(2, name='q0')
        qc0 = ClassicalRegister(2, name='c0')
        circ0 = QuantumCircuit(qr0, qc0, name='circ0')
        circ0.h(qr0[0])
        circ0.cx(qr0[0], qr0[1])
        circ0.measure(qr0, qc0)

        qr1 = QuantumRegister(3, name='q1')
        qc1 = ClassicalRegister(3, name='c1')
        circ1 = QuantumCircuit(qr1, qc1, name='circ0')
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr1[1])
        circ1.cx(qr1[0], qr1[2])
        circ1.measure(qr1, qc1)

        qobj = assemble([circ0, circ1], shots=100, memory=False, seed_simulator=6)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.seed_simulator, 6)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(qobj.experiments[1].config.n_qubits, 3)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(len(qobj.experiments[1].instructions), 6)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default.
        """
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        qobj = assemble(circ)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 1024)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble(circ)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'initialize')
        np.testing.assert_almost_equal(qobj.experiments[0].instructions[0].params,
                                       [0.7071067811865, 0, 0, 0.707106781186])

    def test_assemble_opaque_inst(self):
        """Test opaque instruction is assembled as-is"""
        opaque_inst = Instruction(name='my_inst', num_qubits=4,
                                  num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name='q')
        c = ClassicalRegister(4, name='c')
        circ = QuantumCircuit(q, c, name='circ')
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])
        qobj = assemble(circ)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(len(qobj.experiments[0].instructions), 1)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'my_inst')
        self.assertEqual(qobj.experiments[0].instructions[0].qubits, [0, 2, 5, 3])
        self.assertEqual(qobj.experiments[0].instructions[0].memory, [3, 0])
        self.assertEqual(qobj.experiments[0].instructions[0].params, [0.5, 0.4])

    def test_measure_to_registers_when_conditionals(self):
        """Verify assemble_circuits maps all measure ops on to a register slot
        for a circuit containing conditionals."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr1, cr2)

        qc.measure(qr[0], cr1)  # Measure not required for a later conditional
        qc.measure(qr[1], cr2[1])  # Measure required for a later conditional
        qc.h(qr[1]).c_if(cr2, 3)

        qobj = assemble(qc)

        first_measure, second_measure = [op for op in qobj.experiments[0].instructions
                                         if op.name == 'measure']

        self.assertTrue(hasattr(first_measure, 'register'))
        self.assertEqual(first_measure.register, first_measure.memory)
        self.assertTrue(hasattr(second_measure, 'register'))
        self.assertEqual(second_measure.register, second_measure.memory)

    def test_convert_to_bfunc_plus_conditional(self):
        """Verify assemble_circuits converts conditionals from QASM to Qobj."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[0]).c_if(cr, 1)

        qobj = assemble(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, 'bfunc')
        self.assertEqual(bfunc_op.mask, '0x1')
        self.assertEqual(bfunc_op.val, '0x1')
        self.assertEqual(bfunc_op.relation, '==')

        self.assertTrue(hasattr(h_op, 'conditional'))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_resize_value_to_register(self):
        """Verify assemble_circuits converts the value provided on the classical
        creg to its mapped location on the device register."""
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr1, cr2, cr3)

        qc.h(qr[0]).c_if(cr2, 2)

        qobj = assemble(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, 'bfunc')
        self.assertEqual(bfunc_op.mask, '0xC')
        self.assertEqual(bfunc_op.val, '0x8')
        self.assertEqual(bfunc_op.relation, '==')

        self.assertTrue(hasattr(h_op, 'conditional'))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_assemble_circuits_raises_for_bind_circuit_mismatch(self):
        """Verify assemble_circuits raises error for parameterized circuits without matching
        binds."""
        qr = QuantumRegister(2)
        x = Parameter('x')
        y = Parameter('y')

        full_bound_circ = QuantumCircuit(qr)
        full_param_circ = QuantumCircuit(qr)
        partial_param_circ = QuantumCircuit(qr)

        partial_param_circ.u1(x, qr[0])

        full_param_circ.u1(x, qr[0])
        full_param_circ.u1(y, qr[1])

        partial_bind_args = {'parameter_binds': [{x: 1}, {x: 0}]}
        full_bind_args = {'parameter_binds': [{x: 1, y: 1}, {x: 0, y: 0}]}
        inconsistent_bind_args = {'parameter_binds': [{x: 1}, {x: 0, y: 0}]}

        # Raise when parameters passed for non-parametric circuit
        self.assertRaises(QiskitError, assemble,
                          full_bound_circ, **partial_bind_args)

        # Raise when no parameters passed for parametric circuit
        self.assertRaises(QiskitError, assemble, partial_param_circ)
        self.assertRaises(QiskitError, assemble, full_param_circ)

        # Raise when circuit has more parameters than run_config
        self.assertRaises(QiskitError, assemble,
                          full_param_circ, **partial_bind_args)

        # Raise when not all circuits have all parameters
        self.assertRaises(QiskitError, assemble,
                          [full_param_circ, partial_param_circ], **full_bind_args)

        # Raise when not all binds have all circuit params
        self.assertRaises(QiskitError, assemble,
                          full_param_circ, **inconsistent_bind_args)

    def test_assemble_circuits_binds_parameters(self):
        """Verify assemble_circuits applies parameter bindings and output circuits are bound."""
        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)

        x = Parameter('x')
        y = Parameter('y')

        qc1.u2(x, y, qr[0])

        qc2.rz(x, qr[0])
        qc2.rz(y, qr[0])

        bind_args = {'parameter_binds': [{x: 0, y: 0},
                                         {x: 1, y: 0},
                                         {x: 1, y: 1}]}

        qobj = assemble([qc1, qc2], **bind_args)

        self.assertEqual(len(qobj.experiments), 6)
        self.assertEqual([len(expt.instructions) for expt in qobj.experiments],
                         [1, 1, 1, 2, 2, 2])

        self.assertEqual(qobj.experiments[0].instructions[0].params, [0, 0])
        self.assertEqual(qobj.experiments[1].instructions[0].params, [1, 0])
        self.assertEqual(qobj.experiments[2].instructions[0].params, [1, 1])

        self.assertEqual(qobj.experiments[3].instructions[0].params, [0])
        self.assertEqual(qobj.experiments[3].instructions[1].params, [0])
        self.assertEqual(qobj.experiments[4].instructions[0].params, [1])
        self.assertEqual(qobj.experiments[4].instructions[1].params, [0])
        self.assertEqual(qobj.experiments[5].instructions[0].params, [1])
        self.assertEqual(qobj.experiments[5].instructions[1].params, [1])


class TestPulseAssembler(QiskitTestCase):
    """Tests for assembling schedules to qobj."""

    def setUp(self):
        self.device = pulse.DeviceSpecification.create_from(FakeOpenPulse2Q())

        test_pulse = pulse.SamplePulse(
            samples=np.array([0.02739068, 0.05, 0.05, 0.05, 0.02739068], dtype=np.complex128)
        )
        acquire = pulse.Acquire(5)

        self.schedule = pulse.Schedule(name='fake_experiment')
        self.schedule = self.schedule.insert(0, test_pulse(self.device.q[0].drive))
        self.schedule = self.schedule.insert(5, acquire(self.device.q, self.device.mem))

        self.user_lo_config_dict = {self.device.q[0].drive: 4.91}
        self.user_lo_config = pulse.LoConfig(self.user_lo_config_dict)

        self.default_qubit_lo_freq = [4.9, 5.0]
        self.default_meas_lo_freq = [6.5, 6.6]

        self.config = {
            'meas_level': 1,
            'memory_slots': 2,
            'memory_slot_size': 100,
            'meas_return': 'avg',
            'rep_time': 100
        }

        self.header = {
            'backend_name': 'FakeOpenPulse2Q',
            'backend_version': '0.0.0'
        }

    def test_assemble_single_schedule_without_lo_config(self):
        """Test assembling a single schedule, no lo config."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[],
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.9, 5.0])
        self.assertEqual(len(test_dict['experiments']), 1)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)

    def test_assemble_multi_schedules_without_lo_config(self):
        """Test assembling schedules, no lo config."""
        qobj = assemble([self.schedule, self.schedule],
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.9, 5.0])
        self.assertEqual(len(test_dict['experiments']), 2)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)

    def test_assemble_single_schedule_with_lo_config(self):
        """Test assembling a single schedule, with a single lo config."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=self.user_lo_config,
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.91, 5.0])
        self.assertEqual(len(test_dict['experiments']), 1)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)

    def test_assemble_single_schedule_with_lo_config_dict(self):
        """Test assembling a single schedule, with a single lo config supplied as dictionary."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=self.user_lo_config_dict,
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.91, 5.0])
        self.assertEqual(len(test_dict['experiments']), 1)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)

    def test_assemble_single_schedule_with_multi_lo_configs(self):
        """Test assembling a single schedule, with lo configs (frequency sweep)."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[self.user_lo_config, self.user_lo_config],
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.9, 5.0])
        self.assertEqual(len(test_dict['experiments']), 2)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)
        self.assertDictEqual(test_dict['experiments'][0]['config'],
                             {'qubit_lo_freq': [4.91, 5.0]})

    def test_assemble_multi_schedules_with_multi_lo_configs(self):
        """Test assembling schedules, with the same number of lo configs (n:n setup)."""
        qobj = assemble([self.schedule, self.schedule],
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[self.user_lo_config, self.user_lo_config],
                        **self.config)
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict['config']['qubit_lo_freq'], [4.9, 5.0])
        self.assertEqual(len(test_dict['experiments']), 2)
        self.assertEqual(len(test_dict['experiments'][0]['instructions']), 2)
        self.assertDictEqual(test_dict['experiments'][0]['config'],
                             {'qubit_lo_freq': [4.91, 5.0]})

    def test_assemble_multi_schedules_with_wrong_number_of_multi_lo_configs(self):
        """Test assembling schedules, with a different number of lo configs (n:m setup)."""
        with self.assertRaises(QiskitError):
            assemble([self.schedule, self.schedule, self.schedule],
                     qobj_header=self.header,
                     qubit_lo_freq=self.default_qubit_lo_freq,
                     meas_lo_freq=self.default_meas_lo_freq,
                     schedule_los=[self.user_lo_config, self.user_lo_config],
                     **self.config)

    def test_assemble_meas_map(self):
        """Test assembling a single schedule, no lo config."""
        acquire = pulse.Acquire(5)
        schedule = acquire(self.device.q, mem_slots=self.device.mem)
        assemble(schedule, meas_map=[[0], [1]])

        with self.assertRaises(QiskitError):
            assemble(schedule, meas_map=[[0, 1, 2]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
