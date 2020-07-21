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
import io
from logging import StreamHandler, getLogger
import sys

import numpy as np
import qiskit.pulse as pulse
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler.assemble import assemble
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule, Acquire, Play
from qiskit.pulse.channels import MemorySlot, AcquireChannel, DriveChannel, MeasureChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.library import gaussian
from qiskit.qobj import QasmQobj, validate_qobj_against_schema
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.pulse.macros import measure
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q, FakeOpenPulse3Q, FakeYorktown, FakeAlmaden
from qiskit.validation.jsonschema import SchemaValidationError


class TestCircuitAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def setUp(self):
        qr = QuantumRegister(2, name='q')
        cr = ClassicalRegister(2, name='c')
        self.circ = QuantumCircuit(qr, cr, name='circ')
        self.circ.h(qr[0])
        self.circ.cx(qr[0], qr[1])
        self.circ.measure(qr, cr)

    def test_assemble_single_circuit(self):
        """Test assembling a single circuit.
        """
        qobj = assemble(self.circ, shots=2000, memory=True)
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.seed_simulator, 6)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(qobj.experiments[1].config.n_qubits, 3)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(len(qobj.experiments[1].instructions), 6)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default.
        """
        qobj = assemble(self.circ)
        validate_qobj_against_schema(qobj)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 1024)

    def test_shots_greater_than_max_shots(self):
        """Test assembling with shots greater than max shots"""
        backend = FakeYorktown()
        self.assertRaises(QiskitError, assemble, backend, shots=1024000)

    def test_default_shots_greater_than_max_shots(self):
        """Test assembling with default shots greater than max shots"""
        backend = FakeYorktown()
        backend._configuration.max_shots = 5

        qobj = assemble(self.circ, backend)

        validate_qobj_against_schema(qobj)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 5)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble(circ)
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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

    def test_assemble_circuits_rases_for_bind_mismatch_over_expressions(self):
        """Verify assemble_circuits raises for invalid binds for circuit including
        ParameterExpressions.
        """
        qr = QuantumRegister(1)
        x = Parameter('x')
        y = Parameter('y')

        expr_circ = QuantumCircuit(qr)

        expr_circ.u1(x+y, qr[0])

        partial_bind_args = {'parameter_binds': [{x: 1}, {x: 0}]}

        # Raise when no parameters passed for parametric circuit
        self.assertRaises(QiskitError, assemble, expr_circ)

        # Raise when circuit has more parameters than run_config
        self.assertRaises(QiskitError, assemble,
                          expr_circ, **partial_bind_args)

    def test_assemble_circuits_binds_parameters(self):
        """Verify assemble_circuits applies parameter bindings and output circuits are bound."""
        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)
        qc3 = QuantumCircuit(qr)

        x = Parameter('x')
        y = Parameter('y')
        sum_ = x + y
        product_ = x * y

        qc1.u2(x, y, qr[0])

        qc2.rz(x, qr[0])
        qc2.rz(y, qr[0])

        qc3.u2(sum_, product_, qr[0])

        bind_args = {'parameter_binds': [{x: 0, y: 0},
                                         {x: 1, y: 0},
                                         {x: 1, y: 1}]}

        qobj = assemble([qc1, qc2, qc3], **bind_args)
        validate_qobj_against_schema(qobj)

        self.assertEqual(len(qobj.experiments), 9)
        self.assertEqual([len(expt.instructions) for expt in qobj.experiments],
                         [1, 1, 1, 2, 2, 2, 1, 1, 1])

        def _qobj_inst_params(expt_no, inst_no):
            expt = qobj.experiments[expt_no]
            inst = expt.instructions[inst_no]
            return [float(p) for p in inst.params]

        self.assertEqual(_qobj_inst_params(0, 0), [0, 0])
        self.assertEqual(_qobj_inst_params(1, 0), [1, 0])
        self.assertEqual(_qobj_inst_params(2, 0), [1, 1])

        self.assertEqual(_qobj_inst_params(3, 0), [0])
        self.assertEqual(_qobj_inst_params(3, 1), [0])
        self.assertEqual(_qobj_inst_params(4, 0), [1])
        self.assertEqual(_qobj_inst_params(4, 1), [0])
        self.assertEqual(_qobj_inst_params(5, 0), [1])
        self.assertEqual(_qobj_inst_params(5, 1), [1])

        self.assertEqual(_qobj_inst_params(6, 0), [0, 0])
        self.assertEqual(_qobj_inst_params(7, 0), [1, 0])
        self.assertEqual(_qobj_inst_params(8, 0), [2, 1])

    def test_init_qubits_default(self):
        """Check that the init_qubits=None assemble option is passed on to the qobj."""
        qobj = assemble(self.circ)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_true(self):
        """Check that the init_qubits=True assemble option is passed on to the qobj."""
        qobj = assemble(self.circ, init_qubits=True)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_false(self):
        """Check that the init_qubits=False assemble option is passed on to the qobj."""
        qobj = assemble(self.circ, init_qubits=False)
        self.assertEqual(qobj.config.init_qubits, False)


class TestPulseAssembler(QiskitTestCase):
    """Tests for assembling schedules to qobj."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.backend_config = self.backend.configuration()

        test_pulse = pulse.Waveform(
            samples=np.array([0.02739068, 0.05, 0.05, 0.05, 0.02739068], dtype=np.complex128),
            name='pulse0'
        )

        self.schedule = pulse.Schedule(name='fake_experiment')
        self.schedule = self.schedule.insert(0, Play(test_pulse, self.backend_config.drive(0)))
        for i in range(self.backend_config.n_qubits):
            self.schedule = self.schedule.insert(5, Acquire(5,
                                                            self.backend_config.acquire(i),
                                                            MemorySlot(i)))

        self.user_lo_config_dict = {self.backend_config.drive(0): 4.91e9}
        self.user_lo_config = pulse.LoConfig(self.user_lo_config_dict)

        self.default_qubit_lo_freq = [4.9e9, 5.0e9]
        self.default_meas_lo_freq = [6.5e9, 6.6e9]

        self.config = {
            'meas_level': 1,
            'memory_slot_size': 100,
            'meas_return': 'avg'
        }

        self.header = {
            'backend_name': 'FakeOpenPulse2Q',
            'backend_version': '0.0.0'
        }

    def test_assemble_sample_pulse(self):
        """Test that the pulse lib and qobj instruction can be paired up."""
        schedule = pulse.Schedule()
        schedule += pulse.Play(pulse.Waveform([0.1]*16, name='test0'),
                               pulse.DriveChannel(0),
                               name='test1')
        schedule += pulse.Play(pulse.SamplePulse([0.1]*16, name='test1'),
                               pulse.DriveChannel(0),
                               name='test2')
        schedule += pulse.Play(pulse.Waveform([0.5]*16, name='test0'),
                               pulse.DriveChannel(0),
                               name='test1')
        qobj = assemble(schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[],
                        **self.config)
        validate_qobj_against_schema(qobj)

        test_dict = qobj.to_dict()
        experiment = test_dict['experiments'][0]
        inst0_name = experiment['instructions'][0]['name']
        inst1_name = experiment['instructions'][1]['name']
        inst2_name = experiment['instructions'][2]['name']
        pulses = {}
        for item in test_dict['config']['pulse_library']:
            pulses[item['name']] = item['samples']
        self.assertTrue(all(name in pulses for name in [inst0_name, inst1_name, inst2_name]))
        # Their pulses are the same
        self.assertEqual(inst0_name, inst1_name)
        self.assertTrue(np.allclose(pulses[inst0_name], [0.1]*16))
        self.assertTrue(np.allclose(pulses[inst2_name], [0.5]*16))

    def test_assemble_single_schedule_without_lo_config(self):
        """Test assembling a single schedule, no lo config."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[],
                        **self.config)
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        validate_qobj_against_schema(qobj)

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
        schedule = Schedule(name='fake_experiment')
        schedule = schedule.insert(5, Acquire(5, AcquireChannel(0), MemorySlot(0)))
        schedule = schedule.insert(5, Acquire(5, AcquireChannel(1), MemorySlot(1)))

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0], [1]])
        validate_qobj_against_schema(qobj)

        with self.assertRaises(QiskitError):
            assemble(schedule,
                     qubit_lo_freq=self.default_qubit_lo_freq,
                     meas_lo_freq=self.default_meas_lo_freq,
                     meas_map=[[0, 1, 2]])

    def test_assemble_memory_slots(self):
        """Test assembling a schedule and inferring number of memoryslots."""
        n_memoryslots = 10

        # single acquisition
        schedule = Acquire(5,
                           self.backend_config.acquire(0),
                           mem_slot=pulse.MemorySlot(n_memoryslots-1))

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0], [1]])
        validate_qobj_against_schema(qobj)

        self.assertEqual(qobj.config.memory_slots, n_memoryslots)
        # this should be in experimental header as well
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots)

        # multiple acquisition
        schedule = Acquire(5, self.backend_config.acquire(0),
                           mem_slot=pulse.MemorySlot(n_memoryslots-1))
        schedule = schedule.insert(10, Acquire(5, self.backend_config.acquire(0),
                                               mem_slot=pulse.MemorySlot(n_memoryslots-1)))

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0], [1]])
        validate_qobj_against_schema(qobj)

        self.assertEqual(qobj.config.memory_slots, n_memoryslots)
        # this should be in experimental header as well
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots)

    def test_assemble_memory_slots_for_schedules(self):
        """Test assembling schedules with different memory slots."""
        n_memoryslots = [10, 5, 7]

        schedules = []
        for n_memoryslot in n_memoryslots:
            schedule = Acquire(5, self.backend_config.acquire(0),
                               mem_slot=pulse.MemorySlot(n_memoryslot-1))
            schedules.append(schedule)

        qobj = assemble(schedules,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0], [1]])
        validate_qobj_against_schema(qobj)

        self.assertEqual(qobj.config.memory_slots, max(n_memoryslots))
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots[0])
        self.assertEqual(qobj.experiments[1].header.memory_slots, n_memoryslots[1])
        self.assertEqual(qobj.experiments[2].header.memory_slots, n_memoryslots[2])

    def test_pulse_name_conflicts(self):
        """Test that pulse name conflicts can be resolved."""
        name_conflict_pulse = pulse.Waveform(
            samples=np.array([0.02, 0.05, 0.05, 0.05, 0.02], dtype=np.complex128),
            name='pulse0'
        )

        self.schedule = self.schedule.insert(1, Play(name_conflict_pulse,
                                                     self.backend_config.drive(1)))

        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[],
                        **self.config)
        validate_qobj_against_schema(qobj)

        self.assertNotEqual(qobj.config.pulse_library[0].name,
                            qobj.config.pulse_library[1].name)

    def test_pulse_name_conflicts_in_other_schedule(self):
        """Test two pulses with the same name in different schedule can be resolved."""
        backend = FakeAlmaden()

        schedules = []
        ch_d0 = pulse.DriveChannel(0)
        for amp in (0.1, 0.2):
            sched = Schedule()
            sched += Play(gaussian(duration=100, amp=amp, sigma=30, name='my_pulse'), ch_d0)
            sched += measure(qubits=[0], backend=backend) << 100
            schedules.append(sched)

        qobj = assemble(schedules, backend)

        # two user pulses and one measurement pulse should be contained
        self.assertEqual(len(qobj.config.pulse_library), 3)

    def test_assemble_with_delay(self):
        """Test that delay instruction is ignored in assembly."""
        orig_schedule = self.schedule
        delay_schedule = orig_schedule + pulse.Delay(10, self.backend_config.drive(0))

        orig_qobj = assemble(orig_schedule, self.backend)
        validate_qobj_against_schema(orig_qobj)
        delay_qobj = assemble(delay_schedule, self.backend)
        validate_qobj_against_schema(delay_qobj)

        self.assertEqual(orig_qobj.experiments[0].to_dict(),
                         delay_qobj.experiments[0].to_dict())

    def test_assemble_schedule_enum(self):
        """Test assembling a schedule with enum input values to assemble."""
        qobj = assemble(self.schedule,
                        qobj_header=self.header,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        schedule_los=[],
                        meas_level=MeasLevel.CLASSIFIED,
                        meas_return=MeasReturnType.AVERAGE)
        validate_qobj_against_schema(qobj)

        test_dict = qobj.to_dict()
        self.assertEqual(test_dict['config']['meas_return'], 'avg')
        self.assertEqual(test_dict['config']['meas_level'], 2)

    def test_assemble_parametric(self):
        """Test that parametric pulses can be assembled properly into a PulseQobj."""
        sched = pulse.Schedule(name='test_parametric')
        sched += Play(pulse.Gaussian(duration=25, sigma=4, amp=0.5j), DriveChannel(0))
        sched += Play(pulse.Drag(duration=25, amp=0.2+0.3j, sigma=7.8, beta=4), DriveChannel(1))
        sched += Play(pulse.Constant(duration=25, amp=1), DriveChannel(2))
        sched += Play(pulse.GaussianSquare(duration=150, amp=0.2,
                                           sigma=8, width=140), MeasureChannel(0)) << sched.duration
        backend = FakeOpenPulse3Q()
        backend.configuration().parametric_pulses = ['gaussian', 'drag',
                                                     'gaussian_square', 'constant']
        qobj = assemble(sched, backend)

        self.assertEqual(qobj.config.pulse_library, [])
        qobj_insts = qobj.experiments[0].instructions
        self.assertTrue(all(inst.name == 'parametric_pulse'
                            for inst in qobj_insts))
        self.assertEqual(qobj_insts[0].pulse_shape, 'gaussian')
        self.assertEqual(qobj_insts[1].pulse_shape, 'drag')
        self.assertEqual(qobj_insts[2].pulse_shape, 'constant')
        self.assertEqual(qobj_insts[3].pulse_shape, 'gaussian_square')
        self.assertDictEqual(qobj_insts[0].parameters, {'duration': 25, 'sigma': 4, 'amp': 0.5j})
        self.assertDictEqual(qobj_insts[1].parameters,
                             {'duration': 25, 'sigma': 7.8, 'amp': 0.2+0.3j, 'beta': 4})
        self.assertDictEqual(qobj_insts[2].parameters, {'duration': 25, 'amp': 1})
        self.assertDictEqual(qobj_insts[3].parameters,
                             {'duration': 150, 'sigma': 8, 'amp': 0.2, 'width': 140})
        self.assertEqual(
            qobj.to_dict()['experiments'][0]['instructions'][0]['parameters']['amp'],
            0.5j)

    def test_assemble_parametric_unsupported(self):
        """Test that parametric pulses are translated to Waveform if they're not supported
        by the backend during assemble time.
        """
        sched = pulse.Schedule(name='test_parametric_to_sample_pulse')
        sched += Play(pulse.Drag(duration=25, amp=0.2+0.3j, sigma=7.8, beta=4), DriveChannel(1))
        sched += Play(pulse.Constant(duration=25, amp=1), DriveChannel(2))

        backend = FakeOpenPulse3Q()
        backend.configuration().parametric_pulses = ['something_extra']

        qobj = assemble(sched, backend)

        self.assertNotEqual(qobj.config.pulse_library, [])
        qobj_insts = qobj.experiments[0].instructions
        self.assertFalse(hasattr(qobj_insts[0], 'pulse_shape'))

    def test_init_qubits_default(self):
        """Check that the init_qubits=None assemble option is passed on to the qobj."""
        qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_true(self):
        """Check that the init_qubits=True assemble option is passed on to the qobj."""
        qobj = assemble(self.schedule, self.backend, init_qubits=True)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_false(self):
        """Check that the init_qubits=False assemble option is passed on to the qobj."""
        qobj = assemble(self.schedule, self.backend, init_qubits=False)
        self.assertEqual(qobj.config.init_qubits, False)

    def test_assemble_backend_rep_times_delays(self):
        """Check that rep_time and rep_delay are properly set from backend values."""
        # use first entry from allowed backend values
        rep_times = [2.0, 3.0, 4.0]  # sec
        rep_delays = [2.5e-3, 3.5e-3, 4.5e-3]
        self.backend_config.rep_times = rep_times
        self.backend_config.rep_delays = rep_delays
        # RuntimeWarning bc using ``rep_delay`` when dynamic rep rates not enabled
        with self.assertWarns(RuntimeWarning):
            qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0]*1e6))
        self.assertEqual(qobj.config.rep_delay, rep_delays[0]*1e6)

        # remove rep_delays from backend config and make sure things work
        # now no warning
        del self.backend_config.rep_delays
        qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0]*1e6))
        self.assertEqual(hasattr(qobj.config, 'rep_delay'), False)

    def test_assemble_user_rep_time_delay(self):
        """Check that user runtime config rep_time and rep_delay work."""
        # set custom rep_time and rep_delay in runtime config
        rep_time = 200.0e-6
        rep_delay = 2.5e-6
        self.config['rep_time'] = rep_time
        self.config['rep_delay'] = rep_delay
        # RuntimeWarning bc using ``rep_delay`` when dynamic rep rates not enabled
        with self.assertWarns(RuntimeWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_time*1e6))
        self.assertEqual(qobj.config.rep_delay, rep_delay*1e6)

        # now remove rep_delay and set enable dynamic rep rates
        # RuntimeWarning bc using ``rep_time`` when dynamic rep rates are enabled
        del self.config['rep_delay']
        self.backend_config.dynamic_reprate_enabled = True
        with self.assertWarns(RuntimeWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_time*1e6))
        self.assertEqual(hasattr(qobj.config, 'rep_delay'), False)

        # finally, only use rep_delay and verify that everything runs w/ no warning
        # rep_time comes from allowed backed rep_times
        rep_times = [0.5, 1.0, 1.5]  # sec
        self.backend_config.rep_times = rep_times
        del self.config['rep_time']
        self.config['rep_delay'] = rep_delay
        qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0]*1e6))
        self.assertEqual(qobj.config.rep_delay, rep_delay*1e6)

    def test_assemble_with_individual_discriminators(self):
        """Test that assembly works with individual discriminators."""
        disc_one = Discriminator('disc_one', test_params=True)
        disc_two = Discriminator('disc_two', test_params=False)

        schedule = Schedule()
        schedule = schedule.append(
            Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one),
        )
        schedule = schedule.append(
            Acquire(5, AcquireChannel(1), MemorySlot(1), discriminator=disc_two),
        )

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0, 1]])
        validate_qobj_against_schema(qobj)

        qobj_discriminators = qobj.experiments[0].instructions[0].discriminators
        self.assertEqual(len(qobj_discriminators), 2)
        self.assertEqual(qobj_discriminators[0].name, 'disc_one')
        self.assertEqual(qobj_discriminators[0].params['test_params'], True)
        self.assertEqual(qobj_discriminators[1].name, 'disc_two')
        self.assertEqual(qobj_discriminators[1].params['test_params'], False)

    def test_assemble_with_single_discriminators(self):
        """Test that assembly works with both a single discriminator."""
        disc_one = Discriminator('disc_one', test_params=True)

        schedule = Schedule()
        schedule = schedule.append(
            Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one),
        )
        schedule = schedule.append(
            Acquire(5, AcquireChannel(1), MemorySlot(1)),
        )

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0, 1]])
        validate_qobj_against_schema(qobj)

        qobj_discriminators = qobj.experiments[0].instructions[0].discriminators
        self.assertEqual(len(qobj_discriminators), 1)
        self.assertEqual(qobj_discriminators[0].name, 'disc_one')
        self.assertEqual(qobj_discriminators[0].params['test_params'], True)

    def test_assemble_with_unequal_discriminators(self):
        """Test that assembly works with incorrect number of discriminators for
        number of qubits."""
        disc_one = Discriminator('disc_one', test_params=True)
        disc_two = Discriminator('disc_two', test_params=False)

        schedule = Schedule()
        schedule += Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one)
        schedule += Acquire(5, AcquireChannel(1), MemorySlot(1), discriminator=disc_two)
        schedule += Acquire(5, AcquireChannel(2), MemorySlot(2))

        with self.assertRaises(QiskitError):
            assemble(schedule,
                     qubit_lo_freq=self.default_qubit_lo_freq,
                     meas_lo_freq=self.default_meas_lo_freq,
                     meas_map=[[0, 1, 2]])

    def test_assemble_with_individual_kernels(self):
        """Test that assembly works with individual kernels."""
        disc_one = Kernel('disc_one', test_params=True)
        disc_two = Kernel('disc_two', test_params=False)

        schedule = Schedule()
        schedule = schedule.append(
            Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one),
        )
        schedule = schedule.append(
            Acquire(5, AcquireChannel(1), MemorySlot(1), kernel=disc_two),
        )

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0, 1]])
        validate_qobj_against_schema(qobj)

        qobj_kernels = qobj.experiments[0].instructions[0].kernels
        self.assertEqual(len(qobj_kernels), 2)
        self.assertEqual(qobj_kernels[0].name, 'disc_one')
        self.assertEqual(qobj_kernels[0].params['test_params'], True)
        self.assertEqual(qobj_kernels[1].name, 'disc_two')
        self.assertEqual(qobj_kernels[1].params['test_params'], False)

    def test_assemble_with_single_kernels(self):
        """Test that assembly works with both a single kernel."""
        disc_one = Kernel('disc_one', test_params=True)

        schedule = Schedule()
        schedule = schedule.append(
            Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one),
        )
        schedule = schedule.append(
            Acquire(5, AcquireChannel(1), MemorySlot(1)),
        )

        qobj = assemble(schedule,
                        qubit_lo_freq=self.default_qubit_lo_freq,
                        meas_lo_freq=self.default_meas_lo_freq,
                        meas_map=[[0, 1]])
        validate_qobj_against_schema(qobj)

        qobj_kernels = qobj.experiments[0].instructions[0].kernels
        self.assertEqual(len(qobj_kernels), 1)
        self.assertEqual(qobj_kernels[0].name, 'disc_one')
        self.assertEqual(qobj_kernels[0].params['test_params'], True)

    def test_assemble_with_unequal_kernels(self):
        """Test that assembly works with incorrect number of discriminators for
        number of qubits."""
        disc_one = Kernel('disc_one', test_params=True)
        disc_two = Kernel('disc_two', test_params=False)

        schedule = Schedule()
        schedule += Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one)
        schedule += Acquire(5, AcquireChannel(1), MemorySlot(1), kernel=disc_two)
        schedule += Acquire(5, AcquireChannel(2), MemorySlot(2))

        with self.assertRaises(QiskitError):
            assemble(schedule,
                     qubit_lo_freq=self.default_qubit_lo_freq,
                     meas_lo_freq=self.default_meas_lo_freq,
                     meas_map=[[0, 1, 2]])


class TestPulseAssemblerMissingKwargs(QiskitTestCase):
    """Verify that errors are raised in case backend is not provided and kwargs are missing."""

    def setUp(self):
        self.schedule = pulse.Schedule(name='fake_experiment')

        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.qubit_lo_freq = list(self.defaults.qubit_freq_est)
        self.meas_lo_freq = list(self.defaults.meas_freq_est)
        self.qubit_lo_range = self.config.qubit_lo_range
        self.meas_lo_range = self.config.meas_lo_range
        self.schedule_los = {pulse.DriveChannel(0): self.qubit_lo_freq[0],
                             pulse.DriveChannel(1): self.qubit_lo_freq[1],
                             pulse.MeasureChannel(0): self.meas_lo_freq[0],
                             pulse.MeasureChannel(1): self.meas_lo_freq[1]}
        self.meas_map = self.config.meas_map
        self.memory_slots = self.config.n_qubits

        # default rep_time and rep_delay
        self.rep_time = self.config.rep_times[0]
        self.rep_delay = None

    def test_defaults(self):
        """Test defaults work."""
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=self.qubit_lo_range,
                        meas_lo_range=self.meas_lo_range,
                        schedule_los=self.schedule_los,
                        meas_map=self.meas_map,
                        memory_slots=self.memory_slots,
                        rep_time=self.rep_time,
                        rep_delay=self.rep_delay)
        validate_qobj_against_schema(qobj)

    def test_missing_qubit_lo_freq(self):
        """Test error raised if qubit_lo_freq missing."""

        with self.assertRaises(QiskitError):
            assemble(self.schedule,
                     qubit_lo_freq=None,
                     meas_lo_freq=self.meas_lo_freq,
                     qubit_lo_range=self.qubit_lo_range,
                     meas_lo_range=self.meas_lo_range,
                     meas_map=self.meas_map,
                     memory_slots=self.memory_slots,
                     rep_time=self.rep_time,
                     rep_delay=self.rep_delay)

    def test_missing_meas_lo_freq(self):
        """Test error raised if meas_lo_freq missing."""

        with self.assertRaises(QiskitError):
            assemble(self.schedule,
                     qubit_lo_freq=self.qubit_lo_freq,
                     meas_lo_freq=None,
                     qubit_lo_range=self.qubit_lo_range,
                     meas_lo_range=self.meas_lo_range,
                     meas_map=self.meas_map,
                     memory_slots=self.memory_slots,
                     rep_time=self.rep_time,
                     rep_delay=self.rep_delay)

    def test_missing_memory_slots(self):
        """Test error is not raised if memory_slots are missing."""
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=self.qubit_lo_range,
                        meas_lo_range=self.meas_lo_range,
                        schedule_los=self.schedule_los,
                        meas_map=self.meas_map,
                        memory_slots=None,
                        rep_time=self.rep_time,
                        rep_delay=self.rep_delay)
        validate_qobj_against_schema(qobj)

    def test_missing_rep_time_and_delay(self):
        """Test qobj is valid if rep_time and rep_delay are missing."""
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=self.qubit_lo_range,
                        meas_lo_range=self.meas_lo_range,
                        schedule_los=self.schedule_los,
                        meas_map=self.meas_map,
                        memory_slots=None,
                        rep_time=None,
                        rep_delay=None)
        validate_qobj_against_schema(qobj)
        self.assertEqual(hasattr(qobj, 'rep_time'), False)
        self.assertEqual(hasattr(qobj, 'rep_delay'), False)

    def test_missing_meas_map(self):
        """Test that assembly still works if meas_map is missing."""
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=self.qubit_lo_range,
                        meas_lo_range=self.meas_lo_range,
                        schedule_los=self.schedule_los,
                        meas_map=None,
                        memory_slots=self.memory_slots,
                        rep_time=self.rep_time,
                        rep_delay=self.rep_delay)
        validate_qobj_against_schema(qobj)

    def test_missing_lo_ranges(self):
        """Test that assembly still works if lo_ranges are missing."""
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=None,
                        meas_lo_range=None,
                        schedule_los=self.schedule_los,
                        meas_map=self.meas_map,
                        memory_slots=self.memory_slots,
                        rep_time=self.rep_time,
                        rep_delay=self.rep_delay)
        validate_qobj_against_schema(qobj)

    def test_unsupported_meas_level(self):
        """Test that assembly raises an error if meas_level is not supported"""
        # pylint: disable=unused-variable
        backend = FakeOpenPulse2Q()
        backend.configuration().meas_levels = [1, 2]
        with self.assertRaises(SchemaValidationError):
            assemble(self.schedule,
                     backend,
                     qubit_lo_freq=self.qubit_lo_freq,
                     meas_lo_freq=self.meas_lo_freq,
                     qubit_lo_range=self.qubit_lo_range,
                     meas_lo_range=self.meas_lo_range,
                     schedule_los=self.schedule_los,
                     meas_level=0,
                     meas_map=self.meas_map,
                     memory_slots=self.memory_slots,
                     rep_time=self.rep_time,
                     rep_delay=self.rep_delay)

    def test_single_and_deprecated_acquire_styles(self):
        """Test that acquires are identically combined with Acquires that take a single channel."""
        backend = FakeOpenPulse2Q()
        new_style_schedule = Schedule()
        acq_dur = 1200
        for i in range(5):
            new_style_schedule += Acquire(acq_dur, AcquireChannel(i), MemorySlot(i))

        deprecated_style_schedule = Schedule()
        for i in range(5):
            deprecated_style_schedule += Acquire(1200, AcquireChannel(i), MemorySlot(i))

        # The Qobj IDs will be different
        n_qobj = assemble(new_style_schedule, backend)
        n_qobj.qobj_id = None
        n_qobj.experiments[0].header.name = None
        d_qobj = assemble(deprecated_style_schedule, backend)
        d_qobj.qobj_id = None
        d_qobj.experiments[0].header.name = None
        self.assertEqual(n_qobj, d_qobj)

        assembled_acquire = n_qobj.experiments[0].instructions[0]
        self.assertEqual(assembled_acquire.qubits, [0, 1, 2, 3, 4])
        self.assertEqual(assembled_acquire.memory_slot, [0, 1, 2, 3, 4])


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogAssembler(QiskitTestCase):
    """Testing the log_assembly option."""

    def setUp(self):
        logger = getLogger()
        logger.setLevel('DEBUG')
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertAssembleLog(self, log_msg):
        """ Runs assemble and checks for logs containing specified message"""
        assemble(self.circuit, shots=2000, memory=True)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        assembly_log_lines = [x for x in output_lines if log_msg in x]
        self.assertTrue(len(assembly_log_lines) == 1)

    def test_assembly_log_time(self):
        """Check Total Assembly Time is logged"""
        self.assertAssembleLog('Total Assembly Time')


if __name__ == '__main__':
    unittest.main(verbosity=2)
