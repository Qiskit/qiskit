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
from qiskit.pulse import Schedule, Acquire
from qiskit.pulse.channels import MemorySlot, AcquireChannel, DriveChannel, MeasureChannel
from qiskit.pulse.pulse_lib import gaussian
from qiskit.qobj import QasmQobj, validate_qobj_against_schema
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.scheduler import measure
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q, FakeOpenPulse3Q, FakeYorktown, FakeAlmaden
from qiskit.validation.jsonschema import SchemaValidationError


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
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        qobj = assemble(circ)
        validate_qobj_against_schema(qobj)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 1024)

    def test_shots_greater_than_max_shots(self):
        """Test assembling with shots greater than max shots"""
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)
        backend = FakeYorktown()

        self.assertRaises(QiskitError, assemble, backend, shots=1024000)

    def test_default_shots_greater_than_max_shots(self):
        """Test assembling with default shots greater than max shots"""
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)
        backend = FakeYorktown()
        backend._configuration.max_shots = 5

        qobj = assemble(circ, backend)

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


class TestPulseAssembler(QiskitTestCase):
    """Tests for assembling schedules to qobj."""

    def setUp(self):
        self.backend_config = FakeOpenPulse2Q().configuration()

        test_pulse = pulse.SamplePulse(
            samples=np.array([0.02739068, 0.05, 0.05, 0.05, 0.02739068], dtype=np.complex128),
            name='pulse0'
        )
        acquire = pulse.Acquire(5)

        self.schedule = pulse.Schedule(name='fake_experiment')
        self.schedule = self.schedule.insert(0, test_pulse(self.backend_config.drive(0)))
        for i in range(self.backend_config.n_qubits):
            self.schedule = self.schedule.insert(5, acquire(self.backend_config.acquire(i),
                                                            MemorySlot(i)))

        self.user_lo_config_dict = {self.backend_config.drive(0): 4.91e9}
        self.user_lo_config = pulse.LoConfig(self.user_lo_config_dict)

        self.default_qubit_lo_freq = [4.9e9, 5.0e9]
        self.default_meas_lo_freq = [6.5e9, 6.6e9]

        self.config = {
            'meas_level': 1,
            'memory_slot_size': 100,
            'meas_return': 'avg',
            'rep_time': 0.0001,
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
        acquire = pulse.Acquire(5)
        schedule = Schedule(name='fake_experiment')
        schedule = schedule.insert(5, acquire(AcquireChannel(0), MemorySlot(0)))
        schedule = schedule.insert(5, acquire(AcquireChannel(1), MemorySlot(1)))

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
        acquire = pulse.Acquire(5)
        n_memoryslots = 10

        # single acquisition
        schedule = acquire(self.backend_config.acquire(0),
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
        schedule = acquire(self.backend_config.acquire(0),
                           mem_slot=pulse.MemorySlot(n_memoryslots-1))
        schedule = schedule.insert(10, acquire(self.backend_config.acquire(0),
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
        acquire = pulse.Acquire(5)
        n_memoryslots = [10, 5, 7]

        schedules = []
        for n_memoryslot in n_memoryslots:
            schedule = acquire(self.backend_config.acquire(0),
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
        name_conflict_pulse = pulse.SamplePulse(
            samples=np.array([0.02, 0.05, 0.05, 0.05, 0.02], dtype=np.complex128),
            name='pulse0'
        )
        self.schedule = self.schedule.insert(1, name_conflict_pulse(self.backend_config.drive(1)))
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
            sched += gaussian(duration=100, amp=amp, sigma=30, name='my_pulse')(ch_d0)
            sched += measure(qubits=[0], backend=backend) << 100
            schedules.append(sched)

        qobj = assemble(schedules, backend)

        # two user pulses and one measurement pulse should be contained
        self.assertEqual(len(qobj.config.pulse_library), 3)

    def test_assemble_with_delay(self):
        """Test that delay instruction is ignored in assembly."""
        backend = FakeOpenPulse2Q()

        orig_schedule = self.schedule
        delay_schedule = orig_schedule + pulse.Delay(10)(self.backend_config.drive(0))

        orig_qobj = assemble(orig_schedule, backend)
        validate_qobj_against_schema(orig_qobj)
        delay_qobj = assemble(delay_schedule, backend)
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
        sched += pulse.Gaussian(duration=25, sigma=4, amp=0.5j)(DriveChannel(0))
        sched += pulse.Drag(duration=25, amp=0.2+0.3j, sigma=7.8, beta=4)(DriveChannel(1))
        sched += pulse.ConstantPulse(duration=25, amp=1)(DriveChannel(2))
        sched += pulse.GaussianSquare(duration=150, amp=0.2,
                                      sigma=8, width=140)(MeasureChannel(0)) << sched.duration
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
        """Test that parametric pulses are translated to SamplePulses if they're not supported
        by the backend during assemble time.
        """
        sched = pulse.Schedule(name='test_parametric_to_sample_pulse')
        sched += pulse.Drag(duration=25, amp=0.2+0.3j, sigma=7.8, beta=4)(DriveChannel(1))
        sched += pulse.ConstantPulse(duration=25, amp=1)(DriveChannel(2))

        backend = FakeOpenPulse3Q()
        backend.configuration().parametric_pulses = ['something_extra']

        qobj = assemble(sched, backend)

        self.assertNotEqual(qobj.config.pulse_library, [])
        qobj_insts = qobj.experiments[0].instructions
        self.assertFalse(hasattr(qobj_insts[0], 'pulse_shape'))


class TestPulseAssemblerMissingKwargs(QiskitTestCase):
    """Verify that errors are raised in case backend is not provided and kwargs are missing."""

    def setUp(self):
        self.schedule = pulse.Schedule(name='fake_experiment')
        self.schedule += pulse.FrameChange(0.)(pulse.DriveChannel(0))

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
        self.rep_time = self.config.rep_times[0]

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
                        rep_time=self.rep_time)
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
                     rep_time=self.rep_time)

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
                     rep_time=self.rep_time)

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
                        rep_time=self.rep_time)
        validate_qobj_against_schema(qobj)

    def test_missing_rep_time(self):
        """Test that assembly still works if rep_time is missing.

        The case of no rep_time will exist for a simulator.
        """
        qobj = assemble(self.schedule,
                        qubit_lo_freq=self.qubit_lo_freq,
                        meas_lo_freq=self.meas_lo_freq,
                        qubit_lo_range=self.qubit_lo_range,
                        meas_lo_range=self.meas_lo_range,
                        schedule_los=self.schedule_los,
                        meas_map=self.meas_map,
                        memory_slots=self.memory_slots,
                        rep_time=None)
        validate_qobj_against_schema(qobj)

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
                        rep_time=self.rep_time)
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
                        rep_time=self.rep_time)
        validate_qobj_against_schema(qobj)

    def test_unsupported_meas_level(self):
        """Test that assembly raises an error if meas_level is not supported"""
        # pylint: disable=unused-variable
        backend = FakeOpenPulse2Q()
        backend.configuration().meas_levels = [1, 2]
        with self.assertRaises(SchemaValidationError):
            qobj = assemble(self.schedule,
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
                            )

    def test_single_and_deprecated_acquire_styles(self):
        """Test that acquires are identically combined with Acquires that take a single channel."""
        backend = FakeOpenPulse2Q()
        new_style_schedule = Schedule()
        acq = Acquire(1200)
        for i in range(5):
            new_style_schedule += acq(AcquireChannel(i), MemorySlot(i))

        deprecated_style_schedule = Schedule()
        deprecated_style_schedule += acq([AcquireChannel(i) for i in range(5)],
                                         [MemorySlot(i) for i in range(5)])

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
