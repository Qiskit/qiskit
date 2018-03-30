# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,broad-except

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Non-string identifiers for circuit and record identifiers test"""

import unittest

from qiskit import (ClassicalRegister, QISKitError, QuantumCircuit,
                    QuantumRegister, QuantumProgram)
from .common import QiskitTestCase


class TestAnonymousIds(QiskitTestCase):
    """Circuits and records can have no name"""

    def setUp(self):
        self.QPS_SPECS_NONAMES = {
            "circuits": [{
                "quantum_registers": [{
                    "size": 3}],
                "classical_registers": [{
                    "size": 3}]
            }]
        }

    ###############################################################
    # Tests to initiate an build a quantum program with anonymous ids
    ###############################################################

    def test_create_program_with_specs_nonames(self):
        """Test Quantum Object Factory creation using Specs definition
        object with no names for circuit nor records.
        """
        result = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_anonymous_classical_register(self):
        """Test create_classical_register with no name.
        """
        q_program = QuantumProgram()
        cr = q_program.create_classical_register(size=3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_anonymous_quantum_register(self):
        """Test create_quantum_register with no name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(size=3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_classical_registers_noname(self):
        """Test create_classical_registers with no name
        """
        q_program = QuantumProgram()
        classical_registers = [{"size": 4},
                               {"size": 2}]
        crs = q_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers_noname(self):
        """Test create_quantum_registers with no name.
        """
        q_program = QuantumProgram()
        quantum_registers = [{"size": 4},
                             {"size": 2}]
        qrs = q_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_create_circuit_noname(self):
        """Test create_circuit with no name
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(size=3)
        cr = q_program.create_classical_register(size=3)
        qc = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits_noname(self):
        """Test create_circuit with several inputs and without names.
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(size=3)
        cr1 = q_program.create_classical_register(size=3)
        qr2 = q_program.create_quantum_register(size=3)
        cr2 = q_program.create_classical_register(size=3)
        qc1 = q_program.create_circuit(qregisters=[qr1], cregisters=[cr1])
        qc2 = q_program.create_circuit(qregisters=[qr2], cregisters=[cr2])
        qc3 = q_program.create_circuit(qregisters=[qr1, qr2], cregisters=[cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_get_register_and_circuit_names_nonames(self):
        """Get the names of the circuits and registers after create them without a name
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(size=3)
        cr1 = q_program.create_classical_register(size=3)
        qr2 = q_program.create_quantum_register(size=3)
        cr2 = q_program.create_classical_register(size=3)
        q_program.create_circuit(qregisters=[qr1], cregisters=[cr1])
        q_program.create_circuit(qregisters=[qr2], cregisters=[cr2])
        q_program.create_circuit(qregisters=[qr1, qr2], cregisters=[cr1, cr2])
        qrn = q_program.get_quantum_register_names()
        crn = q_program.get_classical_register_names()
        qcn = q_program.get_circuit_names()
        self.assertEqual(len(qrn), 2)
        self.assertEqual(len(crn), 2)
        self.assertEqual(len(qcn), 3)

    def test_get_circuit_noname(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qc = q_program.get_circuit()
        self.assertIsInstance(qc, QuantumCircuit)

    def test_get_quantum_register_noname(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qr = q_program.get_quantum_register()
        self.assertIsInstance(qr, QuantumRegister)

    def test_get_classical_register_noname(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        cr = q_program.get_classical_register()
        self.assertIsInstance(cr, ClassicalRegister)

    def test_get_qasm_noname(self):
        """Test the get_qasm using an specification without names.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qc = q_program.get_circuit()

        qrn = list(q_program.get_quantum_register_names())
        self.assertEqual(len(qrn), 1)
        qr = q_program.get_quantum_register(qrn[0])

        crn = list(q_program.get_classical_register_names())
        self.assertEqual(len(crn), 1)
        cr = q_program.get_classical_register(crn[0])

        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm()
        self.assertEqual(len(result), len(qrn[0]) * 9 + len(crn[0]) * 4 + 147)

    def test_get_qasms_noname(self):
        """Test the get_qasms from a qprogram without names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(size=3)
        cr = q_program.create_classical_register(size=3)
        qc1 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc2 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[1], qr[2])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc1.measure(qr[2], cr[2])
        qc2.h(qr)
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        results = dict(zip(q_program.get_circuit_names(), q_program.get_qasms()))
        qr_name_len = len(qr.openqasm_name)
        cr_name_len = len(cr.openqasm_name)
        self.assertEqual(len(results[qc1.name]), qr_name_len * 9 + cr_name_len * 4 + 147)
        self.assertEqual(len(results[qc2.name]), qr_name_len * 7 + cr_name_len * 4 + 137)

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates, using an specification without names.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qc = q_program.get_circuit()
        qr = q_program.get_quantum_register()
        cr = q_program.get_classical_register()
        qc.u1(0.3, qr[0])
        qc.u2(0.2, 0.1, qr[1])
        qc.u3(0.3, 0.2, 0.1, qr[2])
        qc.s(qr[1])
        qc.s(qr[2]).inverse()
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.x(qr[2]).c_if(cr, 0)
        qc.y(qr[2]).c_if(cr, 1)
        qc.z(qr[2]).c_if(cr, 2)
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm()
        self.assertEqual(len(result), (len(qr.openqasm_name) * 23 +
                                       len(cr.openqasm_name) * 7 +
                                       385))

    ###############################################################
    # Test for compile
    ###############################################################

    def test_compile_program_noname(self):
        """Test compile with a no name.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qc = q_program.get_circuit()
        qr = q_program.get_quantum_register()
        cr = q_program.get_classical_register()
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        out = q_program.compile()
        self.log.info(out)
        self.assertEqual(len(out), 3)

    def test_get_execution_list_noname(self):
        """Test get_execution_list for circuits without name.

        NO?T SURE WHAT THIS TESTS
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qc = q_program.get_circuit()
        qr = q_program.get_quantum_register()
        cr = q_program.get_classical_register()
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qobj = q_program.compile()
        result = q_program.get_execution_list(qobj, print_func=self.log.info)
        self.assertEqual(len(result), 0)

    def test_change_circuit_qobj_after_compile_noname(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_NONAMES)
        qr = q_program.get_quantum_register()
        cr = q_program.get_classical_register()
        qc2 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc3 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = [qc2.name, qc3.name]
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals': [1, 2, 3, 4]}
        qobj1 = q_program.compile(circuits, backend=backend, shots=shots, seed=88, config=config)
        qobj1['circuits'][0]['config']['shots'] = 50
        qobj1['circuits'][0]['config']['xvals'] = [1, 1, 1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = q_program.compile(circuits, backend=backend, shots=shots, seed=88, config=config)
        self.assertTrue(qobj1['circuits'][0]['config']['shots'] == 50)
        self.assertTrue(qobj1['circuits'][1]['config']['shots'] == 1)
        self.assertTrue(qobj1['circuits'][0]['config']['xvals'] == [1, 1, 1])
        self.assertTrue(qobj1['circuits'][1]['config']['xvals'] == [1, 2, 3, 4])
        self.assertTrue(qobj1['config']['shots'] == 1024)
        self.assertTrue(qobj2['circuits'][0]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][1]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][0]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2['circuits'][1]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])

    # def test_add_circuit_noname(self):
        # """Test add two circuits without names. Also tests get_counts without circuit name.
        #
        # CANT WORK OUT WHAT THIS DOES
        # """
        # q_program = QuantumProgram()
        # qr = q_program.create_quantum_register(size=2)
        # cr = q_program.create_classical_register(size=2)
        # qc1 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        # qc2 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        # qc1.h(qr[0])
        # qc1.measure(qr[0], cr[0])
        # qc2.measure(qr[1], cr[1])
        # new_circuit = qc1 + qc2
        # q_program.add_circuit(quantum_circuit=new_circuit)
        # backend = 'local_qasm_simulator'  # the backend to run on
        # shots = 1024  # the number of shots in the experiment.
        # result = q_program.execute(backend=backend, shots=shots, seed=78)
        # self.assertEqual(result.get_counts(new_circuit.name), {'01': 519, '00': 505})
        # self.assertRaises(QISKitError, result.get_counts)


class TestZeroIds(QiskitTestCase):
    """Circuits and records can have zero as names"""

    def setUp(self):
        self.QPS_SPECS_ZEROS = {
            "circuits": [{
                "name": 0,
                "quantum_registers": [{
                    "name": 0,
                    "size": 3}],
                "classical_registers": [{
                    "name": "",
                    "size": 3}]
            }]
        }

    ###############################################################
    # Tests to initiate an build a quantum program with zeros ids
    ###############################################################

    def test_create_program_with_specs(self):
        """Test Quantum Object Factory creation using Specs definition
        object with zeros names for circuit nor records.
        """
        result = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_classical_register(self):
        """Test create_classical_register with zero name
        """
        q_program = QuantumProgram()
        cr = q_program.create_classical_register(0, 3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        """Test create_quantum_register with zero name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(0, 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_classical_register_name(self):
        """Test duplicated create_quantum_register with zeros as names.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register(0, 3)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitError,
                          q_program.create_classical_register, 0, 2)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same name (a zero) and size.

        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(0, 3)
        qr2 = q_program.create_quantum_register(0, 3)
        self.assertIs(qr1, qr2)

    def test_create_classical_register_same(self):
        """Test create_classical_register of same name (a zero) and size.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register(0, 3)
        cr2 = q_program.create_classical_register(0, 3)
        self.assertIs(cr1, cr2)

    def test_create_classical_registers(self):
        """Test create_classical_registers with 0 as a name.
        """
        q_program = QuantumProgram()
        classical_registers = [{"name": 0, "size": 4},
                               {"name": "", "size": 2}]
        crs = q_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers(self):
        """Test create_quantum_registers with 0 as names
        """
        q_program = QuantumProgram()
        quantum_registers = [{"name": 0, "size": 4},
                             {"name": "", "size": 2}]
        qrs = q_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_destroy_classical_register(self):
        """Test destroy_classical_register with 0 as name."""
        q_program = QuantumProgram()
        _ = q_program.create_classical_register(0, 3)
        self.assertIn(0, q_program.get_classical_register_names())
        q_program.destroy_classical_register(0)
        self.assertNotIn(0, q_program.get_classical_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_classical_register(0)
        self.assertIn('Not present', str(context.exception))

    def test_destroy_quantum_register(self):
        """Test destroy_quantum_register with 0 as name."""
        q_program = QuantumProgram()
        _ = q_program.create_quantum_register(0, 3)
        self.assertIn(0, q_program.get_quantum_register_names())
        q_program.destroy_quantum_register(0)
        self.assertNotIn(0, q_program.get_quantum_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_quantum_register(0)
        self.assertIn('Not present', str(context.exception))

    def test_create_circuit(self):
        """Test create_circuit with 0 as a name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(0, 3)
        cr = q_program.create_classical_register("", 3)
        qc = q_program.create_circuit(0, [qr], [cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits(self):
        """Test create_circuit with several inputs with int names.
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(10, 3)
        cr1 = q_program.create_classical_register(20, 3)
        qr2 = q_program.create_quantum_register(11, 3)
        cr2 = q_program.create_classical_register(21, 3)
        qc1 = q_program.create_circuit(30, [qr1], [cr1])
        qc2 = q_program.create_circuit(31, [qr2], [cr2])
        qc3 = q_program.create_circuit(32, [qr1, qr2], [cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_destroy_circuit(self):
        """Test destroy_circuit with an int name."""
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(2, 3)
        cr = q_program.create_classical_register(1, 3)
        _ = q_program.create_circuit(10, [qr], [cr])
        self.assertIn(10, q_program.get_circuit_names())
        q_program.destroy_circuit(10)
        self.assertNotIn(10, q_program.get_circuit_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_circuit(10)
        self.assertIn('Not present', str(context.exception))

    def test_get_register_and_circuit_names(self):
        """Get the names of the circuits and registers when their names are ints.
        """
        qr1n = 10
        qr2n = 11
        cr1n = 12
        cr2n = 13
        qc1n = 14
        qc2n = 15
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(qr1n, 3)
        cr1 = q_program.create_classical_register(cr1n, 3)
        qr2 = q_program.create_quantum_register(qr2n, 3)
        cr2 = q_program.create_classical_register(cr2n, 3)
        q_program.create_circuit(qc1n, [qr1], [cr1])
        q_program.create_circuit(qc2n, [qr2], [cr2])
        q_program.create_circuit(qc2n, [qr1, qr2], [cr1, cr2])
        qrn = q_program.get_quantum_register_names()
        crn = q_program.get_classical_register_names()
        qcn = q_program.get_circuit_names()
        self.assertCountEqual(qrn, [qr1n, qr2n])
        self.assertCountEqual(crn, [cr1n, cr2n])
        self.assertCountEqual(qcn, [qc1n, qc2n])

    def test_get_qasm(self):
        """Test the get_qasm with int name. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        qc = q_program.get_circuit(0)
        qr = q_program.get_quantum_register(0)
        cr = q_program.get_classical_register("")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm(0)
        self.assertEqual(len(result), (147 +
                                       len(qr.openqasm_name) * 9 +
                                       len(cr.openqasm_name) * 4))

    def test_get_qasms(self):
        """Test the get_qasms with int names. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(10, 3)
        cr = q_program.create_classical_register(20, 3)
        qc1 = q_program.create_circuit(101, [qr], [cr])
        qc2 = q_program.create_circuit(102, [qr], [cr])
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[1], qr[2])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc1.measure(qr[2], cr[2])
        qc2.h(qr)
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        result = q_program.get_qasms([101, 102])
        self.assertEqual(len(result[0]), (147 +
                                          len(qr.openqasm_name) * 9 +
                                          len(cr.openqasm_name) * 4))
        self.assertEqual(len(result[1]), (137 +
                                          len(qr.openqasm_name) * 7 +
                                          len(cr.openqasm_name) * 4))

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates. Names are ints.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        qc = q_program.get_circuit(0)
        qr = q_program.get_quantum_register(0)
        cr = q_program.get_classical_register("")
        qc.u1(0.3, qr[0])
        qc.u2(0.2, 0.1, qr[1])
        qc.u3(0.3, 0.2, 0.1, qr[2])
        qc.s(qr[1])
        qc.s(qr[2]).inverse()
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.x(qr[2]).c_if(cr, 0)
        qc.y(qr[2]).c_if(cr, 1)
        qc.z(qr[2]).c_if(cr, 2)
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm(0)
        self.assertEqual(len(result), (385 +
                                       len(qr.openqasm_name) * 23 +
                                       len(cr.openqasm_name) * 7))

    ###############################################################
    # Test for compile when names are integers
    ###############################################################

    def test_compile_program(self):
        """Test compile_program. Names are integers
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        qc = q_program.get_circuit(0)
        qr = q_program.get_quantum_register(0)
        cr = q_program.get_classical_register("")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        out = q_program.compile([0], backend=backend,
                                coupling_map=coupling_map, qobj_id='cooljob')
        self.log.info(out)
        self.assertEqual(len(out), 3)

    def test_get_execution_list(self):
        """Test get_execution_list with int names.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        qc = q_program.get_circuit(0)
        qr = q_program.get_quantum_register(0)
        cr = q_program.get_classical_register("")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = q_program.compile([0], backend=backend,
                                 coupling_map=coupling_map, qobj_id='cooljob')
        result = q_program.get_execution_list(qobj, print_func=self.log.info)
        self.log.info(result)
        self.assertEqual(result, [0])

    def test_change_circuit_qobj_after_compile(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_ZEROS)
        qr = q_program.get_quantum_register(0)
        cr = q_program.get_classical_register("")
        qc2 = q_program.create_circuit(102, [qr], [cr])
        qc3 = q_program.create_circuit(103, [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = [102, 103]
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals': [1, 2, 3, 4]}
        qobj1 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        qobj1['circuits'][0]['config']['shots'] = 50
        qobj1['circuits'][0]['config']['xvals'] = [1, 1, 1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        self.assertTrue(qobj1['circuits'][0]['config']['shots'] == 50)
        self.assertTrue(qobj1['circuits'][1]['config']['shots'] == 1)
        self.assertTrue(qobj1['circuits'][0]['config']['xvals'] == [1, 1, 1])
        self.assertTrue(qobj1['circuits'][1]['config']['xvals'] == [1, 2, 3, 4])
        self.assertTrue(qobj1['config']['shots'] == 1024)
        self.assertTrue(qobj2['circuits'][0]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][1]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][0]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2['circuits'][1]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])

    def test_add_circuit(self):
        """Test add two circuits with zero names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(0, 2)
        cr = q_program.create_classical_register("", 2)
        qc1 = q_program.create_circuit(0, [qr], [cr])
        qc2 = q_program.create_circuit("", [qr], [cr])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        q_program.add_circuit(1001, new_circuit)
        circuits = [1001]
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = q_program.execute(circuits, backend=backend, shots=shots, seed=78)
        counts = result.get_counts(1001)
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.025 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


class TestIntegerIds(QiskitTestCase):
    """Circuits and records can have integers as names"""

    def setUp(self):
        self.QPS_SPECS_INT = {
            "circuits": [{
                "name": 1,
                "quantum_registers": [{
                    "name": 40,
                    "size": 3}],
                "classical_registers": [{
                    "name": 50,
                    "size": 3}]
            }]
        }

    ###############################################################
    # Tests to initiate an build a quantum program with integer ids
    ###############################################################

    def test_create_program_with_specs(self):
        """Test Quantum Object Factory creation using Specs definition
        object with int names for circuit nor records.
        """
        result = QuantumProgram(specs=self.QPS_SPECS_INT)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_classical_register(self):
        """Test create_classical_register with int name
        """
        q_program = QuantumProgram()
        cr = q_program.create_classical_register(42, 3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        """Test create_quantum_register with int name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(32, 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_classical_register_name(self):
        """Test duplicated create_quantum_register with int as names.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register(2, 3)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitError,
                          q_program.create_classical_register, 2, 2)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same int name and size.

        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(1, 3)
        qr2 = q_program.create_quantum_register(1, 3)
        self.assertIs(qr1, qr2)

    def test_create_classical_register_same(self):
        """Test create_classical_register of same int name and size.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register(2, 3)
        cr2 = q_program.create_classical_register(2, 3)
        self.assertIs(cr1, cr2)

    def test_create_classical_registers(self):
        """Test create_classical_registers with int name.
        """
        q_program = QuantumProgram()
        classical_registers = [{"name": 1, "size": 4},
                               {"name": 2, "size": 2}]
        crs = q_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers(self):
        """Test create_quantum_registers with int names
        """
        q_program = QuantumProgram()
        quantum_registers = [{"name": 1, "size": 4},
                             {"name": 2, "size": 2}]
        qrs = q_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_destroy_classical_register(self):
        """Test destroy_classical_register with int name."""
        q_program = QuantumProgram()
        _ = q_program.create_classical_register(1, 3)
        self.assertIn(1, q_program.get_classical_register_names())
        q_program.destroy_classical_register(1)
        self.assertNotIn(1, q_program.get_classical_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_classical_register(1)
        self.assertIn('Not present', str(context.exception))

    def test_destroy_quantum_register(self):
        """Test destroy_quantum_register with int name."""
        q_program = QuantumProgram()
        _ = q_program.create_quantum_register(1, 3)
        self.assertIn(1, q_program.get_quantum_register_names())
        q_program.destroy_quantum_register(1)
        self.assertNotIn(1, q_program.get_quantum_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_quantum_register(1)
        self.assertIn('Not present', str(context.exception))

    def test_create_circuit(self):
        """Test create_circuit with int names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(1, 3)
        cr = q_program.create_classical_register(2, 3)
        qc = q_program.create_circuit(3, [qr], [cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits(self):
        """Test create_circuit with several inputs with int names.
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(10, 3)
        cr1 = q_program.create_classical_register(20, 3)
        qr2 = q_program.create_quantum_register(11, 3)
        cr2 = q_program.create_classical_register(21, 3)
        qc1 = q_program.create_circuit(30, [qr1], [cr1])
        qc2 = q_program.create_circuit(31, [qr2], [cr2])
        qc3 = q_program.create_circuit(32, [qr1, qr2], [cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_destroy_circuit(self):
        """Test destroy_circuit with an int name."""
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(2, 3)
        cr = q_program.create_classical_register(1, 3)
        _ = q_program.create_circuit(10, [qr], [cr])
        self.assertIn(10, q_program.get_circuit_names())
        q_program.destroy_circuit(10)
        self.assertNotIn(10, q_program.get_circuit_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_circuit(10)
        self.assertIn('Not present', str(context.exception))

    def test_get_register_and_circuit_names(self):
        """Get the names of the circuits and registers when their names are ints.
        """
        qr1n = 10
        qr2n = 11
        cr1n = 12
        cr2n = 13
        qc1n = 14
        qc2n = 15
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(qr1n, 3)
        cr1 = q_program.create_classical_register(cr1n, 3)
        qr2 = q_program.create_quantum_register(qr2n, 3)
        cr2 = q_program.create_classical_register(cr2n, 3)
        q_program.create_circuit(qc1n, [qr1], [cr1])
        q_program.create_circuit(qc2n, [qr2], [cr2])
        q_program.create_circuit(qc2n, [qr1, qr2], [cr1, cr2])
        qrn = q_program.get_quantum_register_names()
        crn = q_program.get_classical_register_names()
        qcn = q_program.get_circuit_names()
        self.assertCountEqual(qrn, [qr1n, qr2n])
        self.assertCountEqual(crn, [cr1n, cr2n])
        self.assertCountEqual(qcn, [qc1n, qc2n])

    def test_get_qasm(self):
        """Test the get_qasm with int name. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_INT)
        qc = q_program.get_circuit(1)
        qr = q_program.get_quantum_register(40)
        cr = q_program.get_classical_register(50)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm(1)
        self.assertEqual(len(result), (147 +
                                       len(qr.openqasm_name) * 9 +
                                       len(cr.openqasm_name) * 4))

    def test_get_qasms(self):
        """Test the get_qasms with int names. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(10, 3)
        cr = q_program.create_classical_register(20, 3)
        qc1 = q_program.create_circuit(101, [qr], [cr])
        qc2 = q_program.create_circuit(102, [qr], [cr])
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[1], qr[2])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc1.measure(qr[2], cr[2])
        qc2.h(qr)
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        result = q_program.get_qasms([101, 102])
        self.assertEqual(len(result[0]), (147 +
                                          len(qr.openqasm_name) * 9 +
                                          len(cr.openqasm_name) * 4))
        self.assertEqual(len(result[1]), (137 +
                                          len(qr.openqasm_name) * 7 +
                                          len(cr.openqasm_name) * 4))

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates. Names are ints.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_INT)
        qc = q_program.get_circuit(1)
        qr = q_program.get_quantum_register(40)
        cr = q_program.get_classical_register(50)
        qc.u1(0.3, qr[0])
        qc.u2(0.2, 0.1, qr[1])
        qc.u3(0.3, 0.2, 0.1, qr[2])
        qc.s(qr[1])
        qc.s(qr[2]).inverse()
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.x(qr[2]).c_if(cr, 0)
        qc.y(qr[2]).c_if(cr, 1)
        qc.z(qr[2]).c_if(cr, 2)
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm(1)
        self.assertEqual(len(result), (385 +
                                       len(qr.openqasm_name) * 23 +
                                       len(cr.openqasm_name) * 7))

    ###############################################################
    # Test for compile when names are integers
    ###############################################################

    def test_compile_program(self):
        """Test compile_program. Names are integers
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_INT)
        qc = q_program.get_circuit(1)
        qr = q_program.get_quantum_register(40)
        cr = q_program.get_classical_register(50)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        out = q_program.compile([1], backend=backend,
                                coupling_map=coupling_map, qobj_id='cooljob')
        self.log.info(out)
        self.assertEqual(len(out), 3)

    def test_get_execution_list(self):
        """Test get_execution_list with int names.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_INT)
        qc = q_program.get_circuit(1)
        qr = q_program.get_quantum_register(40)
        cr = q_program.get_classical_register(50)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = q_program.compile([1], backend=backend,
                                 coupling_map=coupling_map, qobj_id='cooljob')
        result = q_program.get_execution_list(qobj, print_func=self.log.info)
        self.log.info(result)
        self.assertEqual(result, [1])

    def test_change_circuit_qobj_after_compile(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_INT)
        qr = q_program.get_quantum_register(40)
        cr = q_program.get_classical_register(50)
        qc2 = q_program.create_circuit(102, [qr], [cr])
        qc3 = q_program.create_circuit(103, [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = [102, 103]
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals': [1, 2, 3, 4]}
        qobj1 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        qobj1['circuits'][0]['config']['shots'] = 50
        qobj1['circuits'][0]['config']['xvals'] = [1, 1, 1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        self.assertTrue(qobj1['circuits'][0]['config']['shots'] == 50)
        self.assertTrue(qobj1['circuits'][1]['config']['shots'] == 1)
        self.assertTrue(qobj1['circuits'][0]['config']['xvals'] == [1, 1, 1])
        self.assertTrue(qobj1['circuits'][1]['config']['xvals'] == [1, 2, 3, 4])
        self.assertTrue(qobj1['config']['shots'] == 1024)
        self.assertTrue(qobj2['circuits'][0]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][1]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][0]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2['circuits'][1]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])

    def test_add_circuit(self):
        """Test add two circuits with int names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(1, 2)
        cr = q_program.create_classical_register(2, 2)
        qc1 = q_program.create_circuit(10, [qr], [cr])
        qc2 = q_program.create_circuit(20, [qr], [cr])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        q_program.add_circuit(1001, new_circuit)
        # new_circuit.measure(qr[0], cr[0])
        circuits = [1001]
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=78)
        counts = result.get_counts(1001)
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.025 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


class TestTupleIds(QiskitTestCase):
    """Circuits and records can have tuples as names"""

    def setUp(self):
        self.QPS_SPECS_TUPLE = {
            "circuits": [{
                "name": (1.1, 1j),
                "quantum_registers": [{
                    "name": (40.1, 40j),
                    "size": 3}],
                "classical_registers": [{
                    "name": (50.1, 50j),
                    "size": 3}]
            }]
        }

    ###############################################################
    # Tests to initiate an build a quantum program with tuple ids
    ###############################################################

    def test_create_program_with_specs(self):
        """Test Quantum Object Factory creation using Specs definition
        object with tuple names for circuit nor records.
        """
        result = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_classical_register(self):
        """Test create_classical_register with tuple name
        """
        q_program = QuantumProgram()
        cr = q_program.create_classical_register((50.1, 50j), 3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        """Test create_quantum_register with tuple name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register((32.1, 32j), 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_classical_register_name(self):
        """Test duplicated create_quantum_register with int as names.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register((2.1, 2j), 3)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitError,
                          q_program.create_classical_register, (2.1, 2j), 2)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same tuple name and size.

        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register((1.1, 1j), 3)
        qr2 = q_program.create_quantum_register((1.1, 1j), 3)
        self.assertIs(qr1, qr2)

    def test_create_classical_register_same(self):
        """Test create_classical_register of same tuple name and size.
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register((2.1, 2j), 3)
        cr2 = q_program.create_classical_register((2.1, 2j), 3)
        self.assertIs(cr1, cr2)

    def test_create_classical_registers(self):
        """Test create_classical_registers with tuple name.
        """
        q_program = QuantumProgram()
        classical_registers = [{"name": (1.1, 1j), "size": 4},
                               {"name": (2.1, 2j), "size": 2}]
        crs = q_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers(self):
        """Test create_quantum_registers with tuple names
        """
        q_program = QuantumProgram()
        quantum_registers = [{"name": (1.1, 1j), "size": 4},
                             {"name": (2.1, 2j), "size": 2}]
        qrs = q_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_destroy_classical_register(self):
        """Test destroy_classical_register with tuple name."""
        q_program = QuantumProgram()
        _ = q_program.create_classical_register((1.1, 1j), 3)
        self.assertIn((1.1, 1j), q_program.get_classical_register_names())
        q_program.destroy_classical_register((1.1, 1j))
        self.assertNotIn((1.1, 1j), q_program.get_classical_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_classical_register((1.1, 1j))
        self.assertIn('Not present', str(context.exception))

    def test_destroy_quantum_register(self):
        """Test destroy_quantum_register with tuple name."""
        q_program = QuantumProgram()
        _ = q_program.create_quantum_register((1.1, 1j), 3)
        self.assertIn((1.1, 1j), q_program.get_quantum_register_names())
        q_program.destroy_quantum_register((1.1, 1j))
        self.assertNotIn((1.1, 1j), q_program.get_quantum_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_quantum_register((1.1, 1j))
        self.assertIn('Not present', str(context.exception))

    def test_create_circuit(self):
        """Test create_circuit with tuple names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register((1.1, 1j), 3)
        cr = q_program.create_classical_register((2.1, 2j), 3)
        qc = q_program.create_circuit((3.1, 3j), [qr], [cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits(self):
        """Test create_circuit with several inputs with tuple names.
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register((10.1, 10j), 3)
        cr1 = q_program.create_classical_register((20.1, 20j), 3)
        qr2 = q_program.create_quantum_register((11.1, 11j), 3)
        cr2 = q_program.create_classical_register((21.1, 21j), 3)
        qc1 = q_program.create_circuit((30.1, 30j), [qr1], [cr1])
        qc2 = q_program.create_circuit((31.1, 31j), [qr2], [cr2])
        qc3 = q_program.create_circuit((32.1, 32j), [qr1, qr2], [cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_destroy_circuit(self):
        """Test destroy_circuit with an tuple name."""
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register((2.1, 2j), 3)
        cr = q_program.create_classical_register((1.1, 1j), 3)
        _ = q_program.create_circuit((10.1, 10j), [qr], [cr])
        self.assertIn((10.1, 10j), q_program.get_circuit_names())
        q_program.destroy_circuit((10.1, 10j))
        self.assertNotIn((10.1, 10j), q_program.get_circuit_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_circuit((10.1, 10j))
        self.assertIn('Not present', str(context.exception))

    def test_get_register_and_circuit_names(self):
        """Get the names of the circuits and registers when their names are ints.
        """
        qr1n = (10.1, 10j)
        qr2n = (11.1, 11j)
        cr1n = (12.1, 12j)
        cr2n = (13.1, 13j)
        qc1n = (14.1, 14j)
        qc2n = (15.1, 15j)
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register(qr1n, 3)
        cr1 = q_program.create_classical_register(cr1n, 3)
        qr2 = q_program.create_quantum_register(qr2n, 3)
        cr2 = q_program.create_classical_register(cr2n, 3)
        q_program.create_circuit(qc1n, [qr1], [cr1])
        q_program.create_circuit(qc2n, [qr2], [cr2])
        q_program.create_circuit(qc2n, [qr1, qr2], [cr1, cr2])
        qrn = q_program.get_quantum_register_names()
        crn = q_program.get_classical_register_names()
        qcn = q_program.get_circuit_names()
        self.assertCountEqual(qrn, [qr1n, qr2n])
        self.assertCountEqual(crn, [cr1n, cr2n])
        self.assertCountEqual(qcn, [qc1n, qc2n])

    def test_get_qasm(self):
        """Test the get_qasm with tuple name. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        qc = q_program.get_circuit((1.1, 1j))
        qr = q_program.get_quantum_register((40.1, 40j))
        cr = q_program.get_classical_register((50.1, 50j))
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm((1.1, 1j))
        self.assertEqual(len(qr.openqasm_name) * 9 +
                         len(cr.openqasm_name) * 4 + 147, len(result))

    def test_get_qasms(self):
        """Test the get_qasms with tuple names. They need to be coverted to OpenQASM format.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register((10.1, 10j), 3)
        cr = q_program.create_classical_register((20.1, 20j), 3)
        qc1 = q_program.create_circuit((101.1, 101j), [qr], [cr])
        qc2 = q_program.create_circuit((102.1, 102j), [qr], [cr])
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[1], qr[2])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc1.measure(qr[2], cr[2])
        qc2.h(qr)
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        result = q_program.get_qasms([(101.1, 101j), (102.1, 102j)])
        self.assertEqual(len(qr.openqasm_name) * 9 +
                         len(cr.openqasm_name) * 4 + 147, len(result[0]))
        self.assertEqual(len(qr.openqasm_name) * 7 +
                         len(cr.openqasm_name) * 4 + 137, len(result[1]))

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates. Names are tuples.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        qc = q_program.get_circuit((1.1, 1j))
        qr = q_program.get_quantum_register((40.1, 40j))
        cr = q_program.get_classical_register((50.1, 50j))
        qc.u1(0.3, qr[0])
        qc.u2(0.2, 0.1, qr[1])
        qc.u3(0.3, 0.2, 0.1, qr[2])
        qc.s(qr[1])
        qc.s(qr[2]).inverse()
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.x(qr[2]).c_if(cr, 0)
        qc.y(qr[2]).c_if(cr, 1)
        qc.z(qr[2]).c_if(cr, 2)
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm((1.1, 1j))
        self.assertEqual(len(qr.openqasm_name) * 23 +
                         len(cr.openqasm_name) * 7 + 385, len(result))

    ###############################################################
    # Test for compile when names are tuples
    ###############################################################

    def test_compile_program(self):
        """Test compile_program. Names are tuples
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        qc = q_program.get_circuit((1.1, 1j))
        qr = q_program.get_quantum_register((40.1, 40j))
        cr = q_program.get_classical_register((50.1, 50j))
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        out = q_program.compile([(1.1, 1j)], backend=backend,
                                coupling_map=coupling_map, qobj_id='cooljob')
        self.log.info(out)
        self.assertEqual(len(out), 3)

    def test_get_execution_list(self):
        """Test get_execution_list with tuple names.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        qc = q_program.get_circuit((1.1, 1j))
        qr = q_program.get_quantum_register((40.1, 40j))
        cr = q_program.get_classical_register((50.1, 50j))
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = q_program.compile([(1.1, 1j)], backend=backend,
                                 coupling_map=coupling_map, qobj_id='cooljob')
        result = q_program.get_execution_list(qobj, print_func=self.log.info)
        self.log.info(result)
        self.assertCountEqual(result, [(1.1, 1j)])

    def test_change_circuit_qobj_after_compile(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS_TUPLE)
        qr = q_program.get_quantum_register((40.1, 40j))
        cr = q_program.get_classical_register((50.1, 50j))
        qc2 = q_program.create_circuit((102.1, 102j), [qr], [cr])
        qc3 = q_program.create_circuit((103.1, 103j), [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = [(102.1, 102j), (103.1, 103j)]
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals': [1, 2, 3, 4]}
        qobj1 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        qobj1['circuits'][0]['config']['shots'] = 50
        qobj1['circuits'][0]['config']['xvals'] = [1, 1, 1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        self.assertTrue(qobj1['circuits'][0]['config']['shots'] == 50)
        self.assertTrue(qobj1['circuits'][1]['config']['shots'] == 1)
        self.assertTrue(qobj1['circuits'][0]['config']['xvals'] == [1, 1, 1])
        self.assertTrue(qobj1['circuits'][1]['config']['xvals'] == [1, 2, 3, 4])
        self.assertTrue(qobj1['config']['shots'] == 1024)
        self.assertTrue(qobj2['circuits'][0]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][1]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][0]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2['circuits'][1]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])

    def test_add_circuit(self):
        """Test add two circuits with tuple names.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(1, 2)
        cr = q_program.create_classical_register(2, 2)
        qc1 = q_program.create_circuit((10.1, 10j), [qr], [cr])
        qc2 = q_program.create_circuit((20.1, 20j), [qr], [cr])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        q_program.add_circuit((1001.1, 1001j), new_circuit)
        circuits = [(1001.1, 1001j)]
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=78)
        counts = result.get_counts((1001.1, 1001j))
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.025 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


if __name__ == '__main__':
    unittest.main(verbosity=2)
