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

available_backends = QuantumProgram().available_backends()


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

    def test_create_program_with_specsnonames(self):
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
        qr_name_len = len(qr.name)
        cr_name_len = len(cr.name)
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
        self.assertEqual(len(result), (len(qr.name) * 23 +
                                       len(cr.name) * 7 +
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
        self.assertEqual(len(result), 1)

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

    def test_add_circuit_noname(self):
        """Test add two circuits without names. Also tests get_counts without circuit name.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register(size=2)
        cr = q_program.create_classical_register(size=2)
        qc1 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc2 = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        q_program.add_circuit(quantum_circuit=new_circuit)
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = q_program.execute(backend=backend, shots=shots, seed=78)
        counts = result.get_counts(new_circuit.name)
        target = {'00': shots / 2, '01': shots / 2}
        threshold = 0.025 * shots
        self.assertDictAlmostEqual(counts, target, threshold)
        self.assertRaises(QISKitError, result.get_counts)


class TestQobj(QiskitTestCase):
    """Check the objects compiled for different backends create names properly"""

    def setUp(self):
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr2", 2)
        cr = qp.create_classical_register(None, 2)
        qc = qp.create_circuit("qc0", [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qp.add_circuit("qc10", qc)
        circuits = ["qc10"]
        self.qr_name = qr.name
        self.cr_name = cr.name
        self.qp = qp
        self.circuits = circuits

    def test_local_qasm_simulator(self):
        backend = 'local_qasm_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)

    @unittest.skipIf('local_clifford_simulator' not in available_backends,
                     'local_clifford_simulator is not available')
    def test_local_clifford_simulator(self):
        backend = 'local_clifford_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)

    @unittest.skipIf('local_qiskit_simulator' not in available_backends,
                     'local_qiskit_simulator is not available')
    def test_local_qiskit_simulator(self):
        backend = 'local_qiskit_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)

    def test_local_sympy_qasm_simulator(self):
        backend = 'local_sympy_qasm_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)

    def test_local_sympy_unitary_simulator(self):
        backend = 'local_sympy_unitary_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)

    def test_local_unitary_simulator(self):
        backend = 'local_unitary_simulator'
        qobj = self.qp.compile(self.circuits, backend=backend, shots=1024)
        cc = qobj['circuits'][0]['compiled_circuit']
        ccq = qobj['circuits'][0]['compiled_circuit_qasm']
        self.assertIn(self.qr_name, map(lambda x: x[0], cc['header']['qubit_labels']))
        self.assertIn(self.qr_name, ccq)
        self.assertIn(self.cr_name, map(lambda x: x[0], cc['header']['clbit_labels']))
        self.assertIn(self.cr_name, ccq)


class TestAnonymousIdsNoQuantumProgram(QiskitTestCase):
    """Test the anonymous use of registers.
    TODO: this needs to be expanded, ending up with the rest of the tests
    in the file not using QuantumProgram when it is deprecated.
    """

    def test_create_anonymous_classical_register(self):
        """Test creating a ClassicalRegister with no name.
        """
        cr = ClassicalRegister(size=3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_anonymous_quantum_register(self):
        """Test creating a QuantumRegister with no name.
        """
        qr = QuantumRegister(size=3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_anonymous_classical_registers(self):
        """Test creating several ClassicalRegister with no name.
        """
        cr1 = ClassicalRegister(size=3)
        cr2 = ClassicalRegister(size=3)
        self.assertNotEqual(cr1.name, cr2.name)

    def test_create_anonymous_quantum_registers(self):
        """Test creating several QuantumRegister with no name.
        """
        qr1 = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)
        self.assertNotEqual(qr1.name, qr2.name)

    def test_create_anonymous_mixed_registers(self):
        """Test creating several Registers with no name.
        """
        cr0 = ClassicalRegister(size=3)
        qr0 = QuantumRegister(size=3)
        # Get the current index counte of the registers
        cr_index = int(cr0.name[1:])
        qr_index = int(qr0.name[1:])

        cr1 = ClassicalRegister(size=3)
        _ = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)

        # Check that the counters for each kind are incremented separately.
        cr_current = int(cr1.name[1:])
        qr_current = int(qr2.name[1:])
        self.assertEqual(cr_current, cr_index + 1)
        self.assertEqual(qr_current, qr_index + 2)

    def test_create_circuit_noname(self):
        """Test create_circuit with no name
        """
        q_program = QuantumProgram()
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        qc = q_program.create_circuit(qregisters=[qr], cregisters=[cr])
        self.assertIsInstance(qc, QuantumCircuit)


class TestInvalidIds(QiskitTestCase):
    """Circuits and records with invalid IDs"""

    def test_invalid_type_circuit_name(self):
        """Test create_circuit with invalid type name
        """
        q_program = QuantumProgram()
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        self.assertRaises(QISKitError, q_program.create_circuit,
                          qregisters=[qr], cregisters=[cr], name=1)

    def test_invalid_type_qr(self):
        """Test create_quantum_register with an invalid type name.
        """
        q_program = QuantumProgram()
        self.assertRaises(QISKitError, q_program.create_quantum_register, size=3, name=1)

    def test_invalid_type_cr(self):
        """Test create_classical_register with an invalid type name.
        """
        q_program = QuantumProgram()
        self.assertRaises(QISKitError, q_program.create_classical_register, size=3, name=1)

    def test_invalid_type_qr_spec(self):
        """QPS_SPECS_NONAMES defines a quantum register with an invalid type name
        """
        QPS_SPECS_NONAMES = {
            "circuits": [{
                "quantum_registers": [{
                    "name": 1,
                    "size": 3}],
                "classical_registers": [{
                    "size": 3}]
            }]
        }

        self.assertRaises(QISKitError, QuantumProgram, specs=QPS_SPECS_NONAMES)

    def test_invalid_type_cr_spec(self):
        """QPS_SPECS_NONAMES defines a classical register with an invalid type name
        """
        QPS_SPECS_NONAMES = {
            "circuits": [{
                "quantum_registers": [{
                    "size": 3}],
                "classical_registers": [{
                    "name": 1,
                    "size": 3}]
            }]
        }

        self.assertRaises(QISKitError, QuantumProgram, specs=QPS_SPECS_NONAMES)

    def test_invalid_qasmname_qr(self):
        """Test create_quantum_register with an invalid QASM name (do not start with lowercase).
        """
        q_program = QuantumProgram()
        self.assertRaises(QISKitError, q_program.create_quantum_register, size=3, name='Qr')

    def test_invalid_qasmname_cr(self):
        """Test create_classical_register with an invalid QASM name (do not start with lowercase).
        """
        q_program = QuantumProgram()
        self.assertRaises(QISKitError, q_program.create_classical_register, size=3, name='Cr')

    def test_invalid_qasmname_qr_spec(self):
        """QPS_SPECS_NONAMES defines a quantum register with invalid QASM name (do not start
        with lowercase).
        """
        QPS_SPECS_NONAMES = {
            "circuits": [{
                "quantum_registers": [{
                    "name": 'Qr',
                    "size": 3}],
                "classical_registers": [{
                    "size": 3}]
            }]
        }

        self.assertRaises(QISKitError, QuantumProgram, specs=QPS_SPECS_NONAMES)

    def test_invalid_qasmname_cr_spec(self):
        """QPS_SPECS_NONAMES defines a classical register with invalid QASM name (do not start
        with lowercase).
        """
        QPS_SPECS_NONAMES = {
            "circuits": [{
                "quantum_registers": [{
                    "size": 3}],
                "classical_registers": [{
                    "name": "Cr",
                    "size": 3}]
            }]
        }

        self.assertRaises(QISKitError, QuantumProgram, specs=QPS_SPECS_NONAMES)


if __name__ == '__main__':
    unittest.main(verbosity=2)
