# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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

"""
Quantum Program QISKit Test

Authors: Ismael Faro <Ismael.Faro1@ibm.com>
         Jesus Perez <jesusper@us.ibm.com>
         Jay Gambetta
"""

import sys
import os
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QISKitException


QASM_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../examples/qasm/simple8qbit.qasm')


# We need the environment variable for Travis.
try:
    # We don't know from where the user is running the example,
    # so we need a relative position from this file path.
    # TODO: Relative imports for intra-package imports are highly discouraged.
    # http://stackoverflow.com/a/7506006
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import Qconfig
    API_TOKEN = Qconfig.APItoken
    # TODO: Why "APItoken" is in the root (the unique) and "url" inside "config"?
    # (also unique) -> make it consistent.
    URL = Qconfig.config["url"]
except ImportError:
    API_TOKEN = os.environ["QE_TOKEN"]
    URL = os.environ["QE_URL"]


# Define Program Specifications.
QPS_SPECS = {
    "circuits": [{
        "name": "circuitName",
        "quantum_registers": [{
            "name": "qname",
            "size": 3}],
        "classical_registers": [{
            "name": "cname",
            "size": 3}]
    }]
}


class TestQuantumProgram(unittest.TestCase):
    """QISKIT QuatumProgram Object Tests."""

    ###############################################################
    # Tests to initiate an build a quantum program
    ###############################################################

    def test_create_program_with_specs(self):
        """Test Quantum Object Factory creation using Specs deffinition object.

        If all is correct we get a object intstance of QuantumProgram

        Previusly:
            Objects:
                QPS_SPECS
            Libraries:
                from qiskit import QuantumProgram

        """
        result = QuantumProgram(specs=QPS_SPECS)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_program(self):
        """Test Quantum Object Factory creation Without Specs deffinition object.

        If all is correct we get a object intstance of QuantumProgram

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        result = QuantumProgram()
        self.assertIsInstance(result, QuantumProgram)

    def test_config_scripts_file(self):
        """Test Qconfig.

        in this case we check if the URL API is defined.

        Previusly:
            Libraries:
                import Qconfig
        """
        self.assertEqual(
            URL,
            "https://quantumexperience.ng.bluemix.net/api")

    def test_create_classical_register(self):
        """Test create_classical_register.

        If all is correct we get a object intstance of ClassicalRegister

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        QP_program = QuantumProgram()
        cr = QP_program.create_classical_register("cr", 3, verbose=False)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        """Test create_quantum_register.

        If all is correct we get a object intstance of QuantumRegister

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 3, verbose=False)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_quantum_register(self):
        """Test create_quantum_register.

        If all is correct we get a object intstance of QuantumRegister and
        QISKitException

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitException
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr", 3, verbose=False)
        self.assertIsInstance(qr1, QuantumRegister)
        self.assertRaises(QISKitException, QP_program.create_quantum_register,
                          "qr", 2, verbose=False)

    def test_fail_create_classical_register(self):
        """Test create_quantum_register.

        If all is correct we get a object intstance of QuantumRegister and
        QISKitException

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitException
        """
        QP_program = QuantumProgram()
        cr1 = QP_program.create_classical_register("cr", 3, verbose=False)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitException,
                          QP_program.create_classical_register, "cr", 2,
                          verbose=False)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same name and size.

        If all is correct we get a single classical register

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr", 3, verbose=False)
        qr2 = QP_program.create_quantum_register("qr", 3, verbose=False)
        self.assertIs(qr1, qr2)

    def test_create_classical_register_same(self):
        """Test create_classical_register of same name and size.

        If all is correct we get a single classical register

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        QP_program = QuantumProgram()
        cr1 = QP_program.create_classical_register("cr", 3, verbose=False)
        cr2 = QP_program.create_classical_register("cr", 3, verbose=False)
        self.assertIs(cr1, cr2)

    def test_create_classical_registers(self):
        """Test create_classical_registers.

        If all is correct we get a object intstance of list[ClassicalRegister]

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        QP_program = QuantumProgram()
        classical_registers = [{"name": "c1", "size": 4},
                               {"name": "c2", "size": 2}]
        crs = QP_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers(self):
        """Test create_quantum_registers.

        If all is correct we get a object intstance of list[QuantumRegister]

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        QP_program = QuantumProgram()
        quantum_registers = [{"name": "q1", "size": 4},
                             {"name": "q2", "size": 2}]
        qrs = QP_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_create_circuit(self):
        """Test create_circuit.

        If all is correct we get a object intstance of QuantumCircuit

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumCircuit
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 3, verbose=False)
        cr = QP_program.create_classical_register("cr", 3, verbose=False)
        qc = QP_program.create_circuit("qc", [qr], [cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits(self):
        """Test create_circuit with several inputs.

        If all is correct we get a object intstance of QuantumCircuit

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumCircuit
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr1", 3, verbose=False)
        cr1 = QP_program.create_classical_register("cr1", 3, verbose=False)
        qr2 = QP_program.create_quantum_register("qr2", 3, verbose=False)
        cr2 = QP_program.create_classical_register("cr2", 3, verbose=False)
        qc1 = QP_program.create_circuit("qc1", [qr1], [cr1])
        qc2 = QP_program.create_circuit("qc2", [qr2], [cr2])
        qc3 = QP_program.create_circuit("qc2", [qr1, qr2], [cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_load_qasm_file(self):
        """Test load_qasm_file and get_circuit.

        If all is correct we should get the qasm file loaded in QASM_FILE_PATH

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram()
        name = QP_program.load_qasm_file(QASM_FILE_PATH, name="",
                                         verbose=False)
        result = QP_program.get_circuit(name)
        to_check = result.qasm()
        self.assertEqual(len(to_check), 1767)

    def test_fail_load_qasm_file(self):
        """Test fail_load_qasm_file.

        If all is correct we should get a QISKitException

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QISKitException
        """
        QP_program = QuantumProgram()
        self.assertRaises(QISKitException,
                          QP_program.load_qasm_file, "", name=None,
                          verbose=False)

    def test_load_qasm_text(self):
        """Test load_qasm_text and get_circuit.

        If all is correct we should get the qasm file loaded from the string

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram()
        QASM_string = "// A simple 8 qubit example\nOPENQASM 2.0;\n"
        QASM_string += "include \"qelib1.inc\";\nqreg a[4];\n"
        QASM_string += "qreg b[4];\ncreg c[4];\ncreg d[4];\nh a;\ncx a, b;\n"
        QASM_string += "barrier a;\nbarrier b;\nmeasure a[0]->c[0];\n"
        QASM_string += "measure a[1]->c[1];\nmeasure a[2]->c[2];\n"
        QASM_string += "measure a[3]->c[3];\nmeasure b[0]->d[0];\n"
        QASM_string += "measure b[1]->d[1];\nmeasure b[2]->d[2];\n"
        QASM_string += "measure b[3]->d[3];"
        name = QP_program.load_qasm_text("", QASM_string, verbose=False)
        result = QP_program.get_circuit(name)
        to_check = result.qasm()
        self.assertEqual(len(to_check), 1767)

    def test_get_register_and_circuit(self):
        """Test get_quantum_registers, get_classical_registers, and get_circuit.

        If all is correct we get a object intstance of QuantumCircuit,
        QuantumRegister, ClassicalRegister

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertIsInstance(qr, QuantumRegister)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_get_register_and_circuit_names(self):
        """Get the names of the circuits and registers.

        If all is correct we should get the arrays of the names

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr1", 3, verbose=False)
        cr1 = QP_program.create_classical_register("cr1", 3, verbose=False)
        qr2 = QP_program.create_quantum_register("qr2", 3, verbose=False)
        cr2 = QP_program.create_classical_register("cr2", 3, verbose=False)
        QP_program.create_circuit("qc1", [qr1], [cr1])
        QP_program.create_circuit("qc2", [qr2], [cr2])
        QP_program.create_circuit("qc2", [qr1, qr2], [cr1, cr2])
        qrn = QP_program.get_quantum_register_names()
        crn = QP_program.get_classical_register_names()
        qcn = QP_program.get_circuit_names()
        self.assertEqual(qrn, ['qr1', 'qr2'])
        self.assertEqual(crn, ['cr1', 'cr2'])
        self.assertEqual(qcn, ['qc1', 'qc2'])

    def test_get_qasm(self):
        """Test the get_qasm.

        If all correct the qasm output should be of a certain lenght

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = QP_program.get_qasm("circuitName")
        self.assertEqual(len(result), 212)

    def test_get_qasms(self):
        """Test the get_qasms.

        If all correct the qasm output for each circuit should be of a certain
        lenght

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 3, verbose=False)
        cr = QP_program.create_classical_register("cr", 3, verbose=False)
        qc1 = QP_program.create_circuit("qc1", [qr], [cr])
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
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
        result = QP_program.get_qasms(["qc1", "qc2"])
        self.assertEqual(len(result[0]), 173)
        self.assertEqual(len(result[1]), 159)

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates.

        If all correct the qasm output should be of a certain lenght

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
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
        result = QP_program.get_qasm('circuitName')
        self.assertEqual(len(result), 535)

    def test_get_initial_circuit(self):
        """Test get_initial_circuit.

        If all correct is should be of the circuit form.

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_initial_circuit()
        self.assertIsInstance(qc, QuantumCircuit)

    @unittest.skip
    def test_contact_multiple_vertical_circuits(self):
        pass

    ###############################################################
    # Tests for working with backends
    ###############################################################

    def test_setup_api(self):
        """Check the api is set up.

        If all correct is should be true.
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        self.assertTrue(QP_program.set_api(API_TOKEN, URL))

    def test_available_backends_exist(self):
        """Test if there are available backends.

        If all correct some should exists (even if offline).
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        available_backends = QP_program.available_backends()
        self.assertTrue(available_backends)

    def test_local_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists (even if ofline).
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        local_backends = QP_program.local_backends()
        self.assertTrue(local_backends)

    def test_online_backends_exist(self):
        """Test if there are online backends.

        If all correct some should exists.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        online_backends = QP_program.online_backends()
        # print(online_backends)
        self.assertTrue(online_backends)

    def test_online_devices(self):
        """Test if there are online backends (which are devices).

        If all correct some should exists. NEED internet connection for this.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        qp = QuantumProgram(specs=QPS_SPECS)
        qp.set_api(API_TOKEN, URL)
        online_devices = qp.online_devices()
        # print(online_devices)
        self.assertTrue(isinstance(online_devices, list))

    def test_online_simulators(self):
        """Test if there are online backends (which are simulators).

        If all correct some should exists. NEED internet connection for this.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        qp = QuantumProgram(specs=QPS_SPECS)
        qp.set_api(API_TOKEN, URL)
        online_simulators = qp.online_simulators()
        # print(online_simulators)
        self.assertTrue(isinstance(online_simulators, list))


    def test_backend_status(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        backend_list = QP_program.online_backends()
        if backend_list:
            backend = backend_list[0]
        result = QP_program.get_backend_status(backend)
        self.assertIn(result['available'], [True, False])

    def test_get_backend_configuration(self):
        qp = QuantumProgram(specs=QPS_SPECS)
        qp.set_api(API_TOKEN, URL)
        backend_list = qp.available_backends()
        for backend in backend_list:
            qp.get_backend_configuration(backend)

    def test_get_backend_configuration_fail(self):
        qp = QuantumProgram(specs=QPS_SPECS)
        qp.set_api(API_TOKEN, URL)
        backend = 'fail'
        self.assertRaises(LookupError, qp.get_backend_configuration, backend)

    def test_backend_calibration(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        backend_list = QP_program.online_backends()
        if backend_list:
            backend = backend_list[0]
        result = QP_program.get_backend_calibration(backend)
        self.assertEqual(len(result), 4)





    def test_execute_one_circuit_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[1])
        qc.measure(qr[0], cr[0])
        shots = 1  # the number of shots in the experiment.
        QP_program.set_api(API_TOKEN, URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(
            ['circuitName'], backend=backend, shots=shots, max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_several_circuits_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']
        shots = 1  # the number of shots in the experiment.
        QP_program.set_api(API_TOKEN, URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(
            circuits, backend=backend, shots=shots, max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_program_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        shots = 1  # the number of shots in the experiment.
        QP_program.set_api(API_TOKEN, URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(['qc2'], backend=backend, shots=shots,
                                    max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_one_circuit_real_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc = QP_program.get_circuit("circuitName")
        qc.h(qr[1])
        qc.measure(qr[0], cr[0])
        backend = None
        backend_list = QP_program.online_backends()
        if backend_list:
            backend = backend_list[0]
        shots = 1  # the number of shots in the experiment.
        QP_program.set_api(API_TOKEN, URL)
        result = QP_program.execute(['circuitName'], backend=backend,
                                    shots=shots, max_credits=3)
        self.assertIn(result["status"], ["COMPLETED", "Error"])

    def test_compile_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        backend = 'test'
        coupling_map = None
        QP_program.compile(['circuitName'], backend=backend,
                           coupling_map=coupling_map)
        to_test = QP_program.get_circuit('circuitName')
        self.assertEqual(len(to_test.qasm()), 120)

    def test_run_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = None
        QP_program.set_api(API_TOKEN, URL)
        backend = QP_program.online_simulators()[0]
        QP_program.compile(circuits, backend=backend, shots=shots,
                           max_credits=max_credits, coupling_map=coupling_map)
        result = QP_program.run()
        # TODO: Revire result
        self.assertEqual(result['status'], 'COMPLETED')

    def test_execute_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']

        shots = 1024  # the number of shots in the experiment.
        QP_program.set_api(API_TOKEN, URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    max_credits=3)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_local_qasm_simulator(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_local_qasm_simulator_one_shot(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots)
        print(QP_program.get_qasms(['qc2', 'qc3']))
        self.assertEqual(result['status'], 'COMPLETED')

    def test_local_unitary_simulator(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        circuits = ['qc2', 'qc3']
        backend = 'local_unitary_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots)
        # print(result)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_new_compile(self):
        QP_program = QuantumProgram()
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = None
        QP_program.load_qasm_file(QASM_FILE_PATH, name="circuit-dev")
        circuits = ["circuit-dev"]
        QP_program.compile(circuits, backend=backend, shots=shots,
                           max_credits=max_credits, coupling_map=coupling_map)
        to_check = QP_program.get_circuit("circuit-dev")
        self.assertEqual(len(to_check.qasm()), 1767)

    def test_new_run(self):
        QP_program = QuantumProgram()
        QP_program.set_api(API_TOKEN, URL)
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = None
        QP_program.load_qasm_file(QASM_FILE_PATH, name="circuit-dev")
        circuits = ["circuit-dev"]
        QP_program.compile(circuits, backend=backend, shots=shots,
                           max_credits=max_credits, coupling_map=coupling_map)
        result = QP_program.run()
        self.assertEqual(result['status'], 'COMPLETED')

    def test_new_execute(self):
        QP_program = QuantumProgram()
        QP_program.set_api(API_TOKEN, URL)
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = None
        QP_program.load_qasm_file(QASM_FILE_PATH, "circuit-dev")
        circuits = ["circuit-dev"]
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    max_credits=max_credits,
                                    coupling_map=coupling_map)
        self.assertEqual(result['status'], "COMPLETED")

    def test_add_circuit(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        QP_program.set_api(API_TOKEN, URL)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[1])
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        new_circuit = qc2 + qc3
        QP_program.add_circuit('new_circuit', new_circuit)
        # new_circuit.measure(qr[0], cr[0])
        circuits = ['new_circuit']
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots)
        self.assertEqual(result['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
