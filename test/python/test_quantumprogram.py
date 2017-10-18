# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

"""Quantum Program QISKit Test."""

import os
import unittest

import numpy as np
from qiskit import (ClassicalRegister, QISKitError, QuantumCircuit,
                    QuantumRegister, QuantumProgram, Result,
                    RegisterSizeError)
import qiskit.backends

from .common import QiskitTestCase, TRAVIS_FORK_PULL_REQUEST, Path


# We need the environment variable for Travis.
try:
    import Qconfig

    QE_TOKEN = Qconfig.APItoken
    # TODO: Why "APItoken" is in the root (the unique) and
    # "url" inside "config"?
    # (also unique) -> make it consistent.
    QE_URL = Qconfig.config["url"]
except ImportError:
    if 'QE_TOKEN' in os.environ and 'QE_URL' in os.environ:
        QE_TOKEN = os.environ["QE_TOKEN"]
        QE_URL = os.environ["QE_URL"]


class TestQuantumProgram(QiskitTestCase):
    """QISKIT QuatumProgram Object Tests."""

    def setUp(self):
        self.QASM_FILE_PATH = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)
        self.QASM_FILE_PATH_2 = self._get_resource_path(
            'qasm/plaquette_check.qasm', Path.EXAMPLES)

        self.QPS_SPECS = {
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
        result = QuantumProgram(specs=self.QPS_SPECS)
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

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_config_scripts_file(self):
        """Test Qconfig.

        in this case we check if the QE_URL API is defined.

        Previusly:
            Libraries:
                import Qconfig
        """
        self.assertEqual(
            QE_URL,
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
        cr = QP_program.create_classical_register("cr", 3)
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
        qr = QP_program.create_quantum_register("qr", 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_quantum_register(self):
        """Test create_quantum_register.

        If all is correct we get a object intstance of QuantumRegister and
        QISKitError

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitError
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr", 3)
        self.assertIsInstance(qr1, QuantumRegister)
        self.assertRaises(QISKitError, QP_program.create_quantum_register,
                          "qr", 2)

    def test_fail_create_classical_register(self):
        """Test create_quantum_register.

        If all is correct we get a object intstance of QuantumRegister and
        QISKitError

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitError
        """
        QP_program = QuantumProgram()
        cr1 = QP_program.create_classical_register("cr", 3)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitError,
                          QP_program.create_classical_register, "cr", 2)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same name and size.

        If all is correct we get a single classical register

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        QP_program = QuantumProgram()
        qr1 = QP_program.create_quantum_register("qr", 3)
        qr2 = QP_program.create_quantum_register("qr", 3)
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
        cr1 = QP_program.create_classical_register("cr", 3)
        cr2 = QP_program.create_classical_register("cr", 3)
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
        qr = QP_program.create_quantum_register("qr", 3)
        cr = QP_program.create_classical_register("cr", 3)
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
        qr1 = QP_program.create_quantum_register("qr1", 3)
        cr1 = QP_program.create_classical_register("cr1", 3)
        qr2 = QP_program.create_quantum_register("qr2", 3)
        cr2 = QP_program.create_classical_register("cr2", 3)
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
        name = QP_program.load_qasm_file(self.QASM_FILE_PATH, name="")
        result = QP_program.get_circuit(name)
        to_check = result.qasm()
        self.log.info(to_check)
        self.assertEqual(len(to_check), 554)

    def test_fail_load_qasm_file(self):
        """Test fail_load_qasm_file.

        If all is correct we should get a QISKitError

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QISKitError
        """
        QP_program = QuantumProgram()
        self.assertRaises(QISKitError,
                          QP_program.load_qasm_file, "", name=None)

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
        name = QP_program.load_qasm_text(QASM_string)
        result = QP_program.get_circuit(name)
        to_check = result.qasm()
        self.log.info(to_check)
        self.assertEqual(len(to_check), 554)

    def test_get_register_and_circuit(self):
        """Test get_quantum_registers, get_classical_registers, and get_circuit.

        If all is correct we get a object intstance of QuantumCircuit,
        QuantumRegister, ClassicalRegister

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
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
        qr1 = QP_program.create_quantum_register("qr1", 3)
        cr1 = QP_program.create_classical_register("cr1", 3)
        qr2 = QP_program.create_quantum_register("qr2", 3)
        cr2 = QP_program.create_classical_register("cr2", 3)
        QP_program.create_circuit("qc1", [qr1], [cr1])
        QP_program.create_circuit("qc2", [qr2], [cr2])
        QP_program.create_circuit("qc2", [qr1, qr2], [cr1, cr2])
        qrn = QP_program.get_quantum_register_names()
        crn = QP_program.get_classical_register_names()
        qcn = QP_program.get_circuit_names()
        self.assertEqual(qrn, {'qr1', 'qr2'})
        self.assertEqual(crn, {'cr1', 'cr2'})
        self.assertEqual(qcn, {'qc1', 'qc2'})

    def test_get_qasm(self):
        """Test the get_qasm.

        If all correct the qasm output should be of a certain lenght

        Previusly:
            Libraries:
                from qiskit import QuantumProgram
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
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
        qr = QP_program.create_quantum_register("qr", 3)
        cr = QP_program.create_classical_register("cr", 3)
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
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
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
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = QP_program.get_initial_circuit()
        self.assertIsInstance(qc, QuantumCircuit)

    def test_save(self):
        """Test save.

        Save a Quantum Program in Json file
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)

        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")

        qc.u3(0.3, 0.2, 0.1, qr[0])
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.z(qr[2]).c_if(cr, 1)
        qc.x(qr[2]).c_if(cr, 1)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        result = QP_program.save(self._get_resource_path('test_save.json'),
                                 beauty=True)

        self.assertEqual(result['status'], 'Done')

    def test_save_wrong(self):
        """Test save wrong.

        Save a Quantum Program in Json file: Errors Control
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, QP_program.load)

    def test_load(self):
        """Test load Json.

        Load a Json Quantum Program
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)

        result = QP_program.load(self._get_resource_path('test_load.json'))
        self.assertEqual(result['status'], 'Done')

        check_result = QP_program.get_qasm('circuitName')
        self.assertEqual(len(check_result), 1872)

    def test_load_wrong(self):
        """Test load Json.

        Load a Json Quantum Program: Errors Control.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, QP_program.load)

    ###############################################################
    # Tests for working with backends
    ###############################################################

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_setup_api(self):
        """Check the api is set up.

        If all correct is should be true.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        QP_program.set_api(QE_TOKEN, QE_URL)
        config = QP_program.get_api_config()
        self.assertTrue(config)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_available_backends_exist(self):
        """Test if there are available backends.

        If all correct some should exists (even if offline).
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        QP_program.set_api(QE_TOKEN, QE_URL)
        available_backends = QP_program.available_backends()
        self.assertTrue(available_backends)

    def test_local_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists (even if ofline).
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        local_backends = qiskit.backends.local_backends()
        self.assertTrue(local_backends)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_online_backends_exist(self):
        """Test if there are online backends.

        If all correct some should exists.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        QP_program.set_api(QE_TOKEN, QE_URL)
        online_backends = QP_program.online_backends()
        self.log.info(online_backends)
        self.assertTrue(online_backends)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_online_devices(self):
        """Test if there are online backends (which are devices).

        If all correct some should exists. NEED internet connection for this.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        qp = QuantumProgram(specs=self.QPS_SPECS)
        qp.set_api(QE_TOKEN, QE_URL)
        online_devices = qp.online_devices()
        self.log.info(online_devices)
        self.assertTrue(isinstance(online_devices, list))

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_online_simulators(self):
        """Test if there are online backends (which are simulators).

        If all correct some should exists. NEED internet connection for this.
        """
        # TODO: Jay should we check if we the QX is online before runing.
        qp = QuantumProgram(specs=self.QPS_SPECS)
        qp.set_api(QE_TOKEN, QE_URL)
        online_simulators = qp.online_simulators()
        self.log.info(online_simulators)
        self.assertTrue(isinstance(online_simulators, list))

    def test_backend_status(self):
        """Test backend_status.

        If all correct should return dictionary with available: True/False.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        out = QP_program.get_backend_status("local_qasm_simulator")
        self.assertIn(out['available'], [True])

    def test_backend_status_fail(self):
        """Test backend_status.

        If all correct should return dictionary with available: True/False.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(ValueError, qp.get_backend_status, "fail")

    def test_get_backend_configuration(self):
        """Test get_backend_configuration.

        If all correct should return configuration for the
        local_qasm_simulator.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        config_keys = {'name', 'simulator', 'local', 'description',
                       'coupling_map', 'basis_gates'}
        backend_config = qp.get_backend_configuration("local_qasm_simulator")
        self.assertTrue(config_keys < backend_config.keys())

    def test_get_backend_configuration_fail(self):
        """Test get_backend_configuration fail.

        If all correct should return LookupError.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        # qp.get_backend_configuration("fail")
        self.assertRaises(LookupError, qp.get_backend_configuration, "fail")

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_get_backend_calibration(self):
        """Test get_backend_calibration.

        If all correct should return dictionay on length 4.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend_list = QP_program.online_backends()
        if backend_list:
            backend = backend_list[0]
        result = QP_program.get_backend_calibration(backend)
        self.log.info(result)
        self.assertEqual(len(result), 4)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_get_backend_parameters(self):
        """Test get_backend_parameters.

        If all correct should return dictionay on length 4.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend_list = QP_program.online_backends()
        if backend_list:
            backend = backend_list[0]
        result = QP_program.get_backend_parameters(backend)
        self.log.info(result)
        self.assertEqual(len(result), 4)

    ###############################################################
    # Test for compile
    ###############################################################

    def test_compile_program(self):
        """Test compile_program.

        If all correct should return COMPLETED.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'test'
        coupling_map = None
        out = QP_program.compile(['circuitName'], backend=backend,
                                 coupling_map=coupling_map, qobj_id='cooljob')
        self.log.info(out)
        self.assertEqual(len(out), 3)

    def test_get_compiled_configuration(self):
        """Test compiled_configuration.

        If all correct should return lenght 6 dictionary.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = QP_program.compile(['circuitName'], backend=backend,
                                  coupling_map=coupling_map)
        result = QP_program.get_compiled_configuration(qobj, 'circuitName')
        self.log.info(result)
        self.assertEqual(len(result), 4)

    def test_get_compiled_qasm(self):
        """Test get_compiled_qasm.

        If all correct should return lenght  dictionary.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = QP_program.compile(['circuitName'], backend=backend,
                                  coupling_map=coupling_map)
        result = QP_program.get_compiled_qasm(qobj, 'circuitName',)
        self.log.info(result)
        self.assertEqual(len(result), 184)

    def test_get_execution_list(self):
        """Test get_execution_list.

        If all correct should return {'local_qasm_simulator': ['circuitName']}.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = QP_program.get_circuit("circuitName")
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = QP_program.compile(['circuitName'], backend=backend,
                                  coupling_map=coupling_map, qobj_id='cooljob')
        result = QP_program.get_execution_list(qobj)
        self.log.info(result)
        self.assertEqual(result, ['circuitName'])

    def test_compile_coupling_map(self):
        """Test compile_coupling_map.

        If all correct should return data with the same stats. The circuit may
        be different.
        """
        QP_program = QuantumProgram()
        q = QP_program.create_quantum_register("q", 3)
        c = QP_program.create_classical_register("c", 3)
        qc = QP_program.create_circuit("circuitName", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        coupling_map = {0: [1], 1: [2]}
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2)}
        circuits = ["circuitName"]
        qobj = QP_program.compile(circuits, backend=backend, shots=shots,
                                  coupling_map=coupling_map,
                                  initial_layout=initial_layout, seed=88)
        result = QP_program.run(qobj)
        to_check = QP_program.get_qasm("circuitName")
        self.assertEqual(len(to_check), 160)
        self.assertEqual(result.get_counts("circuitName"),
                         {'000': 518, '111': 506})

    def test_change_circuit_qobj_after_compile(self):
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals':[1, 2, 3, 4]}
        qobj1 = QP_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        qobj1['circuits'][0]['config']['shots'] = 50
        qobj1['circuits'][0]['config']['xvals'] = [1,1,1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = QP_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        self.assertTrue(qobj1['circuits'][0]['config']['shots'] == 50)
        self.assertTrue(qobj1['circuits'][1]['config']['shots'] == 1)
        self.assertTrue(qobj1['circuits'][0]['config']['xvals'] == [1,1,1])
        self.assertTrue(qobj1['circuits'][1]['config']['xvals'] == [1,2,3,4])
        self.assertTrue(qobj1['config']['shots'] == 1024)
        self.assertTrue(qobj2['circuits'][0]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][1]['config']['shots'] == 1000)
        self.assertTrue(qobj2['circuits'][0]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2['circuits'][1]['config']['xvals'] == [
            'only for qobj2', 2, 3, 4])

    ###############################################################
    # Test for running programs
    ###############################################################

    def test_run_program(self):
        """Test run.

        If all correct should the data.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        qobj = QP_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88)
        out = QP_program.run(qobj)
        results2 = out.get_counts('qc2')
        results3 = out.get_counts('qc3')
        self.assertEqual(results2, {'000': 518, '111': 506})
        self.assertEqual(results3, {'001': 119, '111': 129, '110': 134,
                                    '100': 117, '000': 129, '101': 126,
                                    '010': 145, '011': 125})

    def test_run_async_program(self):
        """Test run_async.

        If all correct should the data.
        """
        def _job_done_callback(result):
            try:
                results2 = result.get_counts('qc2')
                results3 = result.get_counts('qc3')
                self.assertEqual(results2, {'000': 518, '111': 506})
                self.assertEqual(results3, {'001': 119, '111': 129, '110': 134,
                                            '100': 117, '000': 129, '101': 126,
                                            '010': 145, '011': 125})
            except Exception as e:
                self.qp_program_exception = e
            finally:
                self.qp_program_finished = True

        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        qobj = QP_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88)

        self.qp_program_finished = False
        self.qp_program_exception = None
        out = QP_program.run_async(qobj, callback=_job_done_callback)

        while not self.qp_program_finished:
            # Wait until the job_done_callback is invoked and completed.
            pass

        if self.qp_program_exception:
            raise self.qp_program_exception

    def test_run_batch(self):
        """Test run_batch

        If all correct should the data.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        qobj_list = [ QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88) ]

        results = QP_program.run_batch(qobj_list)
        for result in results:
            counts2 = result.get_counts('qc2')
            counts3 = result.get_counts('qc3')
            self.assertEqual(counts2, {'000': 518, '111': 506})
            self.assertEqual(counts3, {'001': 119, '111': 129, '110': 134,
                                       '100': 117, '000': 129, '101': 126,
                                       '010': 145, '011': 125})

    def test_run_batch_async(self):
        """Test run_batch_async

        If all correct should the data.
        """
        def _jobs_done_callback(results):
            try:
                for result in results:
                    counts2 = result.get_counts('qc2')
                    counts3 = result.get_counts('qc3')
                    self.assertEqual(counts2, {'000': 518, '111': 506})
                    self.assertEqual(counts3, {'001': 119, '111': 129,
                                               '110': 134, '100': 117,
                                               '000': 129, '101': 126,
                                               '010': 145, '011': 125})
            except Exception as e:
                self.qp_program_exception = e
            finally:
                self.qp_program_finished = True

        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        qobj_list = [ QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88),
                      QP_program.compile(circuits, backend=backend, shots=shots,
                      seed=88) ]

        self.qp_program_finished = False
        self.qp_program_exception = None
        results = QP_program.run_batch_async(qobj_list,
                                             callback=_jobs_done_callback)
        while not self.qp_program_finished:
            # Wait until the job_done_callback is invoked and completed.
            pass

        if self.qp_program_exception:
            raise self.qp_program_exception

    def test_combine_results(self):
        """Test run.

        If all correct should the data.
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 1)
        cr = QP_program.create_classical_register("cr", 1)
        qc1 = QP_program.create_circuit("qc1", [qr], [cr])
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc1.measure(qr[0], cr[0])
        qc2.x(qr[0])
        qc2.measure(qr[0], cr[0])
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        res1 = QP_program.execute(['qc1'], backend=backend, shots=shots)
        res2 = QP_program.execute(['qc2'], backend=backend, shots=shots)
        counts1 = res1.get_counts('qc1')
        counts2 = res2.get_counts('qc2')
        res1 += res2  # combine results
        counts12 = [res1.get_counts('qc1'), res1.get_counts('qc2')]
        self.assertEqual(counts12, [counts1, counts2])

    def test_local_qasm_simulator(self):
        """Test execute.

        If all correct should the data.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc3.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        qc3.measure(qr[2], cr[2])
        circuits = ['qc2', 'qc3']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        out = QP_program.execute(circuits, backend=backend, shots=shots,
                                 seed=88)
        results2 = out.get_counts('qc2')
        results3 = out.get_counts('qc3')
        self.log.info(results3)
        self.assertEqual(results2, {'000': 518, '111': 506})
        self.assertEqual(results3, {'001': 119, '111': 129, '110': 134,
                                    '100': 117, '000': 129, '101': 126,
                                    '010': 145, '011': 125})

    def test_local_qasm_simulator_one_shot(self):
        """Test sinlge shot of local simulator .

        If all correct should the quantum state.
        """
        QP_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = QP_program.get_quantum_register("qname")
        cr = QP_program.get_classical_register("cname")
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc3 = QP_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc3.cx(qr[0], qr[1])
        qc3.cx(qr[0], qr[2])
        circuits = ['qc2', 'qc3']
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    seed=9)
        quantum_state = np.array([0.70710678+0.j, 0.70710678+0.j,
                                  0.00000000+0.j, 0.00000000+0.j,
                                  0.00000000+0.j, 0.00000000+0.j,
                                  0.00000000+0.j, 0.00000000+0.j])
        norm = np.dot(np.conj(quantum_state),
                      result.get_data('qc2')['quantum_state'])
        self.assertAlmostEqual(norm, 1)
        quantum_state = np.array([0.70710678+0.j, 0+0.j,
                                  0.00000000+0.j, 0.00000000+0.j,
                                  0.00000000+0.j, 0.00000000+0.j,
                                  0.00000000+0.j, 0.70710678+0.j])
        norm = np.dot(np.conj(quantum_state),
                      result.get_data('qc3')['quantum_state'])
        self.assertAlmostEqual(norm, 1)

    def test_local_unitary_simulator(self):
        """Test unitary simulator.

        If all correct should the h otimes h and cx.
        """
        QP_program = QuantumProgram()
        q = QP_program.create_quantum_register("q", 2)
        c = QP_program.create_classical_register("c", 2)
        qc1 = QP_program.create_circuit("qc1", [q], [c])
        qc2 = QP_program.create_circuit("qc2", [q], [c])
        qc1.h(q)
        qc2.cx(q[0], q[1])
        circuits = ['qc1', 'qc2']
        backend = 'local_unitary_simulator'  # the backend to run on
        result = QP_program.execute(circuits, backend=backend)
        unitary1 = result.get_data('qc1')['unitary']
        unitary2 = result.get_data('qc2')['unitary']
        unitaryreal1 = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                                 [0.5, 0.5, -0.5, -0.5],
                                 [0.5, -0.5, -0.5, 0.5]])
        unitaryreal2 = np.array([[1,  0,  0, 0], [0, 0,  0,  1],
                                 [0.,  0, 1, 0], [0,  1,  0,  0]])
        norm1 = np.trace(np.dot(np.transpose(np.conj(unitaryreal1)), unitary1))
        norm2 = np.trace(np.dot(np.transpose(np.conj(unitaryreal2)), unitary2))
        self.assertAlmostEqual(norm1, 4)
        self.assertAlmostEqual(norm2, 4)

    def test_run_program_map(self):
        """Test run_program_map.

        If all correct should return 10010.
        """
        QP_program = QuantumProgram()
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 100  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = {0: [1], 1: [2], 2: [3], 3: [4]}
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2), ("q", 3): ("q", 3),
                          ("q", 4): ("q", 4)}
        QP_program.load_qasm_file(self.QASM_FILE_PATH_2, name="circuit-dev")
        circuits = ["circuit-dev"]
        qobj = QP_program.compile(circuits, backend=backend, shots=shots,
                                  max_credits=max_credits, seed=65,
                                  coupling_map=coupling_map,
                                  initial_layout=initial_layout)
        result = QP_program.run(qobj)
        self.assertEqual(result.get_counts("circuit-dev"), {'10010': 100})

    def test_execute_program_map(self):
        """Test execute_program_map.

        If all correct should return 10010.
        """
        QP_program = QuantumProgram()
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 100  # the number of shots in the experiment.
        max_credits = 3
        coupling_map = {0: [1], 1: [2], 2: [3], 3: [4]}
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2), ("q", 3): ("q", 3),
                          ("q", 4): ("q", 4)}
        QP_program.load_qasm_file(self.QASM_FILE_PATH_2, "circuit-dev")
        circuits = ["circuit-dev"]
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    max_credits=max_credits,
                                    coupling_map=coupling_map,
                                    initial_layout=initial_layout, seed=5455)
        self.assertEqual(result.get_counts("circuit-dev"), {'10010': 100})

    def test_average_data(self):
        """Test average_data.

        If all correct should return the data.
        """
        QP_program = QuantumProgram()
        q = QP_program.create_quantum_register("q", 2)
        c = QP_program.create_classical_register("c", 2)
        qc = QP_program.create_circuit("qc", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        circuits = ['qc']
        shots = 10000  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        results = QP_program.execute(circuits, backend=backend, shots=shots)
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        meanzz = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        meanzi = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        meaniz = results.average_data("qc", observable)
        self.assertAlmostEqual(meanzz,  1, places=1)
        self.assertAlmostEqual(meanzi,  0, places=1)
        self.assertAlmostEqual(meaniz,  0, places=1)

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_execute_one_circuit_simulator_online(self):
        """Test execute_one_circuit_simulator_online.

        If all correct should return the data.
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("q", 1)
        cr = QP_program.create_classical_register("c", 1)
        qc = QP_program.create_circuit("qc", [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        shots = 1024  # the number of shots in the experiment.
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend = QP_program.online_simulators()[0]
        # print(backend)
        result = QP_program.execute(['qc'], backend=backend,
                                    shots=shots, max_credits=3,
                                    seed=73846087)
        counts = result.get_counts('qc')
        self.assertEqual(counts, {'0': 498, '1': 526})

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_simulator_online_size(self):
        """Test test_simulator_online_size.

        If all correct should return the data.
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("q", 25)
        cr = QP_program.create_classical_register("c", 25)
        qc = QP_program.create_circuit("qc", [qr], [cr])
        qc.h(qr)
        qc.measure(qr, cr)
        shots = 1  # the number of shots in the experiment.
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend = 'ibmqx_qasm_simulator'
        result = QP_program.execute(['qc'], backend=backend,
                                    shots=shots, max_credits=3,
                                    seed=73846087)
        self.assertRaises(RegisterSizeError, result.get_data, 'qc')

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_execute_several_circuits_simulator_online(self):
        """Test execute_several_circuits_simulator_online.

        If all correct should return the data.
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("q", 2)
        cr = QP_program.create_classical_register("c", 2)
        qc1 = QP_program.create_circuit("qc1", [qr], [cr])
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc1.h(qr)
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        circuits = ['qc1', 'qc2']
        shots = 1024  # the number of shots in the experiment.
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    max_credits=3, seed=1287126141)
        counts1 = result.get_counts('qc1')
        counts2 = result.get_counts('qc2')
        self.assertEqual(counts1,  {'10': 277, '11': 238, '01': 258,
                                    '00': 251})
        self.assertEqual(counts2, {'11': 515, '00': 509})

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_execute_one_circuit_real_online(self):
        """Test execute_one_circuit_real_online.

        If all correct should return a result object
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 1)
        cr = QP_program.create_classical_register("cr", 1)
        qc = QP_program.create_circuit("circuitName", [qr], [cr])
        qc.h(qr)
        qc.measure(qr[0], cr[0])
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend = 'ibmqx_qasm_simulator'
        shots = 1  # the number of shots in the experiment.
        status = QP_program.get_backend_status(backend)
        if status['available'] is False:
            pass
        else:
            result = QP_program.execute(['circuitName'], backend=backend,
                                        shots=shots, max_credits=3)
            self.assertIsInstance(result, Result)

    def test_local_qasm_simulator_two_registers(self):
        """Test local_qasm_simulator_two_registers.

        If all correct should the data.
        """
        QP_program = QuantumProgram()
        q1 = QP_program.create_quantum_register("q1", 2)
        c1 = QP_program.create_classical_register("c1", 2)
        q2 = QP_program.create_quantum_register("q2", 2)
        c2 = QP_program.create_classical_register("c2", 2)
        qc1 = QP_program.create_circuit("qc1", [q1, q2], [c1, c2])
        qc2 = QP_program.create_circuit("qc2", [q1, q2], [c1, c2])

        qc1.x(q1[0])
        qc2.x(q2[1])
        qc1.measure(q1[0], c1[0])
        qc1.measure(q1[1], c1[1])
        qc1.measure(q2[0], c2[0])
        qc1.measure(q2[1], c2[1])
        qc2.measure(q1[0], c1[0])
        qc2.measure(q1[1], c1[1])
        qc2.measure(q2[0], c2[0])
        qc2.measure(q2[1], c2[1])
        circuits = ['qc1', 'qc2']
        shots = 1024  # the number of shots in the experiment.
        backend = 'local_qasm_simulator'
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    seed=8458)
        result1 = result.get_counts('qc1')
        result2 = result.get_counts('qc2')
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_online_qasm_simulator_two_registers(self):
        """Test online_qasm_simulator_two_registers.

        If all correct should the data.
        """
        QP_program = QuantumProgram()
        q1 = QP_program.create_quantum_register("q1", 2)
        c1 = QP_program.create_classical_register("c1", 2)
        q2 = QP_program.create_quantum_register("q2", 2)
        c2 = QP_program.create_classical_register("c2", 2)
        qc1 = QP_program.create_circuit("qc1", [q1, q2], [c1, c2])
        qc2 = QP_program.create_circuit("qc2", [q1, q2], [c1, c2])

        qc1.x(q1[0])
        qc2.x(q2[1])
        qc1.measure(q1[0], c1[0])
        qc1.measure(q1[1], c1[1])
        qc1.measure(q2[0], c2[0])
        qc1.measure(q2[1], c2[1])
        qc2.measure(q1[0], c1[0])
        qc2.measure(q1[1], c1[1])
        qc2.measure(q2[0], c2[0])
        qc2.measure(q2[1], c2[1])
        circuits = ['qc1', 'qc2']
        shots = 1024  # the number of shots in the experiment.
        QP_program.set_api(QE_TOKEN, QE_URL)
        backend = QP_program.online_simulators()[0]
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    seed=8458)
        result1 = result.get_counts('qc1')
        result2 = result.get_counts('qc2')
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})

    ###############################################################
    # More test cases for interesting examples
    ###############################################################

    def test_add_circuit(self):
        """Test add two circuits.

        If all correct should return the data
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 2)
        cr = QP_program.create_classical_register("cr", 2)
        qc1 = QP_program.create_circuit("qc1", [qr], [cr])
        qc2 = QP_program.create_circuit("qc2", [qr], [cr])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        new_circuit = qc1 + qc2
        QP_program.add_circuit('new_circuit', new_circuit)
        # new_circuit.measure(qr[0], cr[0])
        circuits = ['new_circuit']
        backend = 'local_qasm_simulator'  # the backend to run on
        shots = 1024  # the number of shots in the experiment.
        result = QP_program.execute(circuits, backend=backend, shots=shots,
                                    seed=78)
        # print(QP_program.get_qasm('new_circuit'))
        self.assertEqual(result.get_counts('new_circuit'),
                         {'00': 505, '01': 519})

    def test_add_circuit_fail(self):
        """Test add two circuits fail.

        If the circuits have different registers it should return a QISKitError
        """
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_register("qr", 1)
        cr = QP_program.create_classical_register("cr", 1)
        q = QP_program.create_quantum_register("q", 1)
        c = QP_program.create_classical_register("c", 1)
        qc1 = QP_program.create_circuit("qc1", [qr], [cr])
        qc2 = QP_program.create_circuit("qc2", [q], [c])
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(q[0], c[0])
        # new_circuit = qc1 + qc2
        self.assertRaises(QISKitError, qc1.__add__, qc2)

    def test_example_multiple_compile(self):
        """Test a toy example compiling multiple circuits.

        Pass if the results are correct.
        """
        coupling_map = {0: [1, 2],
                        1: [2],
                        2: [],
                        3: [2, 4],
                        4: [2]}
        QPS_SPECS = {
            "circuits": [{
                "name": "ghz",
                "quantum_registers": [{
                    "name": "q",
                    "size": 5
                }],
                "classical_registers": [{
                    "name": "c",
                    "size": 5}
                ]}, {
                "name": "bell",
                "quantum_registers": [{
                    "name": "q",
                    "size": 5
                }],
                "classical_registers": [{
                    "name": "c",
                    "size": 5
                }]}
            ]
        }
        qp = QuantumProgram(specs=QPS_SPECS)
        ghz = qp.get_circuit("ghz")
        bell = qp.get_circuit("bell")
        q = qp.get_quantum_register("q")
        c = qp.get_classical_register("c")
        # Create a GHZ state
        ghz.h(q[0])
        for i in range(4):
            ghz.cx(q[i], q[i+1])
        # Insert a barrier before measurement
        ghz.barrier()
        # Measure all of the qubits in the standard basis
        for i in range(5):
            ghz.measure(q[i], c[i])
        # Create a Bell state
        bell.h(q[0])
        bell.cx(q[0], q[1])
        bell.barrier()
        bell.measure(q[0], c[0])
        bell.measure(q[1], c[1])
        bellobj = qp.compile(["bell"], backend='local_qasm_simulator',
                             shots=2048, seed=10)
        ghzobj = qp.compile(["ghz"], backend='local_qasm_simulator',
                            shots=2048, coupling_map=coupling_map,
                            seed=10)
        bellresult = qp.run(bellobj)
        ghzresult = qp.run(ghzobj)
        self.log.info(bellresult.get_counts("bell"))
        self.log.info(ghzresult.get_counts("ghz"))
        self.assertEqual(bellresult.get_counts("bell"),
                         {'00000': 1034, '00011': 1014})
        self.assertEqual(ghzresult.get_counts("ghz"),
                         {'00000': 1047, '11111': 1001})

    @unittest.skipIf(TRAVIS_FORK_PULL_REQUEST, 'Travis fork pull request')
    def test_example_swap_bits(self):
        """Test a toy example swapping a set bit around.

        Uses the mapper. Pass if results are correct.
        """
        backend = "ibmqx_qasm_simulator"
        coupling_map = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11],
                        4: [5, 12], 5: [6, 13], 6: [7, 14], 7: [15], 8: [9],
                        9: [10], 10: [11], 11: [12], 12: [13], 13: [14],
                        14: [15]}

        def swap(qc, q0, q1):
            """Swap gate."""
            qc.cx(q0, q1)
            qc.cx(q1, q0)
            qc.cx(q0, q1)
        n = 3  # make this at least 3
        QPS_SPECS = {
            "circuits": [{
                "name": "swapping",
                "quantum_registers": [{
                    "name": "q",
                    "size": n},
                    {"name": "r",
                     "size": n}
                ],
                "classical_registers": [
                    {"name": "ans",
                     "size": 2*n},
                ]
            }]
        }
        qp = QuantumProgram(specs=QPS_SPECS)
        qp.set_api(QE_TOKEN, QE_URL)
        if backend not in qp.online_simulators():
            unittest.skip('backend "{}" not available'.format(backend))
        qc = qp.get_circuit("swapping")
        q = qp.get_quantum_register("q")
        r = qp.get_quantum_register("r")
        ans = qp.get_classical_register("ans")
        # Set the first bit of q
        qc.x(q[0])
        # Swap the set bit
        swap(qc, q[0], q[n-1])
        swap(qc, q[n-1], r[n-1])
        swap(qc, r[n-1], q[1])
        swap(qc, q[1], r[1])
        # Insert a barrier before measurement
        qc.barrier()
        # Measure all of the qubits in the standard basis
        for j in range(n):
            qc.measure(q[j], ans[j])
            qc.measure(r[j], ans[j+n])
        # First version: no mapping
        result = qp.execute(["swapping"], backend=backend,
                            coupling_map=None, shots=1024,
                            seed=14)
        self.assertEqual(result.get_counts("swapping"),
                         {'010000': 1024})
        # Second version: map to coupling graph
        result = qp.execute(["swapping"], backend=backend,
                            coupling_map=coupling_map, shots=1024,
                            seed=14)
        self.assertEqual(result.get_counts("swapping"),
                         {'010000': 1024})

    def test_offline(self):
        import string
        import random
        qp = QuantumProgram()
        FAKE_TOKEN = 'thistokenisnotgoingtobesentnowhere'
        FAKE_URL = 'http://{0}.com'.format(
            ''.join(random.choice(string.ascii_lowercase) for _ in range(63))
        )
        # SDK will throw ConnectionError on every call that implies a connection
        self.assertRaises(ConnectionError, qp.set_api, FAKE_TOKEN, FAKE_URL)


if __name__ == '__main__':
    unittest.main(verbosity=2)
