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
"""

import sys
import os
import json
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister

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
    "name": "program-name",
    "circuits": [{
        "name": "circuit-name",
        "quantum_registers": [{
            "name": "qname",
            "size": 3}],
        "classical_registers": [{
            "name": "cname",
            "size": 3}]
    }]
}


class TestQISKit(unittest.TestCase):
    """
    Test Create Program
    """

    def test_create_program_with_Specs(self):
        """
        Test Quantum Object Factory creation using Specs
        """
        result = QuantumProgram(specs=QPS_SPECS)
        self.assertTrue(isinstance(result, QuantumProgram))

    def test_create_program(self):
        """
        Test Quantum Object Factory
        """
        result = QuantumProgram()
        self.assertTrue(isinstance(result, QuantumProgram))

    def test_config_scripts_file(self):
        """
        Test Qconfig
        """
        self.assertEqual(
            URL,
            "https://quantumexperience.ng.bluemix.net/api")

    def test_get_components(self):
        """
        Get the program componentes, like Circuits and Registers
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertIsInstance(qr, QuantumRegister)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_get_individual_components(self):
        """
        Get the program componentes, like Circuits and Registers
        """
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc = QP_program.circuit("circuit-name")
        qr = QP_program.quantum_registers("qname")
        cr = QP_program.classical_registers("cname")
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertIsInstance(qr, QuantumRegister)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_classical_register(self):
        QP_program = QuantumProgram()
        cr = QP_program.create_classical_registers("cr", 3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_registers("qr", 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_circuit(self):
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_registers("qr", 3)
        cr = QP_program.create_classical_registers("cr", 3)
        qc = QP_program.create_circuit("qc", ["qr"], ["cr"])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_create_several_circuits(self):
        QP_program = QuantumProgram()
        qr = QP_program.create_quantum_registers("qr", 3)
        cr = QP_program.create_classical_registers("cr", 3)
        qc1 = QP_program.create_circuit("qc", ["qr"], ["cr"])
        qc2 = QP_program.create_circuit("qc2", ["qr"], ["cr"])
        qc3 = QP_program.create_circuit("qc2", ["qr"], ["cr"])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    @unittest.skip
    def test_load_qasm(self):
        pass

    def test_print_circuit(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])
        result = qc.qasm()
        self.assertEqual(len(result), 78)

    def test_print_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])
        result = QP_program.program_to_text()
        self.assertEqual(len(result), 125)

    def test_create_add_gates(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
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
        result = QP_program.program_to_text()
        print(result)
        self.assertEqual(len(result), 439)

    def test_contact_create_circuit_multiregisters(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qr2 = QP_program.create_quantum_registers("qr", 3)
        cr2 = QP_program.create_classical_registers("cr", 3)
        qc_result = QP_program.create_circuit(name="qc2",
                                              qregisters=["qname", "qr"],
                                              cregisters=[cr, cr2])
        self.assertIsInstance(qc_result, QuantumCircuit)
        self.assertEqual(len(qc_result.qasm()), 90)

    def test_contact_multiple_horizontal_circuits(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit(name="qc2",
                                        qregisters=["qname"],
                                        cregisters=["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc_result = qc2 + qc3
        self.assertIsInstance(qc_result, QuantumCircuit)

    @unittest.skip
    def test_contact_multiple_vertical_circuits(self):
        pass

    def test_setup_api(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        result = QP_program.set_api(API_TOKEN, URL)
        self.assertTrue(result)

    def test_execute_one_circuit_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])

        device = 'simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        result = QP_program.execute(
            [qc], device, shots, max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_several_circuits_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])

        circuits = [qc2, qc3]

        device = 'simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        result = QP_program.execute(
            circuits, device, shots, max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_program_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])

        device = 'simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        result = QP_program.execute([qc], device, shots, max_credits=3)
        self.assertEqual(result["status"], "COMPLETED")

    def test_execute_one_circuit_real_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])

        device = 'IBMQX5qv2'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        result = QP_program.execute([qc], device, shots, max_credits=3)
        print(result)
        self.assertIn(result["status"], ["DONE","Error"])

    @unittest.skip
    def test_execute_one_circuit_simulator_local(self):
        pass

    def test_compile_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()

        qc.h(qr[0])
        qc.h(qr[0])
        qc.measure(qr[0], cr[1])

        device = 'IBMQX5qv2'
        shots = 1024
        credits = 3
        coupling_map = None

        source = QP_program.compile([qc],
                                    device,
                                    coupling_map)[0]['qasm']
 
        self.assertEqual(len(source), 168)

    def test_run_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])

        circuits = [qc2, qc3]

        device = 'simulator'  # the device to run on
        shots = 1024  # the number of shots in the experiment.
        credits = 3
        coupling_map = None

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        QP_program.compile(circuits, device, shots, credits, coupling_map)
        result = QP_program.run()
        self.assertEqual(len(result), 6)

    def test_execute_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])
        circuits = [qc2, qc3]

        device = 'simulator'  # the device to run on
        shots = 1024  # the number of shots in the experiment.
        credits = 3
        coupling_map = None

        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        result = QP_program.execute(circuits, device, shots, max_credits=3)
        # print(result)
        self.assertEqual(result['status'], 'COMPLETED')

        # QP_program.plotter()

    def test_local_qasm_simulator(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])
        circuits = [qc2, qc3]

        device = 'local_qasm_simulator'  # the device to run on
        shots = 1024  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        result = QP_program.execute(circuits, device, shots)
        # print(result)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_local_qasm_simulator_one_shot(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])
        circuits = [qc2, qc3]

        device = 'local_qasm_simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        result = QP_program.execute(circuits, device, shots)
        # print(result)
        self.assertEqual(result['compiled_circuits'][0]['result']['data']['classical_state'], 0)

    def test_local_unitary_simulator(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])
        circuits = [qc2, qc3]

        device = 'local_unitary_simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        result = QP_program.execute(circuits, device, shots)
        # print(result)
        self.assertIsNotNone(result['compiled_circuits'][0]['result']['data']['unitary'])

    def test_load_qasm(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)

        QP_program
        
        result = QP_program.load_qasm("circuit","test/test.qasm")

        self.assertEqual(len(result.qasm()),1569)


if __name__ == '__main__':
    unittest.main()
