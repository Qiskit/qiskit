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
        "name": "circuitName",
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


    def test_new_compile(self):
        QP_program = QuantumProgram()
        device = 'local_qasm_simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        QP_program.load_qasm("circuit-dev","test.qasm")
        circuits = ["circuit-dev"]

        result = QP_program.compile(circuits, device, shots, credits, coupling_map)
        to_check = QP_program.get_circuit("circuit-dev")

        self.assertEqual(len(to_check['QASM']),1569)

    def test_new_run(self):
        QP_program = QuantumProgram()

        device = 'local_qasm_simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        QP_program.load_qasm("circuit-dev","test.qasm")
        circuits = ["circuit-dev"]

        result = QP_program.compile(circuits, device, shots, credits, coupling_map)

        result = QP_program.run()
        print(result)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_new_execute(self):
        QP_program = QuantumProgram()

        device = 'local_qasm_simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        QP_program.load_qasm("circuit-dev","test.qasm")
        circuits = ["circuit-dev"]

        result = QP_program.execute(circuits, device, shots, credits, coupling_map)
        print(result)

        # result = QP_program.run()
        
        print(result)
        self.assertEqual(result['status'], "COMPLETED")



    def test_contact_create_circuit_multiregisters(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qr = QP_program.quantum_registers("qname")
        cr = QP_program.classical_registers("cname")
        qr2 = QP_program.create_quantum_registers("qr", 3)
        cr2 = QP_program.create_classical_registers("cr", 3)
        qc_result = QP_program.create_circuit(name="qc2",
                                              qregisters=["qname", "qr"],
                                              cregisters=["cname", "cr"])
        self.assertIsInstance(qc_result, QuantumCircuit)
        self.assertEqual(len(qc_result.qasm()), 90)

    def test_run_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        
        # qc = QP_program.circuit("circuitName")
        qr = QP_program.quantum_registers("qname")
        cr = QP_program.classical_registers("cname")
        
        qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
        qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        qc2.h(qr[0])
        qc3.h(qr[0])

        circuits = ['qc2', 'qc3']
        circuits = ['qc2']
        device = 'simulator'  # the device to run on
        shots = 1024  # the number of shots in the experiment.
        credits = 3
        coupling_map = None
        print('---------------test_run_program----------')
        apiconnection = QP_program.set_api(
            API_TOKEN, URL)
        QP_program.compile(circuits, device, shots, credits, coupling_map)
        result = QP_program.run()
        
        print(result)
        print('---------------****test_run_program----------')
        # TODO: Revire result
        self.assertEqual(result['status'], 'COMPLETED')

    # def test_add_circuit(self):
    #     QP_program = QuantumProgram(specs=QPS_SPECS)
    #     qc, qr, cr = QP_program.quantum_elements()

    #     qc2 = QP_program.create_circuit("qc2", ["qname"], ["cname"])
    #     qc3 = QP_program.create_circuit("qc3", ["qname"], ["cname"])
        
    #     qc2.h(qr[0])
    #     qc3.h(qr[1])
        
    #     new_circuit = qc2 + qc3
    #     # QP_program.add_circuit('new_circuit',new_circuit)
    #     circuits = ['new_circuit']

    #     circuits = ['qc2']

    #     device = 'local_qasm_simulator'  # the device to run on
    #     shots = 2  # the number of shots in the experiment.
    #     credits = 3
    #     coupling_map = None
    #     result = QP_program.execute(circuits, device, shots)

    #     self.assertEqual(result['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
