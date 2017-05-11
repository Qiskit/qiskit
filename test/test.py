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
QISKit Test

Authors: Ismael Faro
"""

import sys
import json
import unittest

import Qconfig

from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister


sys.path.insert(0, '../')

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
            Qconfig.config["url"],
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
        self.assertEqual(len(result), 101)

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
        self.assertEqual(len(result), 415)

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
        result = QP_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
        self.assertTrue(result)

    def test_execute_one_circuit_simulator_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])

        device = 'simulator'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            Qconfig.APItoken, Qconfig.config["url"])
        result = QP_program.run_circuit(
            "circuit-name", device, shots, max_credits=3)
        self.assertEqual(result["status"], "DONE")

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
            Qconfig.APItoken, Qconfig.config["url"])
        result = QP_program.run_circuits(
            circuits, device, shots, max_credits=3)
        self.assertEqual(result["status"], "RUNNING")

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
            Qconfig.APItoken, Qconfig.config["url"])
        result = QP_program.run_program(device, shots, max_credits=3)
        self.assertEqual(result["status"], "RUNNING")

    @unittest.skip
    def test_execute_one_circuit_real_online(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()
        qc.h(qr[1])

        device = 'qx5q'  # the device to run on
        shots = 1  # the number of shots in the experiment.

        apiconnection = QP_program.set_api(
            Qconfig.APItoken, Qconfig.config["url"])
        result = QP_program.run_circuit(
            "circuit-name", device, shots, max_credits=3)
        self.assertEqual(result["status"], "DONE")

    @unittest.skip
    def test_execute_one_circuit_simulator_local(self):
        pass

    def test_compile_program(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        qc, qr, cr = QP_program.quantum_elements()

        qc.h(qr[0])
        qc.h(qr[0])
        qc.measure(qr[0], cr[1])

        device = 'qx5qv2'
        shots = 1024
        credits = 3
        coupling_map = None

        source = QP_program.compile(device, coupling_map, shots, credits)[
            'compiled_circuits'][0]['qasm']

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
            Qconfig.APItoken, Qconfig.config["url"])
        QP_program.compile(device, coupling_map, shots, credits)
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
            Qconfig.APItoken, Qconfig.config["url"])
        result = QP_program.execute(
            device, coupling_map, shots, credits)['status']
        self.assertEqual(result, 'COMPLETED')

        # QP_program.plotter()

    def test_last(self):
        QP_program = QuantumProgram(specs=QPS_SPECS)
        # QP_program.plotter()


if __name__ == '__main__':
    unittest.main()

# TODO: Topology definition
# topology={
#     hardware={},
#     map={}
# }

# TODO:
# sim1 = myQP.set_scope(topology=topology)
# topology2={
#     map={}
# }


# sim2 = myQP.set_scope( topology=topology2)

# sim1.compile.execute.plot()
# sim2.compile.execute.plot()

# sim1 = myQP.set_scope(hardware={}, map={}, topology={})

# myQP.compile()
#   myQP.parse(versionQasm, qfiles)
#   myQP.unroller()
#   myQP.optimizer(standar)
#   myQP.map(topology, operations)
#   myQP.optimizer(cleaner)
# myQP.execute()

# myQP.execute()
# myQP.execute(debug = {})


# myQP.plot()

# hardware.status()
# hardware.command()
# use methods instead - or have method as well
# c1 = a + b + c
# c2 = a + bp + c

# chemistry1 = make_variational_state + do_measurement_1
# chemistry2 = make_variational_state + do_measurement_2

# p.add_circuit(c1)
