# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Test cases for the circuit qasm_file and qasm_string method."""

from qiskit import QISKitError
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from ..common import QiskitTestCase, Path


class LoadFromQasmTest(QiskitTestCase):
    """Test circuit.from_qasm_* set of methods."""

    def setUp(self):
        self.qasm_file_name = 'entangled_registers.qasm'
        self.qasm_file_path = self._get_resource_path(
            'qasm/' + self.qasm_file_name, Path.EXAMPLES)

    def test_qasm_file(self):
        """Test qasm_file and get_circuit.

        If all is correct we should get the qasm file loaded in _qasm_file_path
        """
        q_circuit = QuantumCircuit.from_qasm_file(self.qasm_file_path)
        dag = DAGCircuit.fromQuantumCircuit(q_circuit)
        a = QuantumRegister(4)
        b = QuantumRegister(4)
        c = ClassicalRegister(4)
        d = ClassicalRegister(4)
        q_circuit_2 = QuantumCircuit(a, b, c, d)
        q_circuit_2.h(a)
        q_circuit_2.cx(a, b)
        q_circuit_2.barrier(a)
        q_circuit_2.barrier(b)
        q_circuit_2.measure(a, c)
        q_circuit_2.measure(b, d)
        dag_2 = DAGCircuit.fromQuantumCircuit(q_circuit_2)
        nx.is_isomorphic(dag.multi_graph, dag2.multi_graph,
                         node_match=_match_dag_nodes)
        self.assertEqual(dag, dag_2)

    def test_fail_qasm_file(self):
        """Test fail_qasm_file.

        If all is correct we should get a QISKitError
        """
        self.assertRaises(QISKitError,
                          QuantumCircuit.from_qasm_file, "")

    def test_fail_qasm_string(self):
        """Test fail_qasm_string.

        If all is correct we should get a QISKitError
        """
        self.assertRaises(QISKitError,
                          QuantumCircuit.from_qasm_str, "")

    def test_qasm_text(self):
        """Test qasm_text and get_circuit.

        If all is correct we should get the qasm file loaded from the string
        """
        qasm_string = "// A simple 8 qubit example\nOPENQASM 2.0;\n"
        qasm_string += "include \"qelib1.inc\";\nqreg a[4];\n"
        qasm_string += "qreg b[4];\ncreg c[4];\ncreg d[4];\nh a;\ncx a, b;\n"
        qasm_string += "barrier a;\nbarrier b;\nmeasure a[0]->c[0];\n"
        qasm_string += "measure a[1]->c[1];\nmeasure a[2]->c[2];\n"
        qasm_string += "measure a[3]->c[3];\nmeasure b[0]->d[0];\n"
        qasm_string += "measure b[1]->d[1];\nmeasure b[2]->d[2];\n"
        qasm_string += "measure b[3]->d[3];"
        q_circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qasm_data_string = q_circuit.qasm()
        self.log.info(qasm_data_string)
        expected_qasm_data_string = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg a[4];
qreg b[4];
creg c[4];
creg d[4];
h a[0];
h a[1];
h a[2];
h a[3];
cx a[0],b[0];
cx a[1],b[1];
cx a[2],b[2];
cx a[3],b[3];
barrier a[0],a[1],a[2],a[3];
barrier b[0],b[1],b[2],b[3];
measure a[0] -> c[0];
measure a[1] -> c[1];
measure a[2] -> c[2];
measure a[3] -> c[3];
measure b[0] -> d[0];
measure b[1] -> d[1];
measure b[2] -> d[2];
measure b[3] -> d[3];
"""
        self.assertEqual(qasm_data_string, expected_qasm_data_string)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 2)

    def test_qasm_text_conditional(self):
        """Test qasm_text and get_circuit when conditionals are present.
        """
        qasm_string = '\n'.join(["OPENQASM 2.0;",
                                 "include \"qelib1.inc\";",
                                 "qreg q[1];",
                                 "creg c0[4];",
                                 "creg c1[4];",
                                 "if(c0==4) x q[0];",
                                 "if(c1==4) x q[0];"]) + '\n'

        q_circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qasm_data_string = q_circuit.qasm()
        self.log.info(qasm_data_string)

        self.assertEqual(qasm_data_string, qasm_string)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 1)

    def test_qasm_example_file(self):
        """Loads qasm/example.qasm.
        """
        qasm_filename = self._get_resource_path('qasm/example.qasm')
        expected = '\n'.join(["OPENQASM 2.0;",
                              "include \"qelib1.inc\";",
                              "qreg q[3];",
                              "qreg r[3];",
                              "creg c[3];",
                              "creg d[3];",
                              "h q[0];",
                              "h q[1];",
                              "h q[2];",
                              "cx q[0],r[0];",
                              "cx q[1],r[1];",
                              "cx q[2],r[2];",
                              "barrier q[0],q[1],q[2];",
                              "measure q[0] -> c[0];",
                              "measure q[1] -> c[1];",
                              "measure q[2] -> c[2];",
                              "measure r[0] -> d[0];",
                              "measure r[1] -> d[1];",
                              "measure r[2] -> d[2];"]) + '\n'

        q_circuit = QuantumCircuit.from_qasm_file(qasm_filename)
        qasm_data_string = q_circuit.qasm()
        self.log.info(qasm_data_string)

        self.assertEqual(qasm_data_string, expected)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 2)
