# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test Qiskit Unroller class."""

from sys import version_info
import unittest

from qiskit import qasm, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.unroll import Unroller, DagUnroller, DAGBackend, JsonBackend
from .common import QiskitTestCase


class UnrollerTest(QiskitTestCase):
    """Test the Unroller."""

    def setUp(self):
        self.seed = 42

    @unittest.skipIf(version_info.minor == 5, "Python 3.5 dictionaries don't preserve \
                                               insertion order, so we need to skip this \
                                               test, until fixed")
    def test_from_ast_to_dag(self):
        """Test Unroller.execute()"""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        expected_result = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
h q[0];
h q[1];
h q[2];
cx q[0],r[0];
cx q[1],r[1];
cx q[2],r[2];
barrier q[0],q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure r[0] -> d[0];
measure r[1] -> d[1];
measure r[2] -> d[2];
"""
        expected_dag = DAGCircuit.fromQuantumCircuit(QuantumCircuit.from_qasm_str(expected_result))
        self.assertEqual(dag_circuit, expected_dag)

    def test_dag_to_dag_expand_gates_default_basis(self):
        """Test DagUnroller.expand_gates()"""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, DAGBackend())
        expanded_dag_circuit = dag_unroller.expand_gates()
        expected_result = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
U(pi/2,0,pi) q[2];
CX q[2],r[2];
measure r[2] -> d[2];
U(pi/2,0,pi) q[1];
CX q[1],r[1];
measure r[1] -> d[1];
U(pi/2,0,pi) q[0];
CX q[0],r[0];
barrier q[0],q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure r[0] -> d[0];
"""
        expected_dag = DAGCircuit.fromQuantumCircuit(QuantumCircuit.from_qasm_str(expected_result))
        self.assertEqual(expanded_dag_circuit, expected_dag)

    def test_dag_to_dag_expand_gates_custom_basis(self):
        """Test DagUnroller.expand_gates() to a gate basis."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, DAGBackend())
        expanded_dag_circuit = dag_unroller.expand_gates(basis=["cx", "u1", "u2", "u3"])
        expected_result = """\
OPENQASM 2.0;
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
gate u2(phi,lambda) q
{
  U((pi/2),phi,lambda) q;
}
gate h a
{
  u2(0,pi) a;
}
gate cx c,t
{
  CX c,t;
}
u2(0,pi) q[2];
cx q[2],r[2];
measure r[2] -> d[2];
u2(0,pi) q[1];
cx q[1],r[1];
measure r[1] -> d[1];
u2(0,pi) q[0];
cx q[0],r[0];
barrier q[0],q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure r[0] -> d[0];
"""
        self.assertEqual(expanded_dag_circuit.qasm(), expected_result)

    # We need to change the way we create clbit_labels and qubit_labels in order to
    # enable this test, as they are lists but the order is not important so comparing
    # them usually fails.
    @unittest.skip("Temporary skipping")
    def test_dag_to_json(self):
        """Test DagUnroller with JSON backend."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, JsonBackend())
        json_circuit = dag_unroller.execute()
        expected_result = {
            'operations':
                [
                    {'qubits': [5], 'texparams': ['0.5 \\pi', '0', '\\pi'],
                     'name': 'U', 'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [5, 2]},
                    {'clbits': [2], 'name': 'measure', 'qubits': [2]},
                    {'qubits': [4], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [4, 1]},
                    {'clbits': [1], 'name': 'measure', 'qubits': [1]},
                    {'qubits': [3], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [3, 0]},
                    {'name': 'barrier', 'qubits': [3, 4, 5]},
                    {'clbits': [5], 'name': 'measure', 'qubits': [5]},
                    {'clbits': [4], 'name': 'measure', 'qubits': [4]},
                    {'clbits': [3], 'name': 'measure', 'qubits': [3]},
                    {'clbits': [0], 'name': 'measure', 'qubits': [0]}
                ],
            'header':
                {
                    'number_of_clbits': 6,
                    'qubit_labels': [['r', 0], ['r', 1], ['r', 2], ['q', 0], ['q', 1], ['q', 2]],
                    'number_of_qubits': 6, 'clbit_labels': [['d', 3], ['c', 3]]
                }
        }

        self.assertEqual(json_circuit, expected_result)
