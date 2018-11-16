# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test Qiskit Unroller class."""

from sys import version_info
import unittest

from qiskit import qasm
from qiskit.unroll import Unroller, DagUnroller, DAGBackend, JsonBackend
from .common import QiskitTestCase


class UnrollerTest(QiskitTestCase):
    """Test the Unroller."""

    def setUp(self):
        self.seed = 42

    @unittest.skipIf(version_info.minor == 5, "Python 3.5 dictionaries don't preserve \
                                               insertion order, so we need to skip this \
                                               test, until fixed")
    def test_execute(self):
        """Test DagUnroller.execute()"""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, DAGBackend())
        unroller_dag_circuit = dag_unroller.execute()
        expected_result = """\
OPENQASM 2.0;
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
U(0.5*pi,0,pi) q[2];
CX q[2],r[2];
measure r[2] -> d[2];
U(0.5*pi,0,pi) q[1];
CX q[1],r[1];
measure r[1] -> d[1];
U(0.5*pi,0,pi) q[0];
CX q[0],r[0];
measure r[0] -> d[0];
barrier q[0],q[1],q[2];
measure q[2] -> c[2];
measure q[1] -> c[1];
measure q[0] -> c[0];
"""
        self.assertEqual(unroller_dag_circuit.qasm(), expected_result)

    @unittest.skipIf(version_info.minor == 5, "Python 3.5 dictionaries don't preserve \
                                               insertion order, so we need to skip this \
                                               test, until fixed")
    def test_execute_with_basis(self):
        """Test unroller.execute() to a gate basis."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend(["cx", "u1", "u2", "u3"])).execute()
        dag_unroller = DagUnroller(dag_circuit,
                                   DAGBackend(["cx", "u1", "u2", "u3"]))
        unroller_dag_circuit = dag_unroller.execute()
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
measure r[0] -> d[0];
barrier q[0],q[1],q[2];
measure q[2] -> c[2];
measure q[1] -> c[1];
measure q[0] -> c[0];
"""
        self.assertEqual(unroller_dag_circuit.qasm(), expected_result)

    def test_expand_gates(self):
        """Test DagUnroller.expand_gates()"""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, DAGBackend())
        expanded_dag_circuit = dag_unroller.expand_gates()
        expected_result = """\
OPENQASM 2.0;
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
U(0.5*pi,0,pi) q[0];
U(0.5*pi,0,pi) q[1];
U(0.5*pi,0,pi) q[2];
CX q[0],r[0];
CX q[1],r[1];
CX q[2],r[2];
barrier q[0],q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure r[0] -> d[0];
measure r[1] -> d[1];
measure r[2] -> d[2];
"""
        self.assertEqual(expanded_dag_circuit.qasm(), expected_result)

    def test_expand_gates_with_basis(self):
        """Test DagUnroller.expand_gates() to a gate basis."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend(["cx", "u1", "u2", "u3"])).execute()
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
gate cx c,t
{
  CX c,t;
}
u2(0,pi) q[0];
u2(0,pi) q[1];
u2(0,pi) q[2];
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
        self.assertEqual(expanded_dag_circuit.qasm(), expected_result)

    def test_from_dag_to_json(self):
        """Test DagUnroller with JSON backend."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, JsonBackend())
        json_circuit = dag_unroller.execute()
        expected_result = {
            'instructions':
                [
                    {'qubits': [2], 'texparams': ['0.5 \\pi', '0', '\\pi'],
                     'name': 'U', 'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [2, 5]},
                    {'memory': [5], 'name': 'measure', 'qubits': [5]},
                    {'qubits': [1], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [1, 4]},
                    {'memory': [4], 'name': 'measure', 'qubits': [4]},
                    {'qubits': [0], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [0, 3]},
                    {'memory': [3], 'name': 'measure', 'qubits': [3]},
                    {'name': 'barrier', 'qubits': [0, 1, 2]},
                    {'memory': [2], 'name': 'measure', 'qubits': [2]},
                    {'memory': [1], 'name': 'measure', 'qubits': [1]},
                    {'memory': [0], 'name': 'measure', 'qubits': [0]}
                ],
            'header':
                {
                    'n_qubits': 6,
                    'memory_slots': 6,
                    'qreg_sizes': [['q', 3], ['r', 3]],
                    'creg_sizes': [['c', 3], ['d', 3]],
                    'qubit_labels': [['q', 0], ['q', 1], ['q', 2], ['r', 0], ['r', 1], ['r', 2]], 
                    'clbit_labels': [['c', 0], ['c', 1], ['c', 2], ['d', 0], ['d', 1], ['d', 2]]
                }
        }

        self.assertEqual(json_circuit, expected_result)

    def test_from_dag_to_json_with_basis(self):
        """Test DagUnroller with JSON backend and a basis."""
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend(["cx", "u1", "u2", "u3"])).execute()
        dag_unroller = DagUnroller(dag_circuit, JsonBackend(["cx", "u1", "u2", "u3"]))
        json_circuit = dag_unroller.execute()
        expected_result = {
            'instructions':
                [
                    {'qubits': [2], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                     'name': 'u2'},
                    {'qubits': [2, 5], 'texparams': [], 'params': [], 'name': 'cx'},
                    {'qubits': [5], 'memory': [5], 'name': 'measure'},
                    {'qubits': [1], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                     'name': 'u2'},
                    {'qubits': [1, 4], 'texparams': [], 'params': [], 'name': 'cx'},
                    {'qubits': [4], 'memory': [4], 'name': 'measure'},
                    {'qubits': [0], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                     'name': 'u2'},
                    {'qubits': [0, 3], 'texparams': [], 'params': [], 'name': 'cx'},
                    {'qubits': [3], 'memory': [3], 'name': 'measure'},
                    {'qubits': [0, 1, 2], 'name': 'barrier'},
                    {'qubits': [2], 'memory': [2], 'name': 'measure'},
                    {'qubits': [1], 'memory': [1], 'name': 'measure'},
                    {'qubits': [0], 'memory': [0], 'name': 'measure'}
                ],
            'header':
                {
                    'n_qubits': 6,
                    'memory_slots': 6,
                    'qreg_sizes': [['q', 3], ['r', 3]],
                    'creg_sizes': [['c', 3], ['d', 3]],
                    'qubit_labels': [['q', 0], ['q', 1], ['q', 2], ['r', 0], ['r', 1], ['r', 2]], 
                    'clbit_labels': [['c', 0], ['c', 1], ['c', 2], ['d', 0], ['d', 1], ['d', 2]]
                }
        }
        self.maxDiff = None

        self.assertEqual(json_circuit, expected_result)
