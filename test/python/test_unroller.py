# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,no-member,bad-continuation
#
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

from sys import version_info
import unittest

from qiskit import QuantumProgram
from qiskit import qasm
from qiskit.unroll import Unroller, DagUnroller, DAGBackend, JsonBackend
from .common import QiskitTestCase


class UnrollerTest(QiskitTestCase):
    """Test the Unroller."""

    def setUp(self):
        self.seed = 42
        self.qp = QuantumProgram()

    @unittest.skipIf(version_info.minor == 5, "Python 3.5 dictionaries don't preserve \
                                               insertion order, so we need to skip this \
                                               test, until fixed")
    def test_execute(self):
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit,
                                   DAGBackend())
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
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend(["cx", "u1", "u2", "u3"])).execute()
        dag_unroller = DagUnroller(dag_circuit, DAGBackend())
        expanded_dag_circuit = dag_unroller.expand_gates(["cx", "u1", "u2", "u3"])
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

    # We need to change the way we create clbit_labels and qubit_labels in order to
    # enable this test, as they are lists but the order is not important so comparing
    # them usually fails.
    @unittest.skip("Temporary skipping")
    def test_from_dag_to_json(self):
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend()).execute()
        dag_unroller = DagUnroller(dag_circuit, JsonBackend())
        json_circuit = dag_unroller.execute()
        expected_result = \
            {'operations':
                [{'qubits': [5], 'texparams': ['0.5 \\pi', '0', '\\pi'],
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
                 {'clbits': [0], 'name': 'measure', 'qubits': [0]}],
             'header':
                 {'number_of_clbits': 6,
                  'qubit_labels': [['r', 0], ['r', 1], ['r', 2], ['q', 0], ['q', 1], ['q', 2]],
                  'number_of_qubits': 6, 'clbit_labels': [['d', 3], ['c', 3]]
                  }
             }

        self.assertEqual(json_circuit, expected_result)

    # We need to change the way we create clbit_labels and qubit_labels in order to
    # enable this test, as they are lists but the order is not important so comparing
    # them usually fails.
    @unittest.skip("Temporary skipping")
    def test_from_dag_to_json_with_basis(self):
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        dag_circuit = Unroller(ast, DAGBackend(["cx", "u1", "u2", "u3"])).execute()
        dag_unroller = DagUnroller(dag_circuit, JsonBackend(["cx", "u1", "u2", "u3"]))
        json_circuit = dag_unroller.execute()
        expected_result = \
            {'operations':
                [{'qubits': [5], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                  'name': 'u2'},
                 {'qubits': [5, 2], 'texparams': [], 'params': [], 'name': 'cx'},
                 {'qubits': [2], 'clbits': [2], 'name': 'measure'},
                 {'qubits': [4], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                  'name': 'u2'},
                 {'qubits': [4, 1], 'texparams': [], 'params': [], 'name': 'cx'},
                 {'qubits': [1], 'clbits': [1], 'name': 'measure'},
                 {'qubits': [3], 'texparams': ['0', '\\pi'], 'params': [0.0, 3.141592653589793],
                  'name': 'u2'},
                 {'qubits': [3, 0], 'texparams': [], 'params': [], 'name': 'cx'},
                 {'qubits': [3, 4, 5], 'name': 'barrier'},
                 {'qubits': [5], 'clbits': [5], 'name': 'measure'},
                 {'qubits': [4], 'clbits': [4], 'name': 'measure'},
                 {'qubits': [3], 'clbits': [3], 'name': 'measure'},
                 {'qubits': [0], 'clbits': [0], 'name': 'measure'}],
             'header':
                 {'clbit_labels': [['d', 3], ['c', 3]],
                  'number_of_qubits': 6,
                  'qubit_labels': [['r', 0], ['r', 1], ['r', 2], ['q', 0], ['q', 1], ['q', 2]],
                  'number_of_clbits': 6
                  }
             }
        self.assertEqual(json_circuit, expected_result)
