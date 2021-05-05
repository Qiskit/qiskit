# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the converters."""

import os
import unittest

from qiskit.converters import ast_to_dag, circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import qasm
from qiskit.test import QiskitTestCase


class TestAstToDag(QiskitTestCase):
    """Test AST to DAG."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)
        self.dag = circuit_to_dag(self.circuit)

    def test_from_ast_to_dag(self):
        """Test Unroller.execute()"""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        ast = qasm.Qasm(os.path.join(qasm_dir, "example.qasm")).parse()
        dag_circuit = ast_to_dag(ast)
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
        expected_dag = circuit_to_dag(QuantumCircuit.from_qasm_str(expected_result))
        self.assertEqual(dag_circuit, expected_dag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
