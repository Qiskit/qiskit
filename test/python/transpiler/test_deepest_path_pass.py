# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DeepestPath pass testing"""

import unittest
from qiskit.transpiler.passes import DeepestPath
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestDeepestPathPass(QiskitTestCase):
    """ Tests for DeepestPath methods. """

    def test_empty_dag_true(self):
        """Test the dag deepest path of an empty dag.
        """
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = DeepestPath()
        pass_.run(dag)
        self.assertListEqual(pass_.property_set['deepest_path'], [])

    def test_nonempty_dag_false(self):
        """Test the dag deepest path non-empty dag.
        path depth = 11 = 9 ops + 2 qubits at start and end of path
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])
        circuit.y(qr[1])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)

        pass_ = DeepestPath()
        pass_.run(dag)
        self.assertEqual(len(pass_.property_set['deepest_path']), 11)


if __name__ == '__main__':
    unittest.main()
