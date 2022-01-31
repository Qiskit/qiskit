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

"""Test the DenseLayout pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import DenseLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeTokyo


class TestDenseLayout(QiskitTestCase):
    """Tests the DenseLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = FakeTokyo().configuration().coupling_map

    def test_5q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 11)
        self.assertEqual(layout[qr[1]], 10)
        self.assertEqual(layout[qr[2]], 6)
        self.assertEqual(layout[qr[3]], 5)
        self.assertEqual(layout[qr[4]], 0)

    def test_6q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map."""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[0], qr1[2])
        circuit.cx(qr1[1], qr0[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr0[0]], 11)
        self.assertEqual(layout[qr0[1]], 10)
        self.assertEqual(layout[qr0[2]], 6)
        self.assertEqual(layout[qr1[0]], 5)
        self.assertEqual(layout[qr1[1]], 1)
        self.assertEqual(layout[qr1[2]], 0)


if __name__ == "__main__":
    unittest.main()
