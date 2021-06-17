# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SabreLayout pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAlmaden


class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = FakeAlmaden().configuration().coupling_map

    def test_5q_circuit_20q_coupling(self):
        """Test finds layout for 5q circuit on 20q device."""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[3], qr[0])
        circuit.x(qr[2])
        circuit.cx(qr[4], qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr[0]], 10)
        self.assertEqual(layout[qr[1]], 12)
        self.assertEqual(layout[qr[2]], 7)
        self.assertEqual(layout[qr[3]], 11)
        self.assertEqual(layout[qr[4]], 13)

    def test_6q_circuit_20q_coupling(self):
        """Test finds layout for 6q circuit on 20q device."""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual(layout[qr0[0]], 2)
        self.assertEqual(layout[qr0[1]], 3)
        self.assertEqual(layout[qr0[2]], 10)
        self.assertEqual(layout[qr1[0]], 1)
        self.assertEqual(layout[qr1[1]], 7)
        self.assertEqual(layout[qr1[2]], 5)


if __name__ == "__main__":
    unittest.main()
