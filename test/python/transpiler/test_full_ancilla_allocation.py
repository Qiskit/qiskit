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

"""Test the FullAncillaAllocation pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.test import QiskitTestCase


class TestFullAncillaAllocation(QiskitTestCase):
    """Tests the ExtendLayout pass"""

    def setUp(self):
        self.cmap5 = CouplingMap([[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]])

    def test_3q_circuit_5q_coupling(self):
        """Allocates 2 ancillas for a 3q circuit in a 5q coupling map

                    0 -> q0
        q0 -> 0     1 -> q1
        q1 -> 1  => 2 -> q2
        q2 -> 2     3 -> ancilla0
                    4 -> ancilla1
        """
        qr = QuantumRegister(3, 'q')
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[1] = qr[1]
        initial_layout[2] = qr[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set['layout'] = initial_layout

        pass_.run(dag)
        after_layout = pass_.property_set['layout']

        ancilla = QuantumRegister(2, 'ancilla')

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], qr[1])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], ancilla[0])
        self.assertEqual(after_layout[4], ancilla[1])

    def test_3q_with_holes_5q_coupling(self):
        """Allocates 3 ancillas for a 2q circuit on a 5q coupling, with holes

                       0 -> q0
        q0 -> 0        1 -> ancilla0
        q1 -> 2    =>  2 ->  q2
                       3 -> ancilla1
                       4 -> ancilla2
        """
        qr = QuantumRegister(2, 'q')
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[2] = qr[1]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set['layout'] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set['layout']

        ancilla = QuantumRegister(3, 'ancilla')

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], ancilla[0])
        self.assertEqual(after_layout[2], qr[1])
        self.assertEqual(after_layout[3], ancilla[1])
        self.assertEqual(after_layout[4], ancilla[2])

    def test_3q_out_of_order_5q_coupling(self):
        """Allocates 2 ancillas a 3q circuit on a 5q coupling map, out of order

                       0 <- q0
        q0 -> 0        1 <- ancilla0
        q1 -> 3   =>   2 <- q2
        q2 -> 2        3 <- q1
                       4 <- ancilla1
        """
        qr = QuantumRegister(3, 'q')
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[3] = qr[1]
        initial_layout[2] = qr[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set['layout'] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set['layout']

        ancilla = QuantumRegister(2, 'ancilla')

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], ancilla[0])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], qr[1])
        self.assertEqual(after_layout[4], ancilla[1])

    def test_name_collision(self):
        """Name collision during ancilla allocation."""
        qr_ancilla = QuantumRegister(3, 'ancilla')
        circuit = QuantumCircuit(qr_ancilla)
        circuit.h(qr_ancilla)
        dag = circuit_to_dag(circuit)

        initial_layout = Layout()
        initial_layout[0] = qr_ancilla[0]
        initial_layout[1] = qr_ancilla[1]
        initial_layout[2] = qr_ancilla[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set['layout'] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set['layout']

        qregs = set(v[0] for v in after_layout.get_virtual_bits().keys())
        self.assertEqual(2, len(qregs))
        self.assertIn(qr_ancilla, qregs)
        qregs.remove(qr_ancilla)
        other_reg = qregs.pop()
        self.assertEqual(len(other_reg), 2)
        self.assertRegex(other_reg.name, r'^ancilla\d+$')


if __name__ == '__main__':
    unittest.main()
