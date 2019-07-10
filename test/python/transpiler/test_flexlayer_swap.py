# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the FlexlayerSwap pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import FlexlayerSwap


class TestFlexlayerSwap(QiskitTestCase):
    """ Tests the FlexlayerSwap pass."""

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
         q0:--(+)-[H]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--

         CouplingMap map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[2], qr[0])

        dag = circuit_to_dag(circuit)
        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        """ Adding a swap
         q0:-------

         q1:---.---
               |
         q2:--(+)--

         CouplingMap map: [1]--[0]--[2]

         q0:--X--(+)--
              |   |
         q1:--|---.---
              |
         q2:--x-------

        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_flexlayer_swap_maps_measurements(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.

        Create a circuit with measures on q1 and q2, following a cx between q1 and q2.
        Since (1, 2) is not in the coupling, one of the two will be required to move.
        Verify that the mapped measure corresponds to one of the two possible layouts following
        the swap.
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[1])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])  # <- changed due to swap insertion

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_flexlayer_swap_doesnt_modify_mapped_circuit(self):
        """Test that flexlayer swap is idempotent.

        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[1])
        dag = circuit_to_dag(circuit)

        mapped_dag = FlexlayerSwap(coupling).run(dag)
        remapped_dag = FlexlayerSwap(coupling).run(mapped_dag)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_far_swap(self):
        """ A far swap that affects coming CXs.
         qr0:--(+)---.--
                |    |
         qr1:---.----|--
                     |
         qr2:--------|--
                     |
         qr3:-------(+)-

         CouplingMap map: [0]--[1]--[2]--[3]

         qr0:-(+)-X----------
               |  |
         qr1:--.--X--X-------
                     |
         qr2:--------X--.----
                        |
         qr3:----------(+)---

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[1], qr[0])
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == '__main__':
    unittest.main()
