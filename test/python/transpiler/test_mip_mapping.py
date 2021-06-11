# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the MIPMapping pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import MIPMapping


class TestMIPMapping(QiskitTestCase):
    """ Tests the MIPMapping pass."""

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

        actual = MIPMapping(coupling)(circuit)

        self.assertEqual(circuit_to_dag(actual), circuit_to_dag(circuit))

    def test_no_swap(self):
        """ Adding no swap if not giving initial layout
         q0:-------
         q1:---.---
               |
         q2:--(+)--
         CouplingMap map: [1]--[0]--[2]
         initial_layout: None
         q0:--(+)--
               |
         q1:---|---
               |
         q2:---.---
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])

        actual = MIPMapping(coupling)(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[2], qr[0])

        self.assertEqual(actual, expected)

    def test_a_single_swap(self):
        """ Adding a swap if fixing initial layout
         q0:-------
         q1:---.---
               |
         q2:--(+)--
         CouplingMap map: [1]--[0]--[2]
         initial_layout: trivial layout
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

        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        self.assertEqual(actual, expected)

    def test_can_map_measurements_correctly(self):
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

        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        expected = QuantumCircuit(qr, cr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])  # <- changed due to swap insertion

        self.assertEqual(actual, expected)

    def test_never_modify_mapped_circuit(self):
        """Test that mip mapping is idempotent.
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

        mapped_dag = MIPMapping(coupling).run(dag)
        remapped_dag = MIPMapping(coupling).run(mapped_dag)

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
         qr0:-(+)--X-------
               |   |
         qr1:--.---X-------

         qr2:------X---.---
                   |   |
         qr3:------X--(+)--
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[3])

        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        expected = QuantumCircuit(qr)
        expected.cx(qr[1], qr[0])
        expected.swap(qr[0], qr[1])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[1], qr[2])

        self.assertEqual(actual, expected)

    def test_search_4qcx4h1(self):
        """Test for 4 cx gates and 1 h gate in a 4q circuit.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier()
        circuit.measure(qr, cr)

        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])
        expected.barrier()
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[1])
        expected.measure(qr[0], cr[2])
        expected.measure(qr[3], cr[3])

        print(actual)
        print(expected)

        self.assertEqual(actual, expected)

    def test_search_multi_creg(self):
        """Test for multiple ClassicalRegisters.
        """
        qr = QuantumRegister(4, 'q')
        cr1 = ClassicalRegister(2, 'c')
        cr2 = ClassicalRegister(2, 'd')
        circuit = QuantumCircuit(qr, cr1, cr2)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr1[0])
        circuit.measure(qr[1], cr1[1])
        circuit.measure(qr[2], cr2[0])
        circuit.measure(qr[3], cr2[1])

        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        expected = QuantumCircuit(qr, cr1, cr2)
        expected.cx(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])
        expected.barrier()
        expected.measure(qr[0], cr2[0])
        expected.measure(qr[1], cr1[0])
        expected.measure(qr[2], cr1[1])
        expected.measure(qr[3], cr2[1])

        print(actual)
        print(expected)

        self.assertEqual(actual, expected)
