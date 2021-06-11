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

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import MIPMapping


class TestMIPMapping(QiskitTestCase):
    """ Tests the MIPMapping pass."""

    def test_no_two_qubit_gates(self):
        """No need for mapping, the CX are distance 1 to each other
         q0:--[H]--
         q1:-------
         CouplingMap map: [0]--[1]
        """
        coupling = CouplingMap([[0, 1]])

        circuit = QuantumCircuit(2)
        circuit.h(0)

        actual = MIPMapping(coupling)(circuit)

        q = QuantumRegister(2, name='q')
        expected = QuantumCircuit(q)
        expected.h(q[0])

        self.assertEqual(actual, expected)

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

        circuit = QuantumCircuit(3)
        circuit.cx(1, 0)
        circuit.h(0)
        circuit.cx(2, 0)

        actual = MIPMapping(coupling)(circuit)

        q = QuantumRegister(3, name='q')
        expected = QuantumCircuit(q)
        expected.cx(q[1], q[0])
        expected.h(q[0])
        expected.cx(q[2], q[0])

        self.assertEqual(actual, circuit)

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

        circuit = QuantumCircuit(3)
        circuit.cx(1, 2)

        actual = MIPMapping(coupling)(circuit)

        q = QuantumRegister(3, name='q')
        expected = QuantumCircuit(q)
        expected.cx(q[2], q[0])

        self.assertEqual(actual, expected)

    def test_ignore_initial_layout(self):
        """ Ignoring initial layout even when it is supplied
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        circuit = QuantumCircuit(3)
        circuit.cx(1, 2)

        property_set = {"layout": Layout.generate_trivial_layout(*circuit.qubits)}
        actual = MIPMapping(coupling)(circuit, property_set)

        q = QuantumRegister(3, name='q')
        expected = QuantumCircuit(q)
        expected.cx(q[2], q[0])

        self.assertEqual(actual, expected)

    def test_can_map_measurements_correctly(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.
        Create a circuit with measures on q1 and q2, following a cx between q1 and q2.
        Since (1, 2) is not in the coupling, one of the two will be required to move.
        Verify that the mapped measure corresponds to one of the two possible layouts following
        the swap.
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[1])

        property_set = {"layout": Layout.generate_trivial_layout(qr)}
        actual = MIPMapping(coupling)(circuit, property_set)

        q = QuantumRegister(3, 'q')
        expected = QuantumCircuit(q, cr)
        expected.cx(q[2], q[0])
        expected.measure(q[2], cr[0])
        expected.measure(q[0], cr[1])  # <- changed due to swap insertion

        self.assertEqual(actual, expected)

    def test_never_modify_mapped_circuit(self):
        """Test that mip mapping is idempotent.
        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        circuit = QuantumCircuit(3, 2)
        circuit.cx(1, 2)
        circuit.measure(1, 0)
        circuit.measure(2, 1)
        dag = circuit_to_dag(circuit)

        mapped_dag = MIPMapping(coupling).run(dag)
        remapped_dag = MIPMapping(coupling).run(mapped_dag)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_no_swap_multi_layer(self):
        """ Can find the best layout for a circuit with multiple layers
         qr0:--(+)---.--
                |    |
         qr1:---.----|--
                     |
         qr2:--------|--
                     |
         qr3:-------(+)-
         CouplingMap map: [0]--[1]--[2]--[3]
         q0:--.-------
              |
         q1:--X---.---
                  |
         q2:------X---

         q3:----------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, name='qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[3])

        property_set = {}
        actual = MIPMapping(coupling)(circuit, property_set)
        actual_final_layout = property_set["final_layout"]

        q = QuantumRegister(4, name='q')
        expected = QuantumCircuit(q)
        expected.cx(q[0], q[1])
        expected.cx(q[1], q[2])
        expected_final_layout = Layout({qr[0]: 1, qr[1]: 0, qr[2]: 3, qr[3]: 2})

        self.assertEqual(actual, expected)
        self.assertEqual(actual_final_layout, expected_final_layout)

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
