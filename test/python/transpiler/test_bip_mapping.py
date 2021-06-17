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

"""Test the BIPMapping pass"""

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import BIPMapping


class TestBIPMapping(QiskitTestCase):
    """Tests the BIPMapping pass."""

    def test_empty(self):
        """Returns the original circuit if the circuit is empty."""
        coupling = CouplingMap([[0, 1]])
        circuit = QuantumCircuit(2)
        actual = BIPMapping(coupling)(circuit)
        self.assertEqual(actual, circuit)

    def test_no_two_qubit_gates(self):
        """Returns the original circuit if the circuit has no 2q-gates
        q0:--[H]--
        q1:-------
        CouplingMap map: [0]--[1]
        """
        coupling = CouplingMap([[0, 1]])

        circuit = QuantumCircuit(2)
        circuit.h(0)

        actual = BIPMapping(coupling)(circuit)

        self.assertEqual(actual, circuit)

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

        actual = BIPMapping(coupling)(circuit)

        q = QuantumRegister(3, name="q")
        expected = QuantumCircuit(q)
        expected.cx(q[1], q[0])
        expected.h(q[0])
        expected.cx(q[2], q[0])

        self.assertEqual(actual, circuit)

    def test_no_swap(self):
        """Adding no swap if not giving initial layout
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

        actual = BIPMapping(coupling)(circuit)

        q = QuantumRegister(3, name="q")
        expected = QuantumCircuit(q)
        expected.cx(q[2], q[0])

        self.assertEqual(actual, expected)

    def test_ignore_initial_layout(self):
        """Ignoring initial layout even when it is supplied"""
        coupling = CouplingMap([[0, 1], [0, 2]])

        circuit = QuantumCircuit(3)
        circuit.cx(1, 2)

        property_set = {"layout": Layout.generate_trivial_layout(*circuit.qubits)}
        actual = BIPMapping(coupling)(circuit, property_set)

        q = QuantumRegister(3, name="q")
        expected = QuantumCircuit(q)
        expected.cx(q[2], q[0])

        self.assertEqual(actual, expected)

    def test_can_map_measurements_correctly(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs."""
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[1])

        actual = BIPMapping(coupling)(circuit)

        q = QuantumRegister(3, "q")
        expected = QuantumCircuit(q, cr)
        expected.cx(q[2], q[0])
        expected.measure(q[2], cr[0])
        expected.measure(q[0], cr[1])  # <- changed due to initial layout change

        self.assertEqual(actual, expected)

    def test_never_modify_mapped_circuit(self):
        """Test that the mapping is idempotent.
        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        circuit = QuantumCircuit(3, 2)
        circuit.cx(1, 2)
        circuit.measure(1, 0)
        circuit.measure(2, 1)
        dag = circuit_to_dag(circuit)

        mapped_dag = BIPMapping(coupling).run(dag)
        remapped_dag = BIPMapping(coupling).run(mapped_dag)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_no_swap_multi_layer(self):
        """Can find the best layout for a circuit with multiple layers
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

        qr = QuantumRegister(4, name="qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[3])

        property_set = {}
        actual = BIPMapping(coupling, objective="depth")(circuit, property_set)
        actual_final_layout = property_set["final_layout"]

        q = QuantumRegister(4, name="q")
        expected = QuantumCircuit(q)
        expected.cx(q[2], q[1])
        expected.cx(q[1], q[0])
        expected_final_layout = Layout({qr[0]: 1, qr[1]: 2, qr[2]: 3, qr[3]: 0})

        # print(actual)
        # print(expected)

        self.assertEqual(actual, expected)
        self.assertEqual(actual_final_layout, expected_final_layout)

    def test_unmappable_cnots_in_a_layer(self):
        """Test mapping of a circuit with 2 cnots in a layer into T-shape coupling."""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.measure(qr, cr)

        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        actual = BIPMapping(coupling)(circuit)

        # Fails to map and returns the original circuit
        self.assertEqual(actual, circuit)

    def test_multi_cregs(self):
        """Test for multiple ClassicalRegisters."""
        qr = QuantumRegister(4, "qr")
        cr1 = ClassicalRegister(2, "c")
        cr2 = ClassicalRegister(2, "d")
        circuit = QuantumCircuit(qr, cr1, cr2)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr1[0])
        circuit.measure(qr[1], cr2[0])
        circuit.measure(qr[2], cr1[1])
        circuit.measure(qr[3], cr2[1])

        coupling = CouplingMap([[0, 1], [0, 2], [2, 3]])  # linear [1, 0, 2, 3]
        actual = BIPMapping(coupling, objective="depth")(circuit)

        q = QuantumRegister(4, name="q")
        expected = QuantumCircuit(q, cr1, cr2)
        expected.cx(q[1], q[0])
        expected.cx(q[3], q[2])
        expected.swap(q[2], q[3])
        expected.cx(q[0], q[2])
        expected.h(q[0])
        expected.cx(q[0], q[1])
        expected.barrier()
        expected.measure(q[0], cr2[0])  # cbit is changed
        expected.measure(q[1], cr1[0])  # cbit is changed
        expected.measure(q[2], cr1[1])
        expected.measure(q[3], cr2[1])

        print(actual)
        print(expected)
        self.assertEqual(actual, expected)

    def test_swaps_in_dummy_steps(self):
        """Test the case when swaps are inserted in dummy steps."""
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(2, 3)
        circuit.barrier()
        circuit.h([0, 1, 2, 3])
        circuit.barrier()
        circuit.cx(0, 3)
        circuit.cx(1, 2)
        circuit.barrier()
        circuit.cx(0, 2)
        circuit.cx(1, 3)

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        actual = BIPMapping(coupling, objective="depth")(circuit)

        q = QuantumRegister(4, name="q")
        expected = QuantumCircuit(q)
        expected.cx(1, 0)
        expected.cx(2, 3)
        expected.barrier()
        expected.h([0, 1, 2, 3])
        expected.barrier()
        expected.swap(1, 2)  # swap in a dummy step
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.barrier()
        expected.swap(0, 1)
        expected.swap(1, 2)  # swap in a dummy step
        expected.cx(1, 0)
        expected.cx(2, 3)

        # print(actual)
        # print(expected)

        self.assertEqual(actual, expected)
