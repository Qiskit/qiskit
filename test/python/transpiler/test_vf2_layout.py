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

"""Test the VF2Layout pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import VF2Layout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeTenerife, FakeRueschlikon


class TestVF2Layout(QiskitTestCase):
    """Tests the VF2Layout pass"""

    seed = 42

    def test_2q_circuit_2q_coupling(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap([[0, 1]]), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 0)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_3q_circuit_5q_coupling(self):
        """3 qubits in Tenerife, without considering the direction

            1                    1
          ↙ ↑                 /  |
        0 ← 2 ← 3           0 - qr1 - qr2
            ↑ ↙                 |   /
            4                   qr0
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap5), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 4)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 3)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_9q_circuit_16q_coupling(self):
        """9 qubits in Rueschlikon, without considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9

        q1_0 -  2 --- 3 --- q1_2 - 5 - q1_1 - 7 --- q0_1
          |    |      |      |     |     |    |      |
          0 - q1_3 - q0_0 - 13 - q0_3 - 11 - q1_4 - q0_2
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap16), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr0[0]], 14)
        self.assertEqual(layout[qr0[1]], 8)
        self.assertEqual(layout[qr0[2]], 9)
        self.assertEqual(layout[qr0[3]], 12)
        self.assertEqual(layout[qr1[0]], 1)
        self.assertEqual(layout[qr1[1]], 6)
        self.assertEqual(layout[qr1[2]], 4)
        self.assertEqual(layout[qr1[3]], 15)
        self.assertEqual(layout[qr1[4]], 10)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_2q_circuit_2q_coupling_sd(self):
        """A simple example, considering the direction
         0  -> 1
        qr1 -> qr0
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap([[0, 1]]), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 0)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_3q_circuit_5q_coupling_sd(self):
        """3 qubits in Tenerife, considering the direction
            1                       1
          ↙ ↑                    ↙  ↑
        0 ← 2 ← 3              0 ← qr2 ← qr1
            ↑ ↙                     ↑  ↙
            4                      qr0
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap5), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 4)
        self.assertEqual(layout[qr[1]], 3)
        self.assertEqual(layout[qr[2]], 2)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_3q_circuit_3q_coupling_non_induced(self):
        """A simple example, check for non-induced subgraph
            1         qr0 -> qr1 -> qr2
           / \
          0 - 2
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap([[0, 1], [1, 2], [2, 0]]), seed=-1)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")


    def test_4q_circuit_5q_coupling_loose_nodes(self):
        """4 qubits in Tenerife, with loose nodes

            1
          ↙ ↑                 /  |
        0 ← 2 ← 3           0 -  -
            ↑ ↙                 |   /
            4
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap5), seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 4)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 3)
        self.assertEqual(layout[qr[3]], 1)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_9q_circuit_16q_coupling_sd(self):
        """9 qubits in Rueschlikon, considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9

        q1[4] →   2   → q0[3] → 4   ← q1[0] ←   6  → q1[3] ← q0[0]
          ↓       ↑      ↓      ↓       ↑       ↓      ↓      ↑
        q0[2] ← q0[1] → 14  ← q1[1] ←  12   → q1[2] → 10   ←  9
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap16), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr0[0]], 8)
        self.assertEqual(layout[qr0[1]], 15)
        self.assertEqual(layout[qr0[2]], 0)
        self.assertEqual(layout[qr0[3]], 3)
        self.assertEqual(layout[qr1[0]], 5)
        self.assertEqual(layout[qr1[1]], 13)
        self.assertEqual(layout[qr1[2]], 11)
        self.assertEqual(layout[qr1[3]], 7)
        self.assertEqual(layout[qr1[4]], 1)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "solution found")

    def test_5q_circuit_16q_coupling_no_solution(self):
        """5 qubits in Rueschlikon, no solution

        q0[1] ↖     ↗ q0[2]
               q0[0]
        q0[3] ↙     ↘ q0[4]
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[0], qr[4])
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap16), seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertIsNone(layout)
        self.assertEqual(pass_.property_set["VF2Layout_stop_reason"], "nonexistent solution")

    def test_seed(self):
        """Different seeds yield different results"""
        seed_1 = 42
        seed_2 = 43

        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2
        dag = circuit_to_dag(circuit)

        pass_1 = VF2Layout(CouplingMap(cmap5), seed=seed_1)
        pass_1.run(dag)
        layout_1 = pass_1.property_set["layout"]

        pass_2 = VF2Layout(CouplingMap(cmap5), seed=seed_2)
        pass_2.run(dag)
        layout_2 = pass_2.property_set["layout"]

        self.assertNotEqual(layout_1, layout_2)


if __name__ == "__main__":
    unittest.main()
