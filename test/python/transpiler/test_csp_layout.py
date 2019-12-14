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

"""Test the CSPLayout pass"""

import unittest
from time import process_time

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CSPLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeTenerife, FakeRueschlikon, FakeTokyo

try:
    import constraint  # pylint: disable=unused-import, import-error

    HAS_CONSTRAINT = True
except Exception:  # pylint: disable=broad-except
    HAS_CONSTRAINT = False


@unittest.skipIf(not HAS_CONSTRAINT, 'python-constraint not installed.')
class TestCSPLayout(QiskitTestCase):
    """Tests the CSPLayout pass"""
    seed = 42

    def test_2q_circuit_2q_coupling(self):
        """ A simple example, without considering the direction
          0 - 1
        qr0 - qr1
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap([[0, 1]]), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)

    def test_3q_circuit_5q_coupling(self):
        """ 3 qubits in Tenerife, without considering the direction
            qr1
           /  |
        qr0 - qr2 - 3
              |   /
               4
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap5), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)

    def test_9q_circuit_16q_coupling(self):
        """ 9 qubits in Rueschlikon, without considering the direction
        q0[1] - q0[0] - q1[3] - q0[3] - q1[0] - q1[1] - q1[2] - 8
          |       |       |       |       |       |       |     |
        q0[2] - q1[4] -- 14 ---- 13 ---- 12 ---- 11 ---- 10 --- 9
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, 'q0')
        qr1 = QuantumRegister(5, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr0[0]], 2)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr0[2]], 0)
        self.assertEqual(layout[qr0[3]], 4)
        self.assertEqual(layout[qr1[0]], 5)
        self.assertEqual(layout[qr1[1]], 6)
        self.assertEqual(layout[qr1[2]], 7)
        self.assertEqual(layout[qr1[3]], 3)
        self.assertEqual(layout[qr1[4]], 15)

    def test_2q_circuit_2q_coupling_sd(self):
        """ A simple example, considering the direction
         0  -> 1
        qr1 -> qr0
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap([[0, 1]]), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 0)

    def test_3q_circuit_5q_coupling_sd(self):
        """ 3 qubits in Tenerife, considering the direction
              qr0
            ↙  ↑
        qr2 ← qr1 ← 3
               ↑  ↙
               4
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap5), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 0)

    def test_9q_circuit_16q_coupling_sd(self):
        """ 9 qubits in Rueschlikon, considering the direction
         q0[1] → q0[0] → q1[3] → q0[3] ← q1[0] ← q1[1] → q1[2] ← 8
           ↓       ↑      ↓      ↓       ↑       ↓        ↓      ↑
         q0[2] ← q1[4] → 14  ←  13   ←  12   →  11   →   10   ←  9
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, 'q0')
        qr1 = QuantumRegister(5, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        self.assertEqual(layout[qr0[0]], 2)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr0[2]], 0)
        self.assertEqual(layout[qr0[3]], 4)
        self.assertEqual(layout[qr1[0]], 5)
        self.assertEqual(layout[qr1[1]], 6)
        self.assertEqual(layout[qr1[2]], 7)
        self.assertEqual(layout[qr1[3]], 3)
        self.assertEqual(layout[qr1[4]], 15)

    def test_5q_circuit_16q_coupling_no_solution(self):
        """ 5 qubits in Rueschlikon, no solution

          q0[1] ↖     ↗ q0[2]
                 q0[0]
          q0[3] ↙     ↘ q0[4]
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[0], qr[4])
        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertIsNone(layout)

    @staticmethod
    def create_hard_dag():
        """Creates a particularly hard circuit (returns its dag) for Tokyo"""
        qasm = """OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[20];
        cx q[13],q[12];
        cx q[6],q[0];
        cx q[5],q[10];
        cx q[10],q[7];
        cx q[5],q[12];
        cx q[2],q[15];
        cx q[16],q[18];
        cx q[6],q[4];
        cx q[10],q[3];
        cx q[11],q[10];
        cx q[18],q[16];
        cx q[5],q[12];
        cx q[4],q[0];
        cx q[18],q[16];
        cx q[2],q[15];
        cx q[7],q[8];
        cx q[9],q[6];
        cx q[16],q[17];
        cx q[9],q[3];
        cx q[14],q[12];
        cx q[2],q[15];
        cx q[1],q[16];
        cx q[5],q[3];
        cx q[8],q[12];
        cx q[2],q[1];
        cx q[5],q[3];
        cx q[13],q[5];
        cx q[12],q[14];
        cx q[12],q[13];
        cx q[6],q[4];
        cx q[15],q[18];
        cx q[15],q[18];
        """
        return circuit_to_dag(QuantumCircuit.from_qasm_str(qasm))

    def test_time_limit(self):
        """Hard to solve situations hit the time limit"""
        dag = TestCSPLayout.create_hard_dag()
        coupling_map = CouplingMap(FakeTokyo().configuration().coupling_map)
        pass_ = CSPLayout(coupling_map, call_limit=None, time_limit=1)

        start = process_time()
        pass_.run(dag)
        runtime = process_time() - start

        self.assertLess(runtime, 2)
        self.assertEqual(pass_.property_set['CSP_stop_reason'], 'time limit reached')

    def test_call_limit(self):
        """Hard to solve situations hit the call limit"""
        dag = TestCSPLayout.create_hard_dag()
        coupling_map = CouplingMap(FakeTokyo().configuration().coupling_map)
        pass_ = CSPLayout(coupling_map, call_limit=1, time_limit=None)

        start = process_time()
        pass_.run(dag)
        runtime = process_time() - start

        self.assertLess(runtime, 1)
        self.assertEqual(pass_.property_set['CSP_stop_reason'], 'call limit reached')

if __name__ == '__main__':
    unittest.main()
