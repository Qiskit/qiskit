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
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import FakeAlmaden
from qiskit.providers.fake_provider import FakeKolkata


class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = FakeAlmaden().configuration().coupling_map

    def test_5q_circuit_20q_coupling(self):
        """Test finds layout for 5q circuit on 20q device."""
        #                ┌───┐
        # q_0: ──■───────┤ X ├───────────────
        #        │       └─┬─┘┌───┐
        # q_1: ──┼────■────┼──┤ X ├───────■──
        #      ┌─┴─┐  │    │  ├───┤┌───┐┌─┴─┐
        # q_2: ┤ X ├──┼────┼──┤ X ├┤ X ├┤ X ├
        #      └───┘┌─┴─┐  │  └───┘└─┬─┘└───┘
        # q_3: ─────┤ X ├──■─────────┼───────
        #           └───┘            │
        # q_4: ──────────────────────■───────
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
        self.assertEqual(layout[qr[0]], 11)
        self.assertEqual(layout[qr[1]], 6)
        self.assertEqual(layout[qr[2]], 12)
        self.assertEqual(layout[qr[3]], 5)
        self.assertEqual(layout[qr[4]], 13)

    def test_6q_circuit_20q_coupling(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
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
        self.assertEqual(layout[qr0[0]], 8)
        self.assertEqual(layout[qr0[1]], 2)
        self.assertEqual(layout[qr0[2]], 10)
        self.assertEqual(layout[qr1[0]], 3)
        self.assertEqual(layout[qr1[1]], 12)
        self.assertEqual(layout[qr1[2]], 11)

    def test_layout_with_classical_bits(self):
        """Test sabre layout with classical bits recreate from issue #8635."""
        qc = QuantumCircuit.from_qasm_str(
            """
OPENQASM 2.0;
include "qelib1.inc";
qreg q4833[1];
qreg q4834[6];
qreg q4835[7];
creg c982[2];
creg c983[2];
creg c984[2];
rzz(0) q4833[0],q4834[4];
cu(0,-6.1035156e-05,0,1e-05) q4834[1],q4835[2];
swap q4834[0],q4834[2];
cu(-1.1920929e-07,0,-0.33333333,0) q4833[0],q4834[2];
ccx q4835[2],q4834[5],q4835[4];
measure q4835[4] -> c984[0];
ccx q4835[2],q4835[5],q4833[0];
measure q4835[5] -> c984[1];
measure q4834[0] -> c982[1];
u(10*pi,0,1.9) q4834[5];
measure q4834[3] -> c984[1];
measure q4835[0] -> c982[0];
rz(0) q4835[1];
"""
        )
        res = transpile(qc, FakeKolkata(), layout_method="sabre", seed_transpiler=1234)
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout
        self.assertEqual(layout[qc.qubits[0]], 14)
        self.assertEqual(layout[qc.qubits[1]], 19)
        self.assertEqual(layout[qc.qubits[2]], 7)
        self.assertEqual(layout[qc.qubits[3]], 13)
        self.assertEqual(layout[qc.qubits[4]], 6)
        self.assertEqual(layout[qc.qubits[5]], 16)
        self.assertEqual(layout[qc.qubits[6]], 18)
        self.assertEqual(layout[qc.qubits[7]], 26)


if __name__ == "__main__":
    unittest.main()
