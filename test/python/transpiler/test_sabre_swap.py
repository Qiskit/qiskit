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

"""Test the Sabre Swap pass"""

import unittest
from qiskit.circuit.library import CCXGate, HGate, Measure
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_trivial_case(self):
        """Test that an already mapped circuit is unchanged.
                  ┌───┐┌───┐
        q_0: ──■──┤ H ├┤ X ├──■──
             ┌─┴─┐└───┘└─┬─┘  │
        q_1: ┤ X ├──■────■────┼──
             └───┘┌─┴─┐       │
        q_2: ──■──┤ X ├───────┼──
             ┌─┴─┐├───┤       │
        q_3: ┤ X ├┤ X ├───────┼──
             └───┘└─┬─┘     ┌─┴─┐
        q_4: ───────■───────┤ X ├
                            └───┘
        """
        coupling = CouplingMap.from_ring(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        """Test lookahead mode's lookahead finds single SWAP gate.
                  ┌───┐
        q_0: ──■──┤ H ├───────────────
             ┌─┴─┐└───┘
        q_1: ┤ X ├──■────■─────────■──
             └───┘┌─┴─┐  │         │
        q_2: ──■──┤ X ├──┼────■────┼──
             ┌─┴─┐└───┘┌─┴─┐┌─┴─┐┌─┴─┐
        q_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├
             └───┘     └───┘└───┘└───┘
        q_4: ─────────────────────────

        """
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        pm = PassManager(SabreSwap(coupling, "lookahead"))
        new_qc = pm.run(qc)

        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_do_not_change_cm(self):
        """Coupling map should not change.
        See https://github.com/Qiskit/qiskit-terra/issues/5675"""
        cm_edges = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
        coupling = CouplingMap(cm_edges)

        passmanager = PassManager(SabreSwap(coupling))
        _ = passmanager.run(QuantumCircuit(1))

        self.assertEqual(set(cm_edges), set(coupling.get_edges()))

    def test_do_not_reorder_measurements(self):
        """Test that SabreSwap doesn't reorder measurements to the same classical bit.

        With the particular coupling map used in this test and the 3q ccx gate, the routing would
        invariably the measurements if the classical successors are not accurately tracked.
        Regression test of gh-7950."""
        coupling = CouplingMap([(0, 2), (2, 0), (1, 2), (2, 1)])
        qc = QuantumCircuit(3, 1)
        qc.compose(CCXGate().definition, [0, 1, 2], [])  # Unroll CCX to 2q operations.
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)  # This measure is 50/50 between the Z states.
        qc.measure(1, 0)  # This measure always overwrites with 0.
        passmanager = PassManager(SabreSwap(coupling))
        transpiled = passmanager.run(qc)

        last_h, h_qubits, _ = transpiled.data[-4]
        self.assertIsInstance(last_h, HGate)
        first_measure, first_measure_qubits, _ = transpiled.data[-2]
        second_measure, second_measure_qubits, _ = transpiled.data[-1]
        self.assertIsInstance(first_measure, Measure)
        self.assertIsInstance(second_measure, Measure)
        # Assert that the first measure is on the same qubit that the HGate was applied to, and the
        # second measurement is on a different qubit (though we don't care which exactly - that
        # depends a little on the randomisation of the pass).
        self.assertEqual(h_qubits, first_measure_qubits)
        self.assertNotEqual(h_qubits, second_measure_qubits)


if __name__ == "__main__":
    unittest.main()
