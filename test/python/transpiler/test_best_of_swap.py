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

"""Testing the "best of" approach with Swappers."""

import unittest
from qiskit.test import QiskitTestCase
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, Depth


class TestBestOfSwap(QiskitTestCase):
    """Testing the "best of" approach with Swappers."""

    def setUp(self):
        self.coupling_map = CouplingMap([[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]])
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[4])
        circuit.cx(qr[2], qr[4])
        circuit.cx(qr[0], qr[4])
        self.circuit = circuit

    def test_Lookahead_greater_Basic(self):
        """Choose Lookahead over Basic"""
        passmanager = PassManager()
        passmanager.append_best_of([BasicSwap(self.coupling_map), Depth()], 'depth')
        passmanager.append_best_of([LookaheadSwap(self.coupling_map), Depth()], 'depth')
        circuit_result = passmanager.run(self.circuit)
        passmanager = PassManager()
        passmanager.append(LookaheadSwap(self.coupling_map))
        circuit_lookahead = passmanager.run(self.circuit)

        self.assertEqual(circuit_result, circuit_lookahead)

    def test_Basic_less_Lookahead(self):
        """Choose Basic over Lookahead"""
        passmanager = PassManager()
        passmanager.append_best_of([BasicSwap(self.coupling_map), Depth()], 'depth')
        passmanager.append_best_of([LookaheadSwap(self.coupling_map), Depth()], 'depth',
                                   reverse=True)
        circuit_result = passmanager.run(self.circuit)
        passmanager = PassManager()
        passmanager.append(BasicSwap(self.coupling_map))
        circuit_basic = passmanager.run(self.circuit)

        self.assertEqual(circuit_result, circuit_basic)


if __name__ == '__main__':
    unittest.main()
