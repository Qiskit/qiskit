# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.test.mock import FakeRueschlikon
from qiskit.test import QiskitTestCase


class CouplingTest(QiskitTestCase):

    def test_empty_coupling_class(self):
        coupling = CouplingMap()
        self.assertEqual(0, coupling.size())
        self.assertEqual([], coupling.physical_qubits)
        self.assertEqual([], coupling.get_edges())
        self.assertFalse(coupling.is_connected())
        self.assertEqual("", str(coupling))

    def test_coupling_str(self):
        coupling_list = [[0, 1], [0, 2], [1, 2]]
        coupling = CouplingMap(coupling_list)
        expected = ("[[0, 1], [0, 2], [1, 2]]")
        self.assertEqual(expected, str(coupling))

    def test_coupling_distance(self):
        coupling_list = [(0, 1), (0, 2), (1, 2)]
        coupling = CouplingMap(coupling_list)
        self.assertTrue(coupling.is_connected())
        physical_qubits = coupling.physical_qubits
        result = coupling.distance(physical_qubits[0], physical_qubits[1])
        self.assertEqual(1, result)

    def test_add_physical_qubits(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_physical_qubit(0)
        self.assertEqual([0], coupling.physical_qubits)
        self.assertEqual("", str(coupling))

    def test_add_physical_qubits_not_int(self):
        coupling = CouplingMap()
        self.assertRaises(CouplingError, coupling.add_physical_qubit, 'q')

    def test_add_edge(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_edge(0, 1)
        expected = ("[[0, 1]]")
        self.assertEqual(expected, str(coupling))

    def test_distance_error(self):
        """Test distance between unconected physical_qubits."""
        graph = CouplingMap()
        graph.add_physical_qubit(0)
        graph.add_physical_qubit(1)
        self.assertRaises(CouplingError, graph.distance, 0, 1)

    def test_init_with_couplinglist(self):
        coupling_list = [[0, 1], [1, 2]]
        coupling = CouplingMap(coupling_list)

        qubits_expected = [0, 1, 2]
        edges_expected = [(0, 1), (1, 2)]

        self.assertEqual(coupling.physical_qubits, qubits_expected)
        self.assertEqual(coupling.get_edges(), edges_expected)
        self.assertEqual(2, coupling.distance(0, 2))

    def test_successful_reduced_map(self):
        """Generate a reduced map
        """
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = CouplingMap(cmap)
        out = coupling_map.reduce([12, 11, 10, 9]).get_edges()
        ans = [(1, 2), (3, 2), (0, 1)]
        self.assertEqual(set(out), set(ans))

    def test_failed_reduced_map(self):
        """Generate a bad disconnected reduced map
        """
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = CouplingMap(cmap)
        with self.assertRaises(CouplingError):
            coupling_map.reduce([12, 11, 10, 3])
