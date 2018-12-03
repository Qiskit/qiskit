# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

from qiskit.mapper import Coupling, CouplingError
from .common import QiskitTestCase


class CouplingTest(QiskitTestCase):

    def test_empty_coupling_class(self):
        coupling = Coupling()
        self.assertEqual(0, coupling.size())
        self.assertEqual([], coupling.physical_qubits)
        self.assertEqual([], coupling.get_edges())
        self.assertFalse(coupling.is_connected())
        self.assertEqual("", str(coupling))

    def test_coupling_str(self):
        coupling_dict = {0: [1, 2], 1: [2]}
        coupling = Coupling(coupling_dict)
        expected = ("[(0, 1), (0, 2), (1, 2)]")
        self.assertEqual(expected, str(coupling))

    def test_coupling_distance(self):
        coupling_dict = {0: [1, 2], 1: [2]}
        coupling = Coupling(coupling_dict)
        self.assertTrue(coupling.is_connected())
        physical_qubits = coupling.physical_qubits
        result = coupling.distance(physical_qubits[0], physical_qubits[1])
        self.assertEqual(1, result)

    def test_add_physical_qubits(self):
        coupling = Coupling()
        self.assertEqual("", str(coupling))
        coupling.add_physical_qubit(0)
        self.assertEqual([0], coupling.physical_qubits)
        self.assertEqual("", str(coupling))

    def test_add_physical_qubits_not_int(self):
        coupling = Coupling()
        self.assertRaises(CouplingError, coupling.add_physical_qubit, 'q')

    def test_add_edge(self):
        coupling = Coupling()
        self.assertEqual("", str(coupling))
        coupling.add_edge(0, 1)
        expected = ("[(0, 1)]")
        self.assertEqual(expected, str(coupling))

    def test_distance_error(self):
        """Test distance between unconected physical_qubits."""
        graph = Coupling()
        graph.add_physical_qubit(0)
        graph.add_physical_qubit(1)
        self.assertRaises(CouplingError, graph.distance, 0, 1)
