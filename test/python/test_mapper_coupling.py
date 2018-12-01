# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

from qiskit import QuantumRegister
from qiskit.mapper import _coupling
from .common import QiskitTestCase


class CouplingTest(QiskitTestCase):

    def test_coupling_dict2list(self):
        input_dict = {0: [1, 2], 1: [2]}
        result = _coupling.coupling_dict2list(input_dict)
        expected = [[0, 1], [0, 2], [1, 2]]
        self.assertEqual(expected, result)

    def test_coupling_dict2list_empty_dict(self):
        self.assertIsNone(_coupling.coupling_dict2list({}))

    def test_coupling_list2dict(self):
        input_list = [[0, 1], [0, 2], [1, 2]]
        result = _coupling.coupling_list2dict(input_list)
        expected = {0: [1, 2], 1: [2]}
        self.assertEqual(expected, result)

    def test_coupling_list2dict_empty_list(self):
        self.assertIsNone(_coupling.coupling_list2dict([]))

    def test_empty_coupling_class(self):
        coupling = _coupling.Coupling()
        self.assertEqual(0, coupling.size())
        self.assertEqual([], coupling.get_qubits())
        self.assertEqual([], coupling.get_edges())
        self.assertFalse(coupling.connected())
        self.assertEqual("", str(coupling))

    def test_coupling_str(self):
        coupling_dict = {0: [1, 2], 1: [2]}
        coupling = _coupling.Coupling(coupling_dict)
        expected = ("qubits: q[0] @ 1, q[1] @ 2, q[2] @ 3\n"
                    "edges: q[0]-q[1], q[0]-q[2], q[1]-q[2]")
        self.assertEqual(expected, str(coupling))

    def test_coupling_compute_distance(self):
        coupling_dict = {0: [1, 2], 1: [2]}
        coupling = _coupling.Coupling(coupling_dict)
        self.assertTrue(coupling.connected())
        coupling.compute_distance()
        qubits = coupling.get_qubits()
        result = coupling.distance(qubits[0], qubits[1])
        self.assertEqual(1, result)

    def test_coupling_compute_distance_coupling_error(self):
        coupling = _coupling.Coupling()
        self.assertRaises(_coupling.CouplingError, coupling.compute_distance)

    def test_add_qubit(self):
        coupling = _coupling.Coupling()
        self.assertEqual("", str(coupling))
        coupling.add_qubit((QuantumRegister(1, 'q'), 0))
        self.assertEqual("qubits: q[0] @ 1", str(coupling))

    def test_add_qubit_not_tuple(self):
        coupling = _coupling.Coupling()
        self.assertRaises(_coupling.CouplingError, coupling.add_qubit, QuantumRegister(1, 'q0'))

    def test_add_qubit_tuple_incorrect_form(self):
        coupling = _coupling.Coupling()
        self.assertRaises(_coupling.CouplingError, coupling.add_qubit,
                          (QuantumRegister(1, 'q'), '0'))

    def test_add_edge(self):
        coupling = _coupling.Coupling()
        self.assertEqual("", str(coupling))
        coupling.add_edge((QuantumRegister(2, 'q'), 0), (QuantumRegister(1, 'q'), 1))
        expected = ("qubits: q[0] @ 1, q[1] @ 2\n"
                    "edges: q[0]-q[1]")
        self.assertEqual(expected, str(coupling))

    def test_distance_error(self):
        """Test distance method validation."""
        graph = _coupling.Coupling({0: [1, 2], 1: [2]})
        self.assertRaises(_coupling.CouplingError, graph.distance, (QuantumRegister(3, 'q0'), 0),
                          (QuantumRegister(3, 'q1'), 1))
