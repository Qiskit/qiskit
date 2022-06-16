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

from qiskit.transpiler import SubCouplingMap, CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.test.mock import FakeRueschlikon
from qiskit.test import QiskitTestCase


class SubCouplingTest(QiskitTestCase):
    def test_empty(self):
        coupling = SubCouplingMap()
        self.assertEqual(0, coupling.size())
        self.assertEqual([], coupling.physical_qubits)
        self.assertEqual([], coupling.get_edges())
        self.assertFalse(coupling.is_connected())
        self.assertEqual("[]", str(coupling))

    def test_create_from_edge_list_with_partial_qubits(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        self.assertEqual([1, 3, 5], coupling.physical_qubits)
        self.assertEqual([(1, 3), (3, 5)], coupling.get_edges())

    def test_create_from_coupling(self):
        original_coupling = CouplingMap.from_grid(2, 3)
        coupling = SubCouplingMap(original_coupling)
        self.assertEqual(original_coupling.physical_qubits, coupling.physical_qubits)
        self.assertEqual(original_coupling.get_edges(), coupling.get_edges())

    def test_create_from_coupling_with_qubit_list(self):
        qubit_list = [1, 3, 4, 5]
        expected_edges = [(1, 4), (4, 1), (3, 4), (4, 3), (4, 5), (5, 4)]
        original_coupling = CouplingMap.from_grid(2, 3)
        coupling = SubCouplingMap(original_coupling, qubit_list=qubit_list)
        self.assertEqual(qubit_list, coupling.physical_qubits)
        self.assertEqual(expected_edges, coupling.get_edges())
        # must not affect original_coupling
        self.assertEqual(
            CouplingMap.from_grid(2, 3).physical_qubits, original_coupling.physical_qubits
        )
        self.assertEqual(CouplingMap.from_grid(2, 3).get_edges(), original_coupling.get_edges())

    def test_add_physical_qubits(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        coupling.add_physical_qubit(7)
        coupling.add_physical_qubit(8)
        self.assertEqual([1, 3, 5, 7, 8], coupling.physical_qubits)
        self.assertEqual([(1, 3), (3, 5)], coupling.get_edges())

    def test_add_physical_qubit_to_empty(self):
        coupling = SubCouplingMap()
        coupling.add_physical_qubit(0)
        self.assertEqual([0], coupling.physical_qubits)
        self.assertEqual([], coupling.get_edges())

    def test_fail_to_add_duplicate_qubit(self):
        coupling = SubCouplingMap()
        coupling.add_physical_qubit(2)
        with self.assertRaises(CouplingError):
            coupling.add_physical_qubit(2)
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        with self.assertRaises(CouplingError):
            coupling.add_physical_qubit(5)

    def test_fail_to_add_physical_qubits_not_int(self):
        coupling = SubCouplingMap()
        with self.assertRaises(CouplingError):
            coupling.add_physical_qubit("q")

    def test_remove_physical_qubit(self):
        coupling = SubCouplingMap([(1, 3), (3, 5), (1, 5)])
        coupling.remove_physical_qubit(5)
        self.assertEqual([1, 3], coupling.physical_qubits)
        self.assertEqual([(1, 3)], coupling.get_edges())

    def test_fail_to_remove_physical_qubit_not_in_graph(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        with self.assertRaises(CouplingError):
            coupling.remove_physical_qubit(7)

    def test_add_edges(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        coupling.add_edge(1, 5)
        coupling.add_edge(5, 7)
        self.assertEqual([1, 3, 5, 7], coupling.physical_qubits)
        self.assertEqual([(1, 3), (3, 5), (1, 5), (5, 7)], coupling.get_edges())

    def test_add_edge_to_empty(self):
        coupling = SubCouplingMap()
        coupling.add_edge(1, 3)
        self.assertEqual([1, 3], coupling.physical_qubits)
        self.assertEqual([(1, 3)], coupling.get_edges())

    def test_can_add_an_isolated_edge(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        coupling.add_edge(7, 8)
        self.assertEqual([1, 3, 5, 7, 8], coupling.physical_qubits)
        self.assertEqual([(1, 3), (3, 5), (7, 8)], coupling.get_edges())

    def test_remove_edges(self):
        coupling = SubCouplingMap([(1, 3), (3, 5), (1, 5), (5, 7)])
        coupling.remove_edge(3, 5)
        coupling.remove_edge(5, 7)
        self.assertEqual([1, 3, 5, 7], coupling.physical_qubits)
        self.assertEqual([(1, 3), (1, 5)], coupling.get_edges())

    def test_remove_edge_may_create_isolated_qubit(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        coupling.remove_edge(3, 5)
        self.assertEqual([1, 3, 5], coupling.physical_qubits)
        self.assertEqual([(1, 3)], coupling.get_edges())

    def test_fail_to_remove_edge_not_in_graph(self):
        coupling = SubCouplingMap([(1, 3), (3, 5)])
        with self.assertRaises(CouplingError):
            coupling.remove_edge(3, 1)
        with self.assertRaises(CouplingError):
            coupling.remove_edge(1, 5)

    def test_neighbors(self):
        """Test neighboring qubits are found correctly."""
        coupling = SubCouplingMap([[1, 3], [1, 5], [3, 1]])
        self.assertEqual(set(coupling.neighbors(1)), {3, 5})
        self.assertEqual(set(coupling.neighbors(3)), {1})
        self.assertEqual(set(coupling.neighbors(5)), set())

    def test_neighbors_of_qubit_not_in_graph_fails(self):
        coupling = SubCouplingMap([[1, 3], [1, 5], [3, 1]])
        with self.assertRaises(CouplingError):
            coupling.neighbors(2)

    def test_distance(self):
        coupling_list = [(0, 1), (0, 2), (1, 2), (2, 3)]
        coupling = SubCouplingMap(coupling_list, qubit_list=[1, 2, 3])
        self.assertTrue(coupling.is_connected())
        self.assertEqual(2, coupling.distance(1, 3))
        self.assertEqual(2, coupling.distance(3, 1))  # undirecte distance

    def test_distance_between_unconnected_qubits_fails(self):
        """Test distance between unconnected physical_qubits."""
        coupling = SubCouplingMap()
        coupling.add_physical_qubit(1)
        coupling.add_physical_qubit(2)
        with self.assertRaises(CouplingError):
            coupling.distance(1, 2)

    def test_distance_between_unknown_qubits_fails(self):
        """Test distance between physical_qubits not in the graph."""
        coupling = SubCouplingMap([(1, 2)])
        with self.assertRaises(CouplingError):
            coupling.distance(0, 2)
        with self.assertRaises(CouplingError):
            coupling.distance(1, 9)

    def test_shortest_undirected_path(self):
        """Test with a graph and (target, source) with different undirected and directed paths"""
        coupling = SubCouplingMap([(0, 1), (1, 2), (2, 3), (2, 0)])
        actual = coupling.shortest_undirected_path(0, 3)
        self.assertEqual([0, 2, 3], actual)

    def test_shortest_undirected_path_between_unknonw_qubits_fails(self):
        coupling = SubCouplingMap([(0, 1), (1, 2), (2, 3), (2, 0)])
        with self.assertRaises(CouplingError):
            coupling.shortest_undirected_path(9, 3)
        with self.assertRaises(CouplingError):
            coupling.shortest_undirected_path(0, 9)

    def test_largest_connected_component(self):
        coupling = SubCouplingMap([(0, 1), (1, 2), (2, 3), (3, 1)])
        coupling.remove_edge(0, 1)
        actual = coupling.largest_connected_component()
        self.assertEqual([1, 2, 3], actual)

    def test_reduce(self):
        """Generate a reduced map"""
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = SubCouplingMap(cmap)
        out = coupling_map.reduce([12, 11, 10, 9]).get_edges()
        ans = [(9, 10), (11, 10), (12, 11)]
        self.assertEqual(set(out), set(ans))

    def test_fail_to_reduce(self):
        """Generate a bad disconnected reduced map"""
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = SubCouplingMap(cmap)
        with self.assertRaises(CouplingError):
            coupling_map.reduce([12, 11, 10, 3])

    def test_symmetric_small_true(self):
        coupling_list = [[0, 1], [1, 0]]
        coupling = SubCouplingMap(coupling_list)

        self.assertTrue(coupling.is_symmetric)

    def test_symmetric_big_false(self):
        coupling_list = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [9, 8],
            [9, 10],
            [7, 8],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]
        coupling = SubCouplingMap(coupling_list)

        self.assertFalse(coupling.is_symmetric)

    def test_make_symmetric(self):
        coupling_list = [[1, 3], [1, 5]]
        coupling = SubCouplingMap(coupling_list)

        coupling.make_symmetric()
        edges = coupling.get_edges()

        self.assertEqual(set(edges), {(1, 3), (1, 5), (5, 1), (3, 1)})
