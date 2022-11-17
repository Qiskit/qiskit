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

import unittest

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.providers.fake_provider import FakeRueschlikon
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals

from ..visualization.visualization import QiskitVisualizationTestCase, path_to_diagram_reference


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
        expected = "[[0, 1], [0, 2], [1, 2]]"
        self.assertEqual(expected, str(coupling))

    def test_coupling_distance(self):
        coupling_list = [(0, 1), (0, 2), (1, 2)]
        coupling = CouplingMap(coupling_list)
        self.assertTrue(coupling.is_connected())
        physical_qubits = coupling.physical_qubits
        result = coupling.distance(physical_qubits[0], physical_qubits[1])
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_add_physical_qubits(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_physical_qubit(0)
        self.assertEqual([0], coupling.physical_qubits)
        self.assertEqual("", str(coupling))

    def test_add_physical_qubits_not_int(self):
        coupling = CouplingMap()
        self.assertRaises(CouplingError, coupling.add_physical_qubit, "q")

    def test_add_edge(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_edge(0, 1)
        expected = "[[0, 1]]"
        self.assertEqual(expected, str(coupling))

    def test_neighbors(self):
        """Test neighboring qubits are found correctly."""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 0]])

        physical_qubits = coupling.physical_qubits
        self.assertEqual(set(coupling.neighbors(physical_qubits[0])), {1, 2})
        self.assertEqual(set(coupling.neighbors(physical_qubits[1])), {0})
        self.assertEqual(set(coupling.neighbors(physical_qubits[2])), set())

    def test_distance_error(self):
        """Test distance between unconnected physical_qubits."""
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
        """Generate a reduced map"""
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = CouplingMap(cmap)
        out = coupling_map.reduce([12, 11, 10, 9]).get_edges()
        ans = [(1, 2), (3, 2), (0, 1)]
        self.assertEqual(set(out), set(ans))

    def test_failed_reduced_map(self):
        """Generate a bad disconnected reduced map"""
        fake = FakeRueschlikon()
        cmap = fake.configuration().coupling_map
        coupling_map = CouplingMap(cmap)
        with self.assertRaises(CouplingError):
            coupling_map.reduce([12, 11, 10, 3])

    def test_symmetric_small_true(self):
        coupling_list = [[0, 1], [1, 0]]
        coupling = CouplingMap(coupling_list)

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
        coupling = CouplingMap(coupling_list)

        self.assertFalse(coupling.is_symmetric)

    def test_make_symmetric(self):
        coupling_list = [[0, 1], [0, 2]]
        coupling = CouplingMap(coupling_list)

        coupling.make_symmetric()
        edges = coupling.get_edges()

        self.assertEqual(set(edges), {(0, 1), (0, 2), (2, 0), (1, 0)})

    def test_full_factory(self):
        coupling = CouplingMap.from_full(4)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_line_factory(self):
        coupling = CouplingMap.from_line(4)
        edges = coupling.get_edges()
        expected = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        self.assertEqual(set(edges), set(expected))

    def test_grid_factory(self):
        coupling = CouplingMap.from_grid(2, 3)
        edges = coupling.get_edges()
        expected = [
            (0, 3),
            (0, 1),
            (3, 0),
            (3, 4),
            (1, 0),
            (1, 4),
            (1, 2),
            (4, 1),
            (4, 3),
            (4, 5),
            (2, 1),
            (2, 5),
            (5, 2),
            (5, 4),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_grid_factory_unidirectional(self):
        coupling = CouplingMap.from_grid(2, 3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [(0, 3), (0, 1), (3, 4), (1, 4), (1, 2), (4, 5), (2, 5)]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_hex_factory(self):
        coupling = CouplingMap.from_heavy_hex(3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 9),
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 9),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 12),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 12),
            (8, 18),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_hex_factory_bidirectional(self):
        coupling = CouplingMap.from_heavy_hex(3, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 9),
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 9),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 12),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 12),
            (8, 18),
            (9, 0),
            (9, 3),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
            (12, 5),
            (12, 8),
            (13, 0),
            (13, 1),
            (14, 1),
            (14, 2),
            (14, 10),
            (15, 3),
            (15, 4),
            (15, 11),
            (16, 4),
            (16, 5),
            (16, 10),
            (17, 6),
            (17, 7),
            (17, 11),
            (18, 7),
            (18, 8),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_square_factory(self):
        coupling = CouplingMap.from_heavy_square(3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 15),
            (1, 16),
            (2, 11),
            (3, 12),
            (3, 17),
            (4, 18),
            (5, 11),
            (6, 12),
            (6, 19),
            (7, 20),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
            (15, 1),
            (16, 2),
            (17, 4),
            (18, 5),
            (19, 7),
            (20, 8),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_square_factory_bidirectional(self):
        coupling = CouplingMap.from_heavy_square(3, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 15),
            (1, 15),
            (1, 16),
            (2, 11),
            (2, 16),
            (3, 12),
            (3, 17),
            (4, 17),
            (4, 18),
            (5, 11),
            (5, 18),
            (6, 12),
            (6, 19),
            (7, 19),
            (7, 20),
            (8, 20),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (11, 2),
            (11, 5),
            (12, 3),
            (12, 6),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
            (15, 0),
            (15, 1),
            (15, 9),
            (16, 1),
            (16, 2),
            (16, 10),
            (17, 3),
            (17, 4),
            (17, 9),
            (17, 13),
            (18, 4),
            (18, 5),
            (18, 10),
            (18, 14),
            (19, 6),
            (19, 7),
            (19, 13),
            (20, 7),
            (20, 8),
            (20, 14),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_hexagonal_lattice_2_2_factory(self):
        coupling = CouplingMap.from_hexagonal_lattice(2, 2, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_hexagonal_lattice_2_2_factory_bidirectional(self):
        coupling = CouplingMap.from_hexagonal_lattice(2, 2, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (14, 15),
            (15, 14),
            (0, 5),
            (5, 0),
            (2, 7),
            (7, 2),
            (4, 9),
            (9, 4),
            (6, 11),
            (11, 6),
            (8, 13),
            (13, 8),
            (10, 15),
            (15, 10),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_subgraph(self):
        coupling = CouplingMap.from_line(6, bidirectional=False)
        with self.assertWarns(DeprecationWarning):
            subgraph = coupling.subgraph([4, 2, 3, 5])
        self.assertEqual(subgraph.size(), 4)
        self.assertEqual([0, 1, 2, 3], subgraph.physical_qubits)
        edge_list = subgraph.get_edges()
        expected = [(0, 1), (1, 2), (2, 3)]
        self.assertEqual(expected, edge_list, f"{edge_list} does not match {expected}")

    def test_implements_iter(self):
        """Test that the object is implicitly iterable."""
        coupling = CouplingMap.from_line(3)
        expected = [(0, 1), (1, 0), (1, 2), (2, 1)]
        self.assertEqual(sorted(coupling), expected)


class CouplingVisualizationTest(QiskitVisualizationTestCase):
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    def test_coupling_draw(self):
        """Test that the coupling map drawing with respect to the reference file is correct."""
        cmap = CouplingMap([[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [2, 6]])
        image_ref = path_to_diagram_reference("coupling_map.png")
        image = cmap.draw()
        self.assertImagesAreEqual(image, image_ref, diff_tolerance=0.01)
