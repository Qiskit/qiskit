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

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test cases for permutation.tree package"""

from typing import TypeVar, Dict

import networkx as nx
from numpy import random

from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing import tree, util, Permutation

_V = TypeVar('_V')


class TestPermutationTree(QiskitTestCase):
    """The test cases"""

    def test_permute_tree_tiny(self) -> None:
        """Test permuting on trees for a tiny 3-ary, 1-height tree."""
        graph = nx.balanced_tree(3, 1)
        permutation = {
            0: 0,
            1: 2,
            2: 3,
            3: 1
            }
        self.permutation_test(graph, permutation)

    def test_permute_tree_small(self) -> None:
        """Test permuting one a tree where every node i is moved to node n-i."""
        graph = nx.balanced_tree(3, 1)
        graph.add_edges_from([
            (2, 4),
            (2, 5),
            (3, 6)
            ])
        permutation = {i: 6 - i for i in range(len(graph.nodes))}
        self.permutation_test(graph, permutation)

    def test_permute_tree_bug1(self) -> None:
        """Test a specific permutation that gave rise to an error.

        The error relates to not having any improper root nodes to swap with.
        """
        graph = nx.balanced_tree(3, 2)
        permutation = {0: 2, 1: 6, 2: 10, 3: 3, 4: 0, 5: 12, 6: 8, 7: 1, 8: 5, 9: 4, 10: 9, 11: 7,
                       12: 11}
        self.permutation_test(graph, permutation)

    def test_permute_tree_bug2(self) -> None:
        """Test a specific permutation that gave rise to an infinite loop."""
        graph = nx.balanced_tree(3, 3)
        permutation = {0: 24, 1: 9, 2: 36, 3: 0, 4: 35, 5: 31, 6: 14, 7: 29, 8: 19, 9: 16, 10: 3,
                       11: 27, 12: 18, 13: 7, 14: 8, 15: 5, 16: 32, 17: 20, 18: 11, 19: 37, 20: 12,
                       21: 25, 22: 28, 23: 22, 24: 10, 25: 1, 26: 34, 27: 26, 28: 15, 29: 23, 30: 6,
                       31: 33, 32: 17, 33: 39, 34: 4, 35: 21, 36: 30, 37: 38, 38: 2, 39: 13}
        self.permutation_test(graph, permutation)

    def test_permute_tree_bug3(self) -> None:
        """Test a specific permutation that gave rise to an error.

        The error relates to how subtrees were handled, even if input tree was pure.
        """
        graph = nx.balanced_tree(3, 2)
        permutation = {12: 11, 4: 6, 10: 3, 7: 2, 11: 10, 8: 9, 5: 1, 3: 0,
                       9: 7, 6: 5, 1: 4, 2: 8, 0: 12}
        self.permutation_test(graph, permutation)

    def test_tree_random_permutation(self) -> None:
        """Test a random permutation on a large 3-ary balanced tree with height 6."""
        graph = nx.balanced_tree(3, 6)
        nodecount = len(graph.nodes)
        rand_perm = random.permutation(range(nodecount))
        permutation = {i: rand_perm[i] for i in range(nodecount)}
        self.permutation_test(graph, permutation)

    def test_permute_random_tree(self) -> None:
        """Test a random permutation on a large randomly generated tree of size 10^3."""
        graph = nx.random_tree(10**3)  # This is actually quite slow
        nodecount = len(graph.nodes)
        rand_perm = random.permutation(range(nodecount))
        permutation = {i: rand_perm[i] for i in range(nodecount)}
        self.permutation_test(graph, permutation)

    def permutation_test(self, graph: nx.Graph, permutation: Permutation[_V]) -> None:
        """Test if permute correctly permutes the given tree graph and a permutation."""
        swaps = tree.permute(graph, permutation)
        util.swap_permutation(swaps, permutation)
        self.assertEqual({i: i for i in range(len(graph.nodes))}, permutation)

    def test_move_improper_basic(self) -> None:
        """Test moving improper pebbles on a simple graph. One even, one odd pebble."""
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (3, 5),
            (3, 6)
            ])
        permutation = {
            0: 0,
            1: 1,
            2: 10,
            3: 3,
            4: 4,
            5: 11,
            6: 6
            }
        graph_tree = tree.Tree(root=3, graph=nx.dfs_tree(graph, 3),
                               pebbles=TestPermutationTree.permutation_to_pebbles(permutation))

        mover = graph_tree.move_improper()
        out = next(mover)
        self.assertEqual([(3, 5)], out)
        out = next(mover)
        self.assertEqual([(0, 2)], out)

    def test_move_improper_basic_2(self) -> None:
        """Test moving two improper pebbles, where one blocks another from being moved."""
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (3, 5),
            (3, 6)
            ])
        permutation = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 11,
            6: 10
            }
        graph_tree = tree.Tree(root=3, graph=nx.dfs_tree(graph, 3),
                               pebbles=TestPermutationTree.permutation_to_pebbles(permutation))

        mover = graph_tree.move_improper()
        out = next(mover)
        self.assertEqual(1, len(out))
        self.assertIn(out[0], {(3, 5), (3, 6)})
        out = next(mover)
        self.assertEqual([], out)

    def test_move_improper_multiple(self) -> None:
        """Test moving multiple pebbles where two can be moved at the same time step.

        One more pebble can be moved into the gap by the first time step.
        """
        graph = nx.path_graph(6)
        permutation = {
            0: 0,
            1: 10,
            2: 11,
            3: 12,
            4: 4,
            5: 14,
            }
        graph_tree = tree.Tree(root=0, graph=nx.dfs_tree(graph, 0),
                               pebbles=TestPermutationTree.permutation_to_pebbles(permutation))

        mover = graph_tree.move_improper()
        out = next(mover)
        self.assertEqual([(0, 1), (4, 5)], out)
        graph_tree.apply_internal_swaps(out)
        out = next(mover)
        self.assertEqual([(1, 2)], out)

    def test_centroid_basic(self) -> None:
        """Test finding the centroid on a small tree."""
        graph = nx.Graph()
        graph.add_edges_from([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (3, 5),
            (3, 6)
            ])

        centroid = tree.centroid(graph)  # type: int
        self.assertEqual(3, centroid)

    def test_medium_regular_tree(self) -> None:
        """Test finding the centroid on a large balanced 3-ary tree with random node ids."""
        graph = nx.balanced_tree(3, 3)
        # Relabel the nodes to a new values to hide the root
        randomised = random.permutation([i for i in graph.nodes])
        relabeling = {i: randomised[i] for i in graph.nodes}
        randomised_graph = nx.relabel_nodes(graph, relabeling)

        centroid = tree.centroid(randomised_graph)  # type: int
        self.assertEqual(relabeling[0], centroid)

    def test_large_regular_tree(self) -> None:
        """Test finding the centroid on a large balanced 3-ary tree with random node ids."""
        graph = nx.balanced_tree(3, 10)
        # Relabel the nodes to a new values to hide the root
        randomised = random.permutation([i for i in graph.nodes])
        relabeling = {i: randomised[i] for i in graph.nodes}
        randomised_graph = nx.relabel_nodes(graph, relabeling)

        centroid = tree.centroid(randomised_graph)  # type: int
        self.assertEqual(relabeling[0], centroid)

    @staticmethod
    def permutation_to_pebbles(permutation: Permutation[_V]) -> Dict[_V, tree.Pebble[_V]]:
        """Given a permutation produce a dict mapping the vertices in the permutation to Pebbles."""
        return {i: tree.Pebble(permutation[i]) for i in permutation}
