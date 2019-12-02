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

"""Test cases for the permutation.cartesian package"""

import random
from typing import List, TypeVar, Callable, Iterable, Mapping

import networkx as nx

from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing import util, Swap, Permutation, path, complete
from qiskit.transpiler.routing.cartesian import permute_grid, permute_cartesian_partial, \
    permute_grid_partial
from .util import valid_parallel_swaps, valid_edge_swaps

_V = TypeVar('_V')
PartialRouter = Callable[[Mapping[int, int]], Iterable[List[Swap[int]]]]


def square(permutation: Mapping[int, int]) -> List[List[Swap[int]]]:
    """used to as input function to make a cartesian product with a 2 by 2 grid"""
    return permute_grid_partial(permutation, 2, 2)


def construct_partial_complete(nodes: List[_V]
                               ) -> Callable[[Mapping[_V, _V]], Iterable[List[Swap[_V]]]]:
    """Set up a partial permuter on the complete graph."""
    return lambda p: complete.partial_permute(p, nodes)


class TestCartesian(QiskitTestCase):
    """The test cases"""

    def cartesian_generic_test(self, graph: nx.Graph, permutation: Permutation, height: int,
                               permute_x: PartialRouter, permute_y: PartialRouter) -> None:
        """Permutes and performs verifications for parallel, valid on graph, depth"""

        size = nx.number_of_nodes(graph)
        width = size // height

        out = list(permute_cartesian_partial(permutation, width, height, permute_x, permute_y))

        self.assertEqual(len(out) <= height + height + width, True)
        valid_parallel_swaps(self, out)
        valid_edge_swaps(self, out, graph)

        util.swap_permutation(out, permutation)
        identity_dict = {i: i for i in range(size)}
        self.assertEqual(identity_dict, permutation)

    def grid_generic_test(self, permutation: Permutation, height: int) -> None:
        """Permutes and performs verifications for parallel, valid on graph, depth"""

        size = len(permutation)
        width = size // height
        graph1 = nx.path_graph(width)
        graph2 = nx.path_graph(height)
        product_graph = nx.cartesian_product(graph2, graph1)
        product_graph = nx.relabel.convert_node_labels_to_integers(product_graph, ordering="sorted")

        self.cartesian_generic_test(product_graph, permutation, height,
                                    path.permute_path_partial, path.permute_path_partial)

    def test_grid_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out = list(permute_grid({}, 0))

        valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_grid_small1(self) -> None:
        """Test a permutation o a small grid"""
        out = list(permute_grid({0: 2, 1: 1, 2: 3, 3: 0}, 2))

        valid_parallel_swaps(self, out)
        self.assertListEqual([[(2, 3)], [(0, 2)]], out)

    def test_grid_small2(self) -> None:
        """Test a permutation of arbitrary permutation"""
        permutation = {0: 2, 1: 1, 2: 3, 3: 0}

        self.grid_generic_test(permutation, 2)

    def test_grid_arbitrary(self) -> None:
        """Test a permutation of arbitrary permutation"""
        permutation = {0: 5, 1: 2, 2: 10, 3: 6, 4: 7, 5: 4, 6: 0, 7: 1, 8: 3, 9: 8, 10: 9, 11: 11}

        self.grid_generic_test(permutation, 3)

    def test_debug_parellel_issue(self) -> None:
        """Used this for debugging a particular issue"""
        permutation = {0: 0, 1: 3, 2: 13, 3: 11, 4: 12, 5: 1, 6: 5, 7: 9, 8: 4,
                       9: 10, 10: 7, 11: 14, 12: 6, 13: 2, 14: 15, 15: 8}

        self.grid_generic_test(permutation, 4)

    def test_cartesian_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out = list(permute_cartesian_partial({}, 0, 0, path.permute_path_partial,
                                             path.permute_path_partial))

        valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_cartesian_path_path(self) -> None:
        """Test a permutation of a grid"""
        permutation = {0: 0, 1: 3, 2: 13, 3: 11, 4: 12, 5: 1, 6: 5, 7: 9,
                       8: 4, 9: 10, 10: 7, 11: 14, 12: 6, 13: 2, 14: 15, 15: 8}

        graph1 = nx.path_graph(4)
        graph2 = nx.path_graph(4)
        product_graph = nx.cartesian_product(graph2, graph1)
        product_graph = nx.relabel.convert_node_labels_to_integers(product_graph, ordering="sorted")

        self.cartesian_generic_test(product_graph, permutation, 4, path.permute_path_partial,
                                    path.permute_path_partial)

    def test_cartesian_grid_path(self) -> None:
        """Test a permutation of a grid-path cartesian product"""
        permutation = {0: 5, 1: 2, 2: 10, 3: 6, 4: 7, 5: 4, 6: 0, 7: 1, 8: 3, 9: 8, 10: 9, 11: 11}

        graph1 = nx.grid_2d_graph(2, 2)
        graph2 = nx.path_graph(3)
        prod_graph = nx.cartesian_product(graph2, graph1)
        prod_graph = nx.relabel.convert_node_labels_to_integers(prod_graph, ordering="sorted")

        self.cartesian_generic_test(prod_graph, permutation, 3, square, path.permute_path_partial)

    def test_cartesian_grid_grid(self) -> None:
        """Test a permutation of a grid-grid cartesian product"""
        permutation = {0: 0, 1: 3, 2: 13, 3: 11, 4: 12, 5: 1, 6: 5, 7: 9,
                       8: 4, 9: 10, 10: 7, 11: 14, 12: 6, 13: 2, 14: 15, 15: 8}

        graph1 = nx.grid_2d_graph(2, 2)
        graph2 = nx.grid_2d_graph(2, 2)
        prod_graph = nx.cartesian_product(graph2, graph1)
        prod_graph = nx.relabel.convert_node_labels_to_integers(prod_graph, ordering="sorted")

        self.cartesian_generic_test(prod_graph, permutation, 4, square, square)

    def test_cartesian_grid_complete(self) -> None:
        """Test a permutation of a grid-complete cartesian product"""
        permutation = {0: 9, 1: 3, 2: 15, 3: 4, 4: 1, 5: 8, 6: 12,
                       7: 14, 8: 13, 9: 6, 10: 2, 11: 10, 12: 5, 13: 11, 14: 0, 15: 7}

        graph1 = nx.grid_2d_graph(2, 2)
        graph2 = nx.complete_graph(4)
        prod_graph = nx.cartesian_product(graph2, graph1)
        prod_graph = nx.relabel.convert_node_labels_to_integers(prod_graph, ordering="sorted")

        self.cartesian_generic_test(prod_graph, permutation, 4, square,
                                    construct_partial_complete(graph2.nodes))

    # todo professional edition, right click on test, profile
    def test_grid_random(self) -> None:
        """Test a random permutation on random shape"""
        height = 50
        size = height ** 2
        items = list(range(size))
        random.shuffle(items)
        permutation = {i: items[i] for i in items}

        self.grid_generic_test(permutation, height)

    def test_k_n_path_2_random(self) -> None:
        """Test a random complete-path graph"""
        width = 800
        height = 2
        size = width * height
        items = list(range(size))
        random.shuffle(items)
        permutation = {i: items[i] for i in items}

        graph1 = nx.complete_graph(width)
        graph2 = nx.path_graph(height)
        prod_graph = nx.cartesian_product(graph2, graph1)
        prod_graph = nx.relabel.convert_node_labels_to_integers(prod_graph, ordering="sorted")

        self.cartesian_generic_test(prod_graph, permutation, height,
                                    construct_partial_complete(graph1.nodes),
                                    path.permute_path_partial)

    def test_k_n_path_5_random(self) -> None:
        """Test a random complete-path graph"""
        width = 800
        height = 5
        size = width * height
        items = list(range(size))
        random.shuffle(items)
        permutation = {i: items[i] for i in items}

        graph1 = nx.complete_graph(width)
        graph2 = nx.path_graph(height)
        prod_graph = nx.cartesian_product(graph2, graph1)
        prod_graph = nx.relabel.convert_node_labels_to_integers(prod_graph, ordering="sorted")

        self.cartesian_generic_test(prod_graph, permutation, height,
                                    construct_partial_complete(graph1.nodes),
                                    path.permute_path_partial)

    def test_grid_small1_partial(self) -> None:
        """Test a permutation on a small grid"""
        mapping = {0: 2, 2: 1}
        out = list(permute_cartesian_partial(mapping, 2, 2,
                                             path.permute_path_partial,
                                             path.permute_path_partial))

        self.assertEqual(2, len(out))
        valid_parallel_swaps(self, out)
        identity_dict = {i: i for i in mapping.values()}
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual(identity_dict, mapping)

    def test_grid_small2_partial(self) -> None:
        """Test a permutation on a small grid"""
        mapping = {2: 3, 3: 2}
        out = list(permute_cartesian_partial(mapping, 2, 3,
                                             path.permute_path_partial,
                                             path.permute_path_partial))

        valid_parallel_swaps(self, out)
        self.assertListEqual([[(3, 2)]], out)

    def test_grid_small3_partial(self) -> None:
        """Test a small permutation on grid."""
        mapping = {1: 0, 4: 4, 3: 5}
        out = list(permute_cartesian_partial(mapping, 3, 2,
                                             path.permute_path_partial,
                                             path.permute_path_partial))

        valid_parallel_swaps(self, out)

    def test_grid_perfect_matching(self) -> None:
        """Test a case where perfect matching was not found."""
        mapping = {3: 1, 5: 5, 1: 0, 2: 3}
        out = list(permute_cartesian_partial(mapping, 2, 4,
                                             path.permute_path_partial,
                                             path.permute_path_partial))

        valid_parallel_swaps(self, out)
        identity_dict = {i: i for i in mapping.values()}
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual(identity_dict, mapping)

    def test_grid_random_partial(self) -> None:
        """Test a large random partial mapping on a grid."""
        length = 10 ** 3
        width = length // 100
        mapped = length // 2

        height = length // width
        destinations = list(range(length))
        random.shuffle(destinations)
        partial_mapping = dict(random.sample(list(enumerate(destinations)), mapped))

        out = list(permute_cartesian_partial(partial_mapping, width, height,
                                             path.permute_path_partial,
                                             path.permute_path_partial))
        valid_parallel_swaps(self, out)
        identity_dict = {i: i for i in partial_mapping.values()}
        util.swap_permutation(out, partial_mapping, allow_missing_keys=True)
        self.assertEqual(identity_dict, partial_mapping)
