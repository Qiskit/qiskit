"""Test cases for the permutation.path package"""
#  arct performs circuit transformations of quantum circuit for architectures
#  Copyright (C) 2019  Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unittest import TestCase
import random
from test.python.transpiler.routing.test_util import TestUtil
import networkx as nx
import numpy as np
from qiskit.transpiler.routing import util
from qiskit.transpiler.routing.path import permute_path, permute_path_partial


class TestPath(TestCase):
    """The test cases"""

    def test_path_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out = list(permute_path({}))

        TestUtil.valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_path_one_single(self) -> None:
        """Test a permutation of 1 element"""
        out = list(permute_path({0: 1, 1: 0}))

        TestUtil.valid_parallel_swaps(self, out)
        self.assertListEqual([[(1, 0)]], out)

    def test_path_two_single(self) -> None:
        """Test a permutation of 2 elements sequentially"""
        out = list(permute_path({0: 1, 1: 0, 2: 3, 3: 2}))

        TestUtil.valid_parallel_swaps(self, out)
        valid_graph = nx.Graph()
        valid_graph.add_edges_from([(1, 0), (3, 2), (2, 1)])
        TestUtil.valid_edge_swaps(self, out, valid_graph)
        self.assertListEqual([[(1, 0), (3, 2)]], out)

    def test_path_two_parallel(self) -> None:
        """Test a permutation of 2 elements parallel"""
        out = list(permute_path({0: 3, 1: 2, 2: 1, 3: 0}))

        TestUtil.valid_parallel_swaps(self, out)
        valid_graph = nx.Graph()
        valid_graph.add_edges_from([(1, 0), (3, 2), (2, 1)])
        TestUtil.valid_edge_swaps(self, out, valid_graph)
        self.assertListEqual([[(2, 1)], [(1, 0), (3, 2)], [(2, 1)], [(1, 0), (3, 2)]], out)

    def test_path_arbitrary(self) -> None:
        """Test a permutation of arbitrary permutation"""
        out = list(permute_path({0: 3, 1: 0, 2: 4, 3: 2, 4: 1}))

        TestUtil.valid_parallel_swaps(self, out)
        self.assertListEqual([[(4, 3)], [(1, 0), (3, 2)], [(2, 1), (4, 3)], [(3, 2)]], out)

    def test_path_random(self) -> None:
        """Test a random permutation"""
        size = 10 ** 3
        rand_permutation = list(np.random.permutation(range(size)))
        permutation = dict(enumerate(rand_permutation))

        out = list(permute_path(permutation))
        self.assertGreaterEqual(size, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, permutation)
        identity_dict = {i: i for i in range(size)}
        self.assertEqual(identity_dict, permutation)

    def test_partial_path_single(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {15: 3}

        out = list(permute_path_partial(mapping))
        self.assertEqual(12, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_small(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {0: 2, 2:0}

        out = list(permute_path_partial(mapping))
        self.assertGreater(4, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_small2(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {1:2, 0:4}

        out = list(permute_path_partial(mapping))
        self.assertGreaterEqual(5, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_medium(self) -> None:
        """Test the greedy path algorithm for medium size"""
        mapping = {4: 2, 0: 1, 5: 0}

        out = list(permute_path_partial(mapping))
        # self.assertGreaterEqual(7, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_random(self) -> None:
        """Test the partial path algorithm with a random partial mapping"""
        length = 10**3
        mapped = length // 2
        destinations = list(np.random.permutation(length))
        partial_permutation = dict(random.sample(list(enumerate(destinations)), mapped))

        out = list(permute_path_partial(partial_permutation, length))
        self.assertGreaterEqual(length, len(out))
        TestUtil.valid_parallel_swaps(self, out)

        util.swap_permutation(out, partial_permutation, allow_missing_keys=True)
        identity_dict = {i: i for i in partial_permutation}
        self.assertEqual(identity_dict, partial_permutation)
