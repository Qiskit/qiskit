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

"""Test cases for the permutation.path package"""

import random
import unittest

import itertools
import networkx as nx
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing import util
from qiskit.transpiler.routing.fast_path import permute_path, permute_path_partial
from .util import valid_parallel_swaps, valid_edge_swaps


class TestFastPath(QiskitTestCase):
    """The test cases"""

    def test_path_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out = list(permute_path({}))

        valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_path_one_single(self) -> None:
        """Test a permutation of 1 element"""
        out = list(permute_path({0: 1, 1: 0}))

        valid_parallel_swaps(self, out)
        self.assertListEqual([[(0, 1)]], out)

    def test_path_two_single(self) -> None:
        """Test a permutation of 2 elements sequentially"""
        out = list(permute_path({0: 1, 1: 0, 2: 3, 3: 2}))

        valid_parallel_swaps(self, out)
        valid_graph = nx.Graph()
        valid_graph.add_edges_from([(1, 0), (3, 2), (2, 1)])
        valid_edge_swaps(self, out, valid_graph)
        self.assertListEqual([[(0, 1), (2, 3)]], out)

    def test_path_two_parallel(self) -> None:
        """Test a permutation of 2 elements parallel"""
        out = list(permute_path({0: 3, 1: 2, 2: 1, 3: 0}))

        valid_parallel_swaps(self, out)
        valid_graph = nx.Graph()
        valid_graph.add_edges_from([(1, 0), (3, 2), (2, 1)])
        valid_edge_swaps(self, out, valid_graph)
        self.assertListEqual([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], out)

    def test_path_arbitrary(self) -> None:
        """Test a permutation of arbitrary permutation"""
        out = list(permute_path({0: 3, 1: 0, 2: 4, 3: 2, 4: 1}))

        valid_parallel_swaps(self, out)
        self.assertListEqual([[(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2)]], out)

    def test_path_random(self) -> None:
        """Test a random permutation"""
        size = 10 ** 3
        rand_permutation = list(np.random.permutation(range(size)))
        permutation = dict(enumerate(rand_permutation))

        out = list(permute_path(permutation))
        self.assertGreaterEqual(size, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, permutation)
        identity_dict = {i: i for i in range(size)}
        self.assertEqual(identity_dict, permutation)

    def test_partial_path_single(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {15: 3}

        out = list(permute_path_partial(mapping))
        self.assertEqual(12, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_small(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {0: 2, 2: 0}

        out = list(permute_path_partial(mapping))
        self.assertGreater(4, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_small2(self) -> None:
        """Test the greedy path algorithm with a random partial mapping"""
        mapping = {1: 2, 0: 4}

        out = list(permute_path_partial(mapping))
        self.assertGreaterEqual(5, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_medium(self) -> None:
        """Test the greedy path algorithm for medium size"""
        mapping = {4: 2, 0: 1, 5: 0}

        out = list(permute_path_partial(mapping))
        # self.assertGreaterEqual(7, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, mapping, allow_missing_keys=True)
        identity_dict = {i: i for i in mapping}
        self.assertEqual(identity_dict, mapping)

    def test_partial_path_random(self) -> None:
        """Test the partial path algorithm with a random partial mapping"""
        length = 10 ** 3
        mapped = length // 2
        destinations = list(np.random.permutation(length))
        partial_permutation = dict(random.sample(list(enumerate(destinations)), mapped))

        out = list(permute_path_partial(partial_permutation, length))
        self.assertGreaterEqual(length, len(out))
        valid_parallel_swaps(self, out)

        util.swap_permutation(out, partial_permutation, allow_missing_keys=True)
        identity_dict = {i: i for i in partial_permutation}
        self.assertEqual(identity_dict, partial_permutation)

    @unittest.skip("Not a test")
    def test_faster_completions(self) -> None:
        """Test if the completion found by the partial permuter is (close to) the best."""
        path_length = 13
        samples = 10 ** 4

        nodes = list(range(path_length))
        for _ in range(samples):
            partial_permutation = util.random_partial_permutation(nodes)
            print(partial_permutation)
            partial_swaps = permute_path_partial(partial_permutation, length=path_length)
            # Iterate over all possible completions
            completion_origins = list(set(nodes) - set(partial_permutation.keys()))
            all_completion_destinations = itertools.permutations(
                set(nodes) - set(partial_permutation.values()))
            for completion_destinations in all_completion_destinations:
                # Find the extra entries of the completion
                completion = dict(zip(completion_origins, completion_destinations))
                completion.update(partial_permutation)  # Construct the permutation
                complete_swaps = permute_path(completion)

                self.assertLessEqual(len(list(partial_swaps)) - 1, len(list(complete_swaps)),
                                     "The completion ({}) was faster than"
                                     "the partial permutation ({}).".format(completion,
                                                                            partial_permutation))
