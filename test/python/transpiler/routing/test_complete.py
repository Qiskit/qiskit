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

"""Test cases for the permutation.complete package"""

from typing import List

from numpy import random

from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing import util, Swap
from qiskit.transpiler.routing.complete import permute, partial_permute
from .util import valid_parallel_swaps


class TestComplete(QiskitTestCase):
    """The test cases"""

    def test_permute_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out = list(permute({}))  # type: List[List[Swap]]

        valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_permute_complete_4(self) -> None:
        """Test a permutation of 4 elements
        
            permutation = (0)(321)
        """
        permutation = {0: 0, 1: 3, 2: 1, 3: 2}

        out = list(permute(permutation))
        valid_parallel_swaps(self, out)
        self.assertEqual(len(out), 2)
        identity_dict = {i: i for i in permutation.values()}
        util.swap_permutation(out, permutation)
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_5(self) -> None:
        """Test a permutation of 5 elements
        
            permutation = (0)(2)(431)
        """
        permutation = {0: 0, 1: 4, 2: 2, 3: 1, 4: 3}

        out = list(permute(permutation))
        valid_parallel_swaps(self, out)
        self.assertEqual(len(out), 2)
        identity_dict = {i: i for i in permutation.values()}
        util.swap_permutation(out, permutation)
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_5_2(self) -> None:
        """Test a permutation of one transposition in 5 elements.
        
            permutation = (02)
        """
        permutation = {0: 2, 1: 1, 2: 0, 3: 3, 4: 4}

        out = list(permute(permutation))
        valid_parallel_swaps(self, out)
        self.assertEqual(len(out), 1)
        identity_dict = {i: i for i in permutation.values()}
        util.swap_permutation(out, permutation)
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_4_2(self) -> None:
        """Test a permutation with non-contiguous elements.
        
            permutation = (3960)
        """
        permutation = {3: 9, 9: 6, 6: 0, 0: 3}

        out = list(permute(permutation))
        valid_parallel_swaps(self, out)
        self.assertEqual(len(out), 2)
        identity_dict = {i: i for i in permutation.values()}
        util.swap_permutation(out, permutation)
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_indexing(self) -> None:
        """Test a permutation with non-contiguous indexing: (246)"""
        permutation = {2: 4, 4: 6, 6: 2}
        out = list(permute(permutation))
        self.assertEqual(len(out), 2)
        valid_parallel_swaps(self, out)
        util.swap_permutation(out, permutation)
        identity_dict = {k: k for k in permutation}
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_big(self) -> None:
        """Test whether the swaps for a randomly generated large permutation implement it."""
        items = range(10 ** 5)
        rand_permutation = list(random.permutation(items))
        permutation = {i: rand_permutation[i] for i in items}

        out = list(permute(permutation))
        valid_parallel_swaps(self, out)
        util.swap_permutation(out, permutation)
        identity_dict = {i: i for i in items}
        self.assertEqual(identity_dict, permutation)

    def test_partial_permute_complete_big(self) -> None:
        """Test whether the swaps for a randomly generated large permutation implement it."""
        items = 10 ** 5
        rand_permutation = list(random.permutation(range(items)))
        pairs = [(i, rand_permutation[i]) for i in range(items)]
        random.shuffle(pairs)
        mapping = {k: v for k, v in pairs[0:items // 4]}

        out = list(partial_permute(mapping, list(range(items))))
        valid_parallel_swaps(self, out)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)
