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

"""Test cases for permutation.modular package"""

import unittest
from typing import List

from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing import Swap, util
from qiskit.transpiler.routing.modular import permute, _distinct_permutation, _in_module
from .util import valid_parallel_swaps


class TestPermuteModular(QiskitTestCase):
    """The test cases"""

    def test_modular_small(self) -> None:
        """Test a permutation on a small modular graph with modulesize=2 and modules=2."""
        # Modules: (0,1),(2,3)
        permutation = {0: 2, 1: 1, 2: 3, 3: 0}

        out = permute(permutation, modulesize=2, modules=2)
        valid_parallel_swaps(self, out)
        util.swap_permutation(out, permutation)
        self.assertEqual({i: i for i in permutation}, permutation)

    # IDEA: Implement functionality to support arbitrary assignments of nodes to modules.
    # Maybe pass a function that maps a node to a module?
    @unittest.expectedFailure
    def test_modular_noncontiguous(self) -> None:
        """Test whether non-contiguous node-ids work for the modular permutation."""
        # Modules(2,4)(6,8)
        permutation = {2: 6, 4: 2, 6: 8, 8: 2}

        out = permute(permutation, modulesize=2, modules=2)
        valid_parallel_swaps(self, out)
        self.assertCountEqual([[(2, 3)], [(0, 2)]], out)

    def test_modular_small_intraswaps(self) -> None:
        """Test the intraswap phase of permuting the modular graph."""
        # Modules: (0,1),(2,3)
        permutation = {0: 0, 1: 3, 2: 2, 3: 1}

        out = permute(permutation, modulesize=2, modules=2)
        valid_parallel_swaps(self, out)
        self.assertEqual([[(1, 0), (3, 2)], [(0, 2)], [(0, 1), (2, 3)]], out)

    def test_modular_medium(self) -> None:
        """For a specific mapping test the modular permuter that is taking too long"""
        mapping = {0: 11, 1: 14, 2: 2, 3: 4, 4: 6, 5: 3, 6: 5, 7: 12, 8: 13,
                   9: 9, 10: 8, 11: 10, 12: 1, 13: 7, 14: 0}
        out = permute(mapping, modulesize=3, modules=5)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_modular_big(self) -> None:
        """Test a randomly generated permutation on a large modular graph."""
        modulesize = 100
        modules = 95
        permutation = util.random_partial_permutation(list(range(modules * modulesize)),
                                                      nr_elements=modules * modulesize)

        out = permute(permutation, modulesize=modulesize, modules=modules)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=modulesize)
        util.swap_permutation(out, permutation)

        self.assertEqual({i: i for i in range(modules * modulesize)}, permutation)

    def test_partial_modular_small1(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 8, 7: 3}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small2(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 3, 1: 8, 2: 0}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small3(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {3: 2, 0: 6, 8: 3, 6: 0, 7: 4, 5: 1, 4: 8}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small4(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 6, 1: 8}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small5(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {1: 6, 0: 3, 4: 2, 2: 0, 3: 7, 6: 5}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small6(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 6, 2: 3, 11: 2, 1: 7, 9: 8, 10: 4}
        out = permute(mapping, modulesize=3, modules=4)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small7(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 3, 1: 4, 2: 6, 6: 0, 7: 5, 8: 8}
        out = permute(mapping, modulesize=3, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small8(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {3: 9, 6: 7, 5: 6, 8: 8, 9: 4, 0: 5}
        out = permute(mapping, modulesize=3, modules=4)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small9(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {5: 5, 10: 11, 3: 4, 4: 6, 9: 9, 11: 8, 8: 10}
        out = permute(mapping, modulesize=3, modules=5)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small10(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {0: 0, 1: 2, 4: 4, 5: 3}
        out = permute(mapping, modulesize=2, modules=3)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=2)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small11(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {1: 1, 2: 0}
        out = permute(mapping, modulesize=2, modules=2)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=2)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_small12(self) -> None:
        """For a specific partial mapping test the modular permuter"""
        mapping = {5: 3, 0: 0, 1: 4, 4: 5}
        out = permute(mapping, modulesize=3, modules=2)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=3)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)

    def test_partial_modular_medium(self) -> None:
        """Test a randomly generated partial permutation on a large modular graph."""
        modulesize = 50
        modules = 50

        partial_perm = util.random_partial_permutation(list(range(modules * modulesize)))

        out = permute(partial_perm, modulesize=modulesize, modules=modules)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=modulesize)
        util.swap_permutation(out, partial_perm, allow_missing_keys=True)
        self.assertEqual({i: i for i in partial_perm}, partial_perm)

    def test_partial_modular_big(self) -> None:
        """Test a randomly generated partial permutation on a large modular graph."""
        modulesize = 100
        modules = 90
        partial_perm = util.random_partial_permutation(list(range(modules * modulesize)))

        out = permute(partial_perm, modulesize=modulesize, modules=modules)
        valid_parallel_swaps(self, out)
        self.valid_module_swaps(out, modulesize=modulesize)
        util.swap_permutation(out, partial_perm, allow_missing_keys=True)
        self.assertEqual({i: i for i in partial_perm}, partial_perm)

    def test_distinct_permutation_small(self) -> None:
        """Test the distinct permutation function for a small test case."""
        modules = modulesize = 2
        permutation = {0: 3, 3: 1}

        out = _distinct_permutation(permutation, set(), modulesize, modules)
        self.assertEqual({0, 3}, out)

    def test_distinct_permutation_unmapped(self) -> None:
        """Test the distinct permutation with unmapped nodes"""
        modules = modulesize = 2
        permutation = {0: 3, 3: 1}

        out = _distinct_permutation(permutation, set(), modulesize, modules)
        self.assertEqual({0, 3}, out)

    def valid_module_swaps(self, swaps: List[List[Swap]], modulesize: int) -> None:
        """Check if the proscribed swaps can be performed on the modular graph."""
        for swap_step in swaps:
            for sw1, sw2 in swap_step:
                if _in_module(sw1, modulesize) != _in_module(sw2, modulesize):
                    # Can do inter-module routing
                    self.assertEqual(sw1 % modulesize, 0)
                    self.assertEqual(sw2 % modulesize, 0)
