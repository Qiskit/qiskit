"""Test cases for the permutation.complete package"""
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

from typing import List
from unittest import TestCase

from numpy import random

from qiskit.transpiler.routing import util, Swap
from qiskit.transpiler.routing.complete import permute, partial_permute
from test.python.transpiler.routing.test_util import TestUtil  # pylint: disable=wrong-import-order


class TestComplete(TestCase):
    """The test cases"""

    def test_permute_empty(self) -> None:
        """Test if an empty permutation is not permuted."""
        out: List[List[Swap]] = list(permute({}))

        TestUtil.valid_parallel_swaps(self, out)
        self.assertListEqual([], out)

    def test_permute_complete_4(self) -> None:
        """Test a permutation of 4 elements

            permutation = (0)(321)
        """
        permutation = {0: 0, 1: 3, 2: 1, 3: 2}

        out = list(permute(permutation))
        TestUtil.valid_parallel_swaps(self, out)
        self.assertCountEqual([[(1, 3)], [(2, 1)]], out)

    def test_permute_complete_5(self) -> None:
        """Test a permutation of 5 elements

            permutation = (0)(2)(431)
        """
        permutation = {0: 0, 1: 4, 2: 2, 3: 1, 4: 3}

        out = list(permute(permutation))
        TestUtil.valid_parallel_swaps(self, out)
        self.assertEqual([[(1, 4)], [(3, 1)]], out)

    def test_permute_complete_5_2(self) -> None:
        """Test a permutation of one transposition in 5 elements.

            permutation = (02)
        """
        permutation = {0: 2, 1: 1, 2: 0, 3: 3, 4: 4}

        out = list(permute(permutation))
        TestUtil.valid_parallel_swaps(self, out)
        self.assertEqual([[(0, 2)]], out)

    def test_permute_complete_4_2(self) -> None:
        """Test a permutation with non-contiguous elements.

            permutation = (3960)
        """
        permutation = {3: 9, 9: 6, 6: 0, 0: 3}

        out = list(permute(permutation))
        TestUtil.valid_parallel_swaps(self, out)
        self.assertEqual([[(6, 0), (9, 3)], [(6, 3)]], out)

    def test_permute_complete_indexing(self) -> None:
        """Test a permutation with non-contiguous indexing: (246)"""
        permutation = {2: 4, 4: 6, 6: 2}
        out = list(permute(permutation))
        self.assertEqual(len(out), 2)
        TestUtil.valid_parallel_swaps(self, out)
        util.swap_permutation(out, permutation)
        identity_dict = {k: k for k in permutation}
        self.assertEqual(identity_dict, permutation)

    def test_permute_complete_big(self) -> None:
        """Test whether the swaps for a randomly generated large permutation implement it."""
        items = range(10 ** 5)
        rand_permutation = list(random.permutation(items))
        permutation = {i: rand_permutation[i] for i in items}

        out = list(permute(permutation))
        TestUtil.valid_parallel_swaps(self, out)
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
        TestUtil.valid_parallel_swaps(self, out)
        util.swap_permutation(out, mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping}, mapping)
