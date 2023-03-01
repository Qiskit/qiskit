# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Entangler Map"""

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.utils import get_entangler_map, validate_entangler_map


class TestEntanglerMap(QiskitAlgorithmsTestCase):
    """Test Entangler Map"""

    def test_map_type_linear(self):
        """,ap type linear test"""
        ref_map = [[0, 1], [1, 2], [2, 3]]
        entangler_map = get_entangler_map("linear", 4)

        for (ref_src, ref_targ), (exp_src, exp_targ) in zip(ref_map, entangler_map):
            self.assertEqual(ref_src, exp_src)
            self.assertEqual(ref_targ, exp_targ)

    def test_map_type_full(self):
        """map type full test"""
        ref_map = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        entangler_map = get_entangler_map("full", 4)

        for (ref_src, ref_targ), (exp_src, exp_targ) in zip(ref_map, entangler_map):
            self.assertEqual(ref_src, exp_src)
            self.assertEqual(ref_targ, exp_targ)

    def test_validate_entangler_map(self):
        """validate entangler map test"""
        valid_map = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        self.assertTrue(validate_entangler_map(valid_map, 4))

        valid_map_2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 2]]
        self.assertTrue(validate_entangler_map(valid_map_2, 4, True))

        invalid_map = [[0, 4], [4, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        with self.assertRaises(ValueError):
            validate_entangler_map(invalid_map, 4)

        invalid_map_2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 2]]
        with self.assertRaises(ValueError):
            validate_entangler_map(invalid_map_2, 4)

        wrong_type_map = {0: [1, 2, 3], 1: [2, 3]}
        with self.assertRaises(TypeError):
            validate_entangler_map(wrong_type_map, 4)

        wrong_type_map_2 = [(0, 1), (0, 2), (0, 3)]
        with self.assertRaises(TypeError):
            validate_entangler_map(wrong_type_map_2, 4)


if __name__ == "__main__":
    unittest.main()
