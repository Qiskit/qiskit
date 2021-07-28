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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Test marginal_counts function."""

import unittest

from qiskit.result.utils import marginal_counts


class TestMarginalCounts(unittest.TestCase):
    def test_default(self):
        raw_counts = {"00":11, "11": 12}
        expected = raw_counts.copy()
        result = marginal_counts(raw_counts)
        self.assertEqual(expected, result)

    def test_1qubit(self):
        raw_counts = {"00":11, "11": 12}
        expected = {"0": 11, "1": 12}
        result = marginal_counts(raw_counts, [0])
        self.assertEqual(expected, result)

    def test_1qubit_sum(self):
        raw_counts = {"01":11, "11": 12}
        expected = {"1": 23}
        result = marginal_counts(raw_counts, [0])
        self.assertEqual(expected, result)
