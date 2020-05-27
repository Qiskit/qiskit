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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Test Counts class."""

import unittest

from qiskit.result import counts
from qiskit.result import exceptions
from qiskit.result import utils


class TestCounts(unittest.TestCase):

    def test_just_counts(self):
        raw_counts = {'0x0': 21, '0x2': 12}
        expected = {'0': 21, '10': 12}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_counts_with_exta_formatting_data(self):
        raw_counts = {'0x0': 4, '0x2': 10}
        expected = {'0 0 00': 4, '0 0 10': 10}
        result = counts.Counts(raw_counts, 'test_counts', shots=14,
                               creg_sizes=[['c0', 2], ['c0', 1], ['c1', 1]],
                               memory_slots=4)
        self.assertEqual(result, expected)

    def test_marginal_counts(self):
        raw_counts = {'0x0': 4, '0x1': 7, '0x2': 10, '0x6': 5, '0x9': 11,
                      '0xD': 9, '0xE': 8}
        expected = {'00': 4, '01': 27, '10': 23}
        counts_obj = counts.Counts(raw_counts, shots=54, creg_sizes=[['c0', 4]],
                                   memory_slots=4)
        result = utils.marginal_counts(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_int_outcomes(self):
        raw_counts = {'0x0': 21, '0x2': 12, '0x3': 5, '0x2E': 265}
        expected = {0: 21, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(expected, result)

    def test_most_frequent(self):
        raw_counts = {'0x0': 21, '0x2': 12, '0x3': 5, '0x2E': 265}
        expected = '101110'
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.most_frequent()
        self.assertEqual(expected, result)

    def test_most_frequent_duplicate(self):
        raw_counts = {'0x0': 265, '0x2': 12, '0x3': 5, '0x2E': 265}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.NoMostFrequentCount,
                          counts_obj.most_frequent)
