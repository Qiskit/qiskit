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

""" Test Validation """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.utils.validation import (
    validate_in_set,
    validate_min,
    validate_min_exclusive,
    validate_max,
    validate_max_exclusive,
    validate_range,
    validate_range_exclusive,
    validate_range_exclusive_min,
    validate_range_exclusive_max,
)


class TestValidation(QiskitAlgorithmsTestCase):
    """Validation tests."""

    def test_validate_in_set(self):
        """validate in set test"""
        test_value = "value1"
        validate_in_set("test_value", test_value, {"value1", "value2"})
        with self.assertRaises(ValueError):
            validate_in_set("test_value", test_value, {"value3", "value4"})

    def test_validate_min(self):
        """validate min test"""
        test_value = 2.5
        validate_min("test_value", test_value, -1)
        validate_min("test_value", test_value, 2.5)
        with self.assertRaises(ValueError):
            validate_min("test_value", test_value, 4)
        validate_min_exclusive("test_value", test_value, -1)
        with self.assertRaises(ValueError):
            validate_min_exclusive("test_value", test_value, 2.5)
        with self.assertRaises(ValueError):
            validate_min_exclusive("test_value", test_value, 4)

    def test_validate_max(self):
        """validate max test"""
        test_value = 2.5
        with self.assertRaises(ValueError):
            validate_max("test_value", test_value, -1)
        validate_max("test_value", test_value, 2.5)
        validate_max("test_value", test_value, 4)
        with self.assertRaises(ValueError):
            validate_max_exclusive("test_value", test_value, -1)
        with self.assertRaises(ValueError):
            validate_max_exclusive("test_value", test_value, 2.5)
        validate_max_exclusive("test_value", test_value, 4)

    def test_validate_range(self):
        """validate range test"""
        test_value = 2.5
        with self.assertRaises(ValueError):
            validate_range("test_value", test_value, 0, 2)
        with self.assertRaises(ValueError):
            validate_range("test_value", test_value, 3, 4)
        validate_range("test_value", test_value, 2.5, 3)
        validate_range_exclusive("test_value", test_value, 0, 3)
        with self.assertRaises(ValueError):
            validate_range_exclusive("test_value", test_value, 0, 2.5)
            validate_range_exclusive("test_value", test_value, 2.5, 3)
        validate_range_exclusive_min("test_value", test_value, 0, 3)
        with self.assertRaises(ValueError):
            validate_range_exclusive_min("test_value", test_value, 2.5, 3)
        validate_range_exclusive_min("test_value", test_value, 0, 2.5)
        validate_range_exclusive_max("test_value", test_value, 2.5, 3)
        with self.assertRaises(ValueError):
            validate_range_exclusive_max("test_value", test_value, 0, 2.5)
        validate_range_exclusive_max("test_value", test_value, 2.5, 3)


if __name__ == "__main__":
    unittest.main()
