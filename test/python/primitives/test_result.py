# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BaseResult."""

from dataclasses import dataclass
from typing import Any, Collection

from ddt import data, ddt, unpack

from qiskit.primitives import BaseResult
from qiskit.test import QiskitTestCase


################################################################################
## STUB DATACLASS
################################################################################
@dataclass
class Result(BaseResult):
    """Dummy result dataclass implementing BaseResult."""
    field_A: Collection[Any]
    field_B: Collection[Any]


################################################################################
## TESTS
################################################################################
@ddt
class TestBaseResult(QiskitTestCase):
    """Tests BaseResult."""

    @data(([1], []), ([], [1]), ([1, 2], []), ([1], [1, 2]))
    @unpack
    def test_post_init(self, field_A, field_B):
        """Tests post init({field_A}, {field_B})."""
        self.assertRaises(ValueError, Result, *(field_A, field_B))

    @data(0, 1, 2, 3)
    def test_num_experiments(self, num_experiments):
        """Tests {num_experiments} num_experiments."""
        result = Result([0] * num_experiments, [1] * num_experiments)
        self.assertEqual(num_experiments, result.num_experiments)

    @data(0, 1, 2, 3)
    def test_experiments(self, num_experiments):
        """Test experiments."""
        field_A = list(range(num_experiments))
        field_B = [i + 1 for i in range(num_experiments)]
        experiments = Result(field_A, field_B).experiments
        self.assertIsInstance(experiments, tuple)
        for i, exp in enumerate(experiments):
            self.assertEqual(exp, (i, i + 1))

    def test_field_names(self):
        """Tests field names ("field_A", "field_B")."""
        result = Result([], [])
        self.assertEqual(result._field_names, ("field_A", "field_B"))

    @data(([], []), ([0], [0]), ([0], [1]))
    @unpack
    def test_field_values(self, field_A, field_B):
        """Tests field values ({field_A}, {field_B})."""
        result = Result(field_A, field_B)
        self.assertEqual(result._field_values, (field_A, field_B))
