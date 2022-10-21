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

"""Tests for BasePrimitiveResult."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from ddt import data, ddt, unpack

from qiskit.primitives.base.base_result import BasePrimitiveResult
from qiskit.test import QiskitTestCase


################################################################################
## STUB DATACLASS
################################################################################
@dataclass
class Result(BasePrimitiveResult):
    """Dummy result dataclass implementing BasePrimitiveResult."""

    field_1: Collection[Any]
    field_2: Collection[Any]


################################################################################
## TESTS
################################################################################
@ddt
class TestBasePrimitiveResult(QiskitTestCase):
    """Tests BasePrimitiveResult."""

    @data(0, 1.2, True, "sequence", b"sequence", {"name": "value"})
    def test_post_init_type_error(self, field_1):
        """Tests post init type error."""
        self.assertRaises(TypeError, Result, *(field_1, []))

    @data(([1], []), ([], [1]), ([1, 2], []), ([1], [1, 2]))
    @unpack
    def test_post_init_value_error(self, field_1, field_2):
        """Tests post init value error."""
        self.assertRaises(ValueError, Result, *(field_1, field_2))

    @data(0, 1, 2, 3)
    def test_num_experiments(self, num_experiments):
        """Tests {num_experiments} num_experiments."""
        result = Result([0] * num_experiments, [1] * num_experiments)
        self.assertEqual(num_experiments, result.num_experiments)

    @data(0, 1, 2, 3)
    def test_experiments(self, num_experiments):
        """Test experiment data."""
        field_1 = list(range(num_experiments))
        field_2 = [i + 1 for i in range(num_experiments)]
        experiments = Result(field_1, field_2).experiments
        self.assertIsInstance(experiments, tuple)
        for i, exp in enumerate(experiments):
            self.assertEqual(exp, {"field_1": i, "field_2": i + 1})

    @data(0, 1, 2, 3)
    def test_decompose(self, num_experiments):
        """Test decompose."""
        field_1 = list(range(num_experiments))
        field_2 = [i + 1 for i in range(num_experiments)]
        result = Result(field_1, field_2)
        for i, res in enumerate(result.decompose()):
            self.assertIsInstance(res, Result)
            f1, f2 = (i,), (i + 1,)
            self.assertEqual(res, Result(f1, f2))

    def test_field_names(self):
        """Tests field names ("field_1", "field_2")."""
        result = Result([], [])
        self.assertEqual(result._field_names, ("field_1", "field_2"))

    @data(([], []), ([0], [0]), ([0], [1]))
    @unpack
    def test_field_values(self, field_1, field_2):
        """Tests field values ({field_1}, {field_2})."""
        result = Result(field_1, field_2)
        self.assertEqual(result._field_values, (field_1, field_2))
