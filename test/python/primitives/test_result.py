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
    foo: Collection[Any]
    bar: Collection[Any]


################################################################################
## TESTS
################################################################################
@ddt
class TestBaseResult(QiskitTestCase):
    """Tests BaseResult."""

    @data(([1], []), ([], [1]), ([1, 2], []), ([1], [1, 2]))
    @unpack
    def test_post_init(self, foo, bar):
        """Tests post init({foo}, {bar})."""
        self.assertRaises(ValueError, Result, *(foo, bar))

    @data(0, 1, 2, 3)
    def test_num_experiments(self, num_experiments):
        """Tests {num_experiments} num_experiments."""
        result = Result([0] * num_experiments, [1] * num_experiments)
        self.assertEqual(num_experiments, result.num_experiments)

    @data(0, 1, 2, 3)
    def test_experiments(self, num_experiments):
        """Test experiments."""
        foo = [i for i in range(num_experiments)]
        bar = [i + 1 for i in range(num_experiments)]
        experiments = Result(foo, bar).experiments
        self.assertIsInstance(experiments, tuple)
        for i, exp in enumerate(experiments):
            self.assertEqual(exp, (i, i + 1))

    def test_field_names(self):
        result = Result([], [])
        self.assertEqual(result._field_names, ("foo", "bar"))

    @data(([], []), ([0], [0]), ([0], [1]))
    @unpack
    def test_field_values(self, foo, bar):
        """Tests field values ({foo}, {bar})."""
        result = Result(foo, bar)
        self.assertEqual(result._field_values, (foo, bar))
