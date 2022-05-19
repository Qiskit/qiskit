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

from pytest import mark, raises

from qiskit.primitives import BaseResult


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
class TestBaseResult:
    """Tests BaseResult."""

    @mark.parametrize(
        "foo, bar",
        cases := [([1], []), ([], [1]), ([1, 2], []), ([1], [1, 2])],
        ids=[f"{len(foo)}-{len(bar)}" for foo, bar in cases],
    )
    def test_post_init(self, foo, bar):
        with raises(ValueError):
            assert Result(foo, bar)

    @mark.parametrize(
        "num_experiments",
        cases := [0, 1, 2, 3],
        ids=[f"{num_experiments}" for num_experiments in cases],
    )
    def test_num_experiments(self, num_experiments):
        result = Result([0] * num_experiments, [1] * num_experiments)
        assert result.num_experiments == num_experiments

    @mark.parametrize(
        "foo, bar",
        cases := [([], []), ([0], [0]), ([0], [1])],
        ids=["empty", "equal", "distinct"],
    )
    def test_field_values(self, foo, bar):
        result = Result(foo, bar)
        assert result._field_values == [foo, bar]
