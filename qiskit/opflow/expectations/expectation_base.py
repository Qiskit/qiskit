# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ExpectationBase Class"""

from abc import abstractmethod
from typing import Union

import numpy as np

from qiskit.opflow.converters import ConverterBase
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func


class ExpectationBase(ConverterBase):
    r"""
    Deprecated: A base for Expectation value converters. Expectations are converters which enable the
    computation of the expectation value of an Observable with respect to some state function.
    They traverse an Operator tree, replacing OperatorStateFn measurements with equivalent
    measurements which are more amenable to computation on quantum or classical hardware. For
    example, if one would like to measure the expectation value of an Operator ``o`` expressed
    as a sum of Paulis with respect to some state function, but only has access to diagonal
    measurements on Quantum hardware, we can create a measurement ~StateFn(o),
    use a ``PauliExpectation`` to convert it to a diagonal measurement and circuit
    pre-rotations to a append to the state, and sample this circuit on Quantum hardware with
    a CircuitSampler. All in all, this would be:
    ``my_sampler.convert(my_expect.convert(~StateFn(o)) @ my_state).eval()``.

    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """Accept an Operator and return a new Operator with the measurements replaced by
        alternate methods to compute the expectation value.

        Args:
            operator: The operator to convert.

        Returns:
            The converted operator.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_variance(self, exp_op: OperatorBase) -> Union[list, complex, np.ndarray]:
        """Compute the variance of the expectation estimator.

        Args:
            exp_op: The full expectation value Operator after sampling.

        Returns:
             The variances or lists thereof (if exp_op contains ListOps) of the expectation value
             estimation.
        """
        raise NotImplementedError
