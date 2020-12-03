# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" MatrixExpectation Class """

import logging
from typing import Union

from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..list_ops import ListOp, ComposedOp
from ..state_fns.operator_state_fn import OperatorStateFn

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class MatrixExpectation(ExpectationBase):
    """ An Expectation converter which converts Operator measurements to be matrix-based so they
    can be evaluated by matrix multiplication. """

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept an Operator and return a new Operator with the Pauli measurements replaced by
        Matrix based measurements.

        Args:
            operator: The operator to convert.

        Returns:
            The converted operator.
        """
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            return operator.to_matrix_op()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert)
        else:
            return operator

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        r"""
        Compute the variance of the expectation estimator. Because this expectation
        works by matrix multiplication, the estimation is exact and the variance is
        always 0, but we need to return those values in a way which matches the Operator's
        structure.

        Args:
            exp_op: The full expectation value Operator.

        Returns:
             The variances or lists thereof (if exp_op contains ListOps) of the expectation value
             estimation, equal to 0.
        """

        # Need to do this to mimic Op structure
        def sum_variance(operator):
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])
            else:
                return 0.0

        return sum_variance(exp_op)
