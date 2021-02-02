# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
QDrift Class

"""

import numpy as np

from qiskit.utils import algorithm_globals

from ...list_ops.composed_op import ComposedOp
from ...list_ops.summed_op import SummedOp
from ...operator_base import OperatorBase
from ...primitive_ops.pauli_sum_op import PauliSumOp
from .trotterization_base import TrotterizationBase

# pylint: disable=invalid-name


class QDrift(TrotterizationBase):
    """The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in https://arxiv.org/abs/1811.08017.
    """

    def __init__(self, reps: int = 1) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
        """
        super().__init__(reps=reps)

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if not isinstance(operator, (SummedOp, PauliSumOp)):
            raise TypeError("Trotterization converters can only convert SummedOp or PauliSumOp.")

        if isinstance(operator, PauliSumOp):
            operator_iter = operator
            coeffs = operator.coeffs
        else:
            operator_iter = operator.oplist
            coeffs = [op.coeff for op in operator_iter]

        # We artificially make the weights positive, TODO check approximation performance
        weights = np.abs(coeffs)
        lambd = np.sum(weights)

        N = 2 * (lambd ** 2) * (operator.coeff ** 2)
        factor = lambd * operator.coeff / (N * self.reps)
        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant factor.
        scaled_ops = [(op * (factor / op.coeff)).exp_i() for op in operator_iter]
        sampled_ops = algorithm_globals.random.choice(
            scaled_ops, size=(int(N * self.reps),), p=weights / lambd
        )

        return ComposedOp(sampled_ops).reduce()
