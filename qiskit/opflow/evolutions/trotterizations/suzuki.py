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

"""Suzuki Class"""

from typing import List, Union, cast

from numpy import isreal

from qiskit.circuit import ParameterExpression
from qiskit.opflow.evolutions.trotterizations.trotterization_base import TrotterizationBase
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.utils.deprecation import deprecate_func


class Suzuki(TrotterizationBase):
    r"""
    Deprecated: Suzuki Trotter expansion, composing the evolution circuits of each Operator in the sum
    together by a recursive "bookends" strategy, repeating the whole composed circuit
    ``reps`` times.

    Detailed in https://arxiv.org/pdf/quant-ph/0508139.pdf.
    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, reps: int = 1, order: int = 2) -> None:
        """
        Args:
            reps: The number of times to repeat the expansion circuit.
            order: The order of the expansion to perform.

        """
        super().__init__(reps=reps)
        self._order = order

    @property
    def order(self) -> int:
        """returns order"""
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        """sets order"""
        self._order = order

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if not isinstance(operator, (SummedOp, PauliSumOp)):
            raise TypeError("Trotterization converters can only convert SummedOp or PauliSumOp.")

        if isinstance(operator.coeff, (float, ParameterExpression)):
            coeff = operator.coeff
        else:
            if isreal(operator.coeff):
                coeff = operator.coeff.real
            else:
                raise TypeError(
                    "Coefficient of the operator must be float or ParameterExpression, "
                    f"but {operator.coeff}:{type(operator.coeff)} is given."
                )

        if isinstance(operator, PauliSumOp):
            comp_list = self._recursive_expansion(operator, coeff, self.order, self.reps)
        if isinstance(operator, SummedOp):
            comp_list = Suzuki._recursive_expansion(operator.oplist, coeff, self.order, self.reps)

        single_rep = ComposedOp(cast(List[OperatorBase], comp_list))
        full_evo = single_rep.power(self.reps)
        return full_evo.reduce()

    @staticmethod
    def _recursive_expansion(
        op_list: Union[List[OperatorBase], PauliSumOp],
        evo_time: Union[float, ParameterExpression],
        expansion_order: int,
        reps: int,
    ) -> List[PrimitiveOp]:
        """
        Compute the list of pauli terms for a single slice of the Suzuki expansion
        following the paper https://arxiv.org/pdf/quant-ph/0508139.pdf.

        Args:
            op_list: The slice's weighted Pauli list for the Suzuki expansion
            evo_time: The parameter lambda as defined in said paper,
                adjusted for the evolution time and the number of time slices
            expansion_order: The order for the Suzuki expansion.
            reps: The number of times to repeat the expansion circuit.

        Returns:
            The evolution list after expansion.
        """
        if expansion_order == 1:
            # Base first-order Trotter case
            return [(op * (evo_time / reps)).exp_i() for op in op_list]  # type: ignore
        if expansion_order == 2:
            half = Suzuki._recursive_expansion(op_list, evo_time / 2, expansion_order - 1, reps)
            return list(reversed(half)) + half
        else:
            p_k = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
            side = 2 * Suzuki._recursive_expansion(
                op_list, evo_time * p_k, expansion_order - 2, reps
            )
            middle = Suzuki._recursive_expansion(
                op_list, evo_time * (1 - 4 * p_k), expansion_order - 2, reps
            )
            return side + middle + side
