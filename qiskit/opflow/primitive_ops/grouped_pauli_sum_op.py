# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" GroupedPauliSumOp Class """

from typing import Union

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from .pauli_sum_op import PauliSumOp


class GroupedPauliSumOp(PauliSumOp):
    """Class for PauliSumOp after grouping"""

    def __init__(
            self,
            primitive: SparsePauliOp,
            coeff: Union[int, float, complex, ParameterExpression] = 1.0,
            grouping_type: str = "TPB",
    ) -> None:
        """
        Args:
            primitive: The SparsePauliOp which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive.
            grouping_type: The type of grouping (default value TPB stands for Tensor Product Basis)

        Raises:
            TypeError: invalid parameters.
        """
        super().__init__(primitive, coeff)
        self._grouping_type = grouping_type

    @property
    def grouping_type(self) -> str:
        """
        Returns: Type of Grouping (tpb)
        """
        return self._grouping_type
