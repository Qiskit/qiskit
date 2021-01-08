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

""" TaperedPauliSumOp Class """

from typing import Optional, Union
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterExpression

from .pauli_sum_op import PauliSumOp
from ..converters.z2_symmetries import Z2Symmetries


class TaperedPauliSumOp(PauliSumOp):
    """Class for PauliSumOp after tapering"""

    def __init__(
            self,
            primitive: SparsePauliOp,
            coeff: Union[int, float, complex, ParameterExpression] = 1.0,
            z2_symmetries: Optional[Z2Symmetries] = None
    ) -> None:
        """
        Args:
            primitive: The SparsePauliOp which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive.
            z2_symmetries: Z2 symmetries which the Operator has.

        Raises:
            TypeError: invalid parameters.
        """
        super().__init__(primitive, coeff)
        self._z2_symmetries = z2_symmetries

    @property
    def z2_symmetries(self) -> Z2Symmetries:
        """
        Z2 symmetries which the Operator has.

        Returns:
            The Z2 Symmetries.
        """
        return self._z2_symmetries
