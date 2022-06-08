# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for a Real Variational Principle."""

from abc import ABC
from typing import Union

from qiskit.opflow import (
    CircuitQFI,
)
from .variational_principle import (
    VariationalPrinciple,
)


class RealVariationalPrinciple(VariationalPrinciple, ABC):
    """Class for a Real Variational Principle. The real variant means that we consider real time
    dynamics."""

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ) -> None:
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either ``'lin_comb_full'`` or
                ``'overlap_block_diag'`` or ``'overlap_diag'`` or ``CircuitQFI``.
        """
        super().__init__(
            qfi_method,
            self._grad_method,
        )
