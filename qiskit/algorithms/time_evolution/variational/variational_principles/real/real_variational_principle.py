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

"""Class for a Real Variational Principle."""

from abc import ABC
from typing import Union

from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle \
    import (
    VariationalPrinciple,
)
from qiskit.opflow import (
    CircuitQFI,
)


class RealVariationalPrinciple(ABC, VariationalPrinciple):
    """Class for a Real Variational Principle."""

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ):
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                        ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
        """
        grad_method = "lin_comb"  # we only know how to do this with lin_comb for a real case
        super().__init__(
            qfi_method,
            grad_method,
        )

