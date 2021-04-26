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

"""The Eigensolver interface"""

from abc import ABC, abstractmethod
from typing import List, Optional

from qiskit.opflow import OperatorBase
from ..algorithm_result import AlgorithmResult


class Eigensolver(ABC):
    """The Eigensolver Interface.

    Algorithms that can compute eigenvalues for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_eigenvalues(
            self,
            operator: OperatorBase,
            aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> 'EigensolverResult':
        """
        Computes eigenvalues. Operator and aux_operators can be supplied here and
        if not None will override any already set into algorithm so it can be reused with
        different operators. While an operator is required by algorithms, aux_operators
        are optional. To 'remove' a previous aux_operators array use an empty list here.

        Args:
            operator: Qubit operator of the Observable
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                eigenstate of the minimum eigenvalue main result and their expectation values
                returned. For instance in chemistry these can be dipole operators, total particle
                count operators so we can get values for these at the ground state.

        Returns:
            EigensolverResult
        """
        return EigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            True if aux_operator expectations can be evaluated, False otherwise
        """
        return False


class EigensolverResult(AlgorithmResult):
    """ Eigensolver Result."""

    def __init__(self) -> None:
        super().__init__()
        self.eigenvalues = None
        self.eigenstates = None
        self.aux_operator_eigenvalues = None
