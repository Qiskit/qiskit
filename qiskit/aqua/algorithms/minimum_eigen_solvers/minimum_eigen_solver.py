# -*- coding: utf-8 -*-

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

"""The Minimum Eigensolver interface"""


from abc import ABC, abstractmethod
from typing import List, Optional

from qiskit.aqua.operators import BaseOperator
from .minimum_eigen_solver_result import MinimumEigensolverResult


class MinimumEigensolver(ABC):
    """
    The Minimum Eigensolver Interface.

    Algorithms that can compute a minimum eigenvalue for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_minimum_eigenvalue(
            self, operator: Optional[BaseOperator] = None,
            aux_operators: Optional[List[BaseOperator]] = None) -> MinimumEigensolverResult:
        """
        Computes minimum eigenvalue. Operator and aux_operators can be supplied here and
        if not None will override any already set into algorithm so it can be reused with
        different operators. While an operator is required by algorithms, aux_operators
        are optional. To 'remove' a previous aux_operators array use an empty list here.

        Args:
            operator: If not None replaces operator in algorithm
            aux_operators:  If not None replaces aux_operators in algorithm

        Returns:
            MinimumEigensolverResult
        """
        if operator is not None:
            self.operator = operator
        if aux_operators is not None:
            self.aux_operators = aux_operators if aux_operators else None
        pass

    def supports_aux_operators(self) -> bool:
        """
        If the minimum eigensolver computes an eigenstate of the main operator then it
        can compute the expectation value of the aux_operators for that state. Otherwise
        they will be ignored

        Returns:
            True if aux_operator expectations can be evaluated, False otherwise
        """
        return False

    @property
    @abstractmethod
    def operator(self) -> BaseOperator:
        """ returns operator """
        pass

    @operator.setter
    @abstractmethod
    def operator(self, operator: BaseOperator) -> None:
        """ set operator """
        pass

    @property
    @abstractmethod
    def aux_operators(self) -> List[BaseOperator]:
        """ returns aux operators """
        pass

    @aux_operators.setter
    @abstractmethod
    def aux_operators(self, aux_operators: List[BaseOperator]) -> None:
        """ set aux operators """
        pass
