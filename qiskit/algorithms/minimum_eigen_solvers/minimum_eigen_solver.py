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

import numpy as np
from qiskit.opflow import OperatorBase
from ..algorithm_result import AlgorithmResult


class MinimumEigensolver(ABC):
    """The Minimum Eigensolver Interface.

    Algorithms that can compute a minimum eigenvalue for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> "MinimumEigensolverResult":
        """
        Computes minimum eigenvalue. Operator and aux_operators can be supplied here and
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
            MinimumEigensolverResult
        """
        return MinimumEigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

        If the minimum eigensolver computes an eigenstate of the main operator then it
        can compute the expectation value of the aux_operators for that state. Otherwise
        they will be ignored.

        Returns:
            True if aux_operator expectations can be evaluated, False otherwise
        """
        return False


class MinimumEigensolverResult(AlgorithmResult):
    """Minimum Eigensolver Result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalue = None
        self._eigenstate = None
        self._aux_operator_eigenvalues = None

    @property
    def eigenvalue(self) -> Optional[complex]:
        """returns eigen value"""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: complex) -> None:
        """set eigen value"""
        self._eigenvalue = value

    @property
    def eigenstate(self) -> Optional[np.ndarray]:
        """return eigen state"""
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        """set eigen state"""
        self._eigenstate = value

    @property
    def aux_operator_eigenvalues(self) -> Optional[np.ndarray]:
        """return aux operator eigen values"""
        return self._aux_operator_eigenvalues

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: np.ndarray) -> None:
        """set aux operator eigen values"""
        self._aux_operator_eigenvalues = value
