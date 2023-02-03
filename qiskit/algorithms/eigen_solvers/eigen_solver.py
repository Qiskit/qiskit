# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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
from typing import Optional, List, Tuple

import numpy as np

from qiskit.opflow import OperatorBase
from qiskit.utils.deprecation import deprecate_function
from ..algorithm_result import AlgorithmResult
from ..list_or_dict import ListOrDict


class Eigensolver(ABC):
    """Pending deprecation: Eigensolver Interface.

    The Eigensolver interface has been superseded by the
    :class:`qiskit.algorithms.eigensolvers.Eigensolver` interface.
    This interface will be deprecated in a future release and subsequently
    removed after that.

    Algorithms that can compute eigenvalues for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @deprecate_function(
        "The Eigensolver interface has been superseded by the "
        "qiskit.algorithms.eigensolvers.Eigensolver interface. "
        "This interface will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute_eigenvalues(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> "EigensolverResult":
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
    """Pending deprecation: Eigensolver Result.

    The EigensolverResult class has been superseded by the
    :class:`qiskit.algorithms.eigensolvers.EigensolverResult` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

    """

    @deprecate_function(
        "The EigensolverResult class has been superseded by the "
        "qiskit.algorithms.eigensolvers.EigensolverResult class. "
        "This class will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def __init__(self) -> None:
        super().__init__()
        self._eigenvalues = None
        self._eigenstates = None
        self._aux_operator_eigenvalues = None

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """returns eigen values"""
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """set eigen values"""
        self._eigenvalues = value

    @property
    def eigenstates(self) -> Optional[np.ndarray]:
        """return eigen states"""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """set eigen states"""
        self._eigenstates = value

    @property
    def aux_operator_eigenvalues(self) -> Optional[List[ListOrDict[Tuple[complex, complex]]]]:
        """Return aux operator expectation values.

        These values are in fact tuples formatted as (mean, standard deviation).
        """
        return self._aux_operator_eigenvalues

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: List[ListOrDict[Tuple[complex, complex]]]) -> None:
        """set aux operator eigen values"""
        self._aux_operator_eigenvalues = value
