# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The NumPy minimum eigensolver algorithm."""

from __future__ import annotations

from typing import Callable
import logging
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

# TODO this path will need updating
from ..eigen_solvers.numpy_eigen_solver import NumPyEigensolver
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult
from ..list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


class NumPyMinimumEigensolver(MinimumEigensolver):
    """
    The NumPy minimum eigensolver algorithm.
    """

    def __init__(
        self,
        filter_criterion: Callable[
            [list | np.ndarray, float, ListOrDict[float] | None], bool
        ] = None,
    ) -> None:
        """
        Args:
            filter_criterion: callable that allows to filter eigenvalues/eigenstates. The minimum
                eigensolver is only searching over feasible states and returns an eigenstate that
                has the smallest eigenvalue among feasible states. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to consider this value or not. If there is no
                feasible element, the result can even be empty.
        """
        self._ces = NumPyEigensolver(filter_criterion=filter_criterion)

    @property
    def filter_criterion(
        self,
    ) -> Callable[[list | np.ndarray, float, ListOrDict[float] | None], bool] | None:
        """Filters the eigenstates/eigenvalues."""
        return self._ces.filter_criterion

    @filter_criterion.setter
    def filter_criterion(
        self,
        filter_criterion: Callable[[list | np.ndarray, float, ListOrDict[float] | None], bool],
    ) -> None:
        self._ces.filter_criterion = filter_criterion

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return NumPyEigensolver.supports_aux_operators()

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> NumPyMinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        result_ces = self._ces.compute_eigenvalues(operator, aux_operators)
        result = NumPyMinimumEigensolverResult()
        if result_ces.eigenvalues is not None and len(result_ces.eigenvalues) > 0:
            result.eigenvalue = result_ces.eigenvalues[0]
            result.eigenstate = result_ces.eigenstates[0]
            if result_ces.aux_operator_eigenvalues:
                result.aux_operator_eigenvalues = result_ces.aux_operator_eigenvalues[0]

        logger.debug("NumPy minimum eigensolver result: %s", result)

        return result


class NumPyMinimumEigensolverResult(MinimumEigensolverResult):
    """NumPy minimum eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenstate = None

    @property
    def eigenstate(self) -> np.ndarray | None:
        """The eigenstate corresponding to the computed minimum eigenvalue."""
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        self._eigenstate = value
