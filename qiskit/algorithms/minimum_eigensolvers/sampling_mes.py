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

"""The Sampling Minimum Eigensolver interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from qiskit.opflow import OperatorBase
from ..algorithm_result import AlgorithmResult
from ..list_or_dict import ListOrDict


class SamplingMinimumEigensolver(ABC):
    """The Sampling Minimum Eigensolver Interface."""

    @abstractmethod
    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> "SamplingMinimumEigensolverResult":
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
        pass

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


class SamplingMinimumEigensolverResult(AlgorithmResult):
    """Minimum Eigensolver Result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalue = None
        self._eigenstate = None
        self._optimal_point = None
        self._optimal_parameters = None
        self._aux_operator_values = None
        self._best_measurement = None

    @property
    def eigenvalue(self) -> Optional[complex]:
        """Return the approximation to the eigenvalue."""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: complex) -> None:
        """Set the approximation to the eigenvalue."""
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
    def aux_operator_values(self) -> Optional[ListOrDict[Tuple[complex, complex]]]:
        """Return aux operator expectation values.

        These values are in fact tuples formatted as (mean, standard deviation).
        """
        return self._aux_operator_values

    @aux_operator_values.setter
    def aux_operator_values(self, value: ListOrDict[Tuple[complex, complex]]) -> None:
        """set aux operator eigen values"""
        self._aux_operator_values = value

    @property
    def optimal_point(self) -> Optional[np.ndarray]:
        """Return the optimal point."""
        return self._optimal_point

    @optimal_point.setter
    def optimal_point(self, value: np.ndarray) -> None:
        """Set the optimal point."""
        self._optimal_point = value

    @property
    def optimal_parameters(self) -> Optional[dict]:
        """Return the optimal parameters as dictionary with `{parameter: value}` pairs."""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: dict) -> None:
        """Set the optimal parameters as dictionary with `{parameter: value}` pairs."""
        self._optimal_parameters = value

    @property
    def best_measurement(self) -> Optional[str]:
        """Return the best measurement (as bitstring) over the entire optimization."""
        return self._best_measurement

    @best_measurement.setter
    def best_measurement(self, value: str) -> None:
        """Set the best measurement (as bitstring) over the entire optimization."""
        self._best_measurement = value

    def __str__(self) -> str:
        """Return a string representation of the result."""
        disp = (
            "SamplingMinimumEigensolverResult:\n"
            + f"\tEigenvalue: {self.eigenvalue}\n"
            + f"\tBest measurement: {self.best_measurement}\n"
            + f"\tOptimal point: {self.optimal_point}"
        )
        if self.aux_operator_values is not None:
            disp += f"\n\tAuxiliary operator values: {self.aux_operator_values}\n"

        return disp
