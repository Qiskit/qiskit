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
from typing import Optional, Tuple, Any

import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from ..algorithm_result import AlgorithmResult
from ..list_or_dict import ListOrDict


class SamplingMinimumEigensolver(ABC):
    """The Sampling Minimum Eigensolver Interface."""

    @abstractmethod
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> "SamplingMinimumEigensolverResult":
        """Compute the minimum eigenvalue of a diagonal operator.

        Args:
            operator: Diagonal qubit operator.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                final state.

        Returns:
            A :class:`~.SamplingMinimumEigensolverResult` containing the optimization result.
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
    """Sampling Minimum Eigensolver Result.

    In contrast to the result of a :class:`~.MinimumEigenSolver`, this result also contains
    the best measurement of the overall optimization and the samples of the final state.
    """

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalue = None
        self._eigenstate = None
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
    def best_measurement(self) -> Optional[dict[str, Any]]:
        """Return the best measurement (as bitstring) over the entire optimization."""
        return self._best_measurement

    @best_measurement.setter
    def best_measurement(self, value: dict[str, Any]) -> None:
        """Set the best measurement (as bitstring) over the entire optimization."""
        self._best_measurement = value

    def __str__(self) -> str:
        """Return a string representation of the result."""
        disp = (
            "SamplingMinimumEigensolverResult:\n"
            + f"\tEigenvalue: {self.eigenvalue}\n"
            + f"\tBest measurement\n: {self.best_measurement}\n"
        )
        if self.aux_operator_values is not None:
            disp += f"\n\tAuxiliary operator values: {self.aux_operator_values}\n"

        return disp
