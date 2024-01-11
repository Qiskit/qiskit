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

"""The Sampling Minimum Eigensolver interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from qiskit.result import QuasiDistribution
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
        self._eigenvalue: complex | None = None
        self._eigenstate: QuasiDistribution | None = None
        self._aux_operator_values: ListOrDict[tuple[complex, dict[str, Any]]] | None = None
        self._best_measurement: Mapping[str, Any] | None = None

    @property
    def eigenvalue(self) -> complex | None:
        """Return the approximation to the eigenvalue."""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: complex | None) -> None:
        """Set the approximation to the eigenvalue."""
        self._eigenvalue = value

    @property
    def eigenstate(self) -> QuasiDistribution | None:
        """Return the quasi-distribution sampled from the final state.

        The ansatz is sampled when parameterized with the optimal parameters that where obtained
        computing the minimum eigenvalue. The keys represent a measured classical value and the
        value is a float for the quasi-probability of that result.
        """
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: QuasiDistribution | None) -> None:
        """Set the quasi-distribution sampled from the final state."""
        self._eigenstate = value

    @property
    def aux_operators_evaluated(self) -> ListOrDict[tuple[complex, dict[str, Any]]] | None:
        """Return aux operator expectation values and metadata.

        These are formatted as (mean, metadata).
        """
        return self._aux_operator_values

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(
        self, value: ListOrDict[tuple[complex, dict[str, Any]]] | None
    ) -> None:
        self._aux_operator_values = value

    @property
    def best_measurement(self) -> Mapping[str, Any] | None:
        """Return the best measurement over the entire optimization.

        Possesses keys: ``state``, ``bitstring``, ``value``, ``probability``.
        """
        return self._best_measurement

    @best_measurement.setter
    def best_measurement(self, value: Mapping[str, Any]) -> None:
        """Set the best measurement over the entire optimization."""
        self._best_measurement = value

    def __str__(self) -> str:
        """Return a string representation of the result."""
        disp = (
            "SamplingMinimumEigensolverResult:\n"
            + f"\tEigenvalue: {self.eigenvalue}\n"
            + f"\tBest measurement\n: {self.best_measurement}\n"
        )
        if self.aux_operators_evaluated is not None:
            disp += f"\n\tAuxiliary operator values: {self.aux_operators_evaluated}\n"

        return disp
