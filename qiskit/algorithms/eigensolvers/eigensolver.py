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

"""The eigensolver interface and result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..algorithm_result import AlgorithmResult
from ..list_or_dict import ListOrDict


class Eigensolver(ABC):
    """The eigensolver interface.

    Algorithms that can compute eigenvalues for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> "EigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional.

        Args:
            operator: Qubit operator of the observable.
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                eigenstate of the minimum eigenvalue main result and their expectation values
                returned. For instance, in chemistry, these can be dipole operators and total particle
                count operators, so we can get values for these at the ground state.

        Returns:
             An eigensolver result.
        """
        return EigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

        If the eigensolver computes the eigenvalues of the main operator, then it can compute
        the expectation value of the ``aux_operators`` for that state. Otherwise they will be ignored.

        Returns:
            ``True`` if ``aux_operator`` expectations can be evaluated, ``False`` otherwise.
        """
        return False


class EigensolverResult(AlgorithmResult):
    """Eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalues: np.ndarray | None = None
        self._aux_operators_evaluated: list[
            ListOrDict[tuple[complex, dict[str, Any]]]
        ] | None = None

    @property
    def eigenvalues(self) -> np.ndarray | None:
        """Return the eigenvalues."""
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """Set the eigenvalues."""
        self._eigenvalues = value

    @property
    def aux_operators_evaluated(
        self,
    ) -> list[ListOrDict[tuple[complex, dict[str, Any]]]] | None:
        """Return the aux operator expectation values.

        These values are in fact tuples formatted as (mean, metadata).
        """
        return self._aux_operators_evaluated

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(
        self, value: list[ListOrDict[tuple[complex, dict[str, Any]]]]
    ) -> None:
        """Set the aux operator eigenvalues."""
        self._aux_operators_evaluated = value
