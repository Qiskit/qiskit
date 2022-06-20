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

"""
Abstract Base class of Gradient for Estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..base_estimator import BaseEstimator
from ..estimator_result import EstimatorResult
from .estimator_gradient_result import EstimatorGradientResult


class BaseEstimatorGradient(ABC):
    def __init__(
        self,
        estimator: BaseEstimator,
    ):
        self._estimator = estimator

    @abstractmethod
    def gradient(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorGradientResult:
        ...

    @abstractmethod
    def estimate(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._estimator.close()
