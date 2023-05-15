# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Amplitude Estimation interface."""

from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Callable

import numpy as np

from .estimation_problem import EstimationProblem
from ..algorithm_result import AlgorithmResult


class AmplitudeEstimator(ABC):
    """The Amplitude Estimation interface."""

    @abstractmethod
    def estimate(self, estimation_problem: EstimationProblem) -> "AmplitudeEstimatorResult":
        """Run the amplitude estimation algorithm.

        Args:
            estimation_problem: An ``EstimationProblem`` containing all problem-relevant information
                such as the state preparation and the objective qubits.
        """
        raise NotImplementedError


class AmplitudeEstimatorResult(AlgorithmResult):
    """The results object for amplitude estimation algorithms."""

    def __init__(self) -> None:
        super().__init__()
        self._circuit_results: np.ndarray | dict[str, int] | None = None
        self._shots: int | None = None
        self._estimation: float | None = None
        self._estimation_processed: float | None = None
        self._num_oracle_queries: int | None = None
        self._post_processing: Callable[[float], float] | None = None
        self._confidence_interval: tuple[float, float] | None = None
        self._confidence_interval_processed: tuple[float, float] | None = None

    @property
    def circuit_results(self) -> np.ndarray | dict[str, int] | None:
        """Return the circuit results. Can be a statevector or counts dictionary."""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(self, value: np.ndarray | dict[str, int]) -> None:
        """Set the circuit results."""
        self._circuit_results = value

    @property
    def shots(self) -> int:
        """Return the number of shots used. Is 1 for statevector-based simulations."""
        return self._shots

    @shots.setter
    def shots(self, value: int) -> None:
        """Set the number of shots used."""
        self._shots = value

    @property
    def estimation(self) -> float:
        r"""Return the estimation for the amplitude in :math:`[0, 1]`."""
        return self._estimation

    @estimation.setter
    def estimation(self, value: float) -> None:
        r"""Set the estimation for the amplitude in :math:`[0, 1]`."""
        self._estimation = value

    @property
    def estimation_processed(self) -> float:
        """Return the estimation for the amplitude after the post-processing has been applied."""
        return self._estimation_processed

    @estimation_processed.setter
    def estimation_processed(self, value: float) -> None:
        """Set the estimation for the amplitude after the post-processing has been applied."""
        self._estimation_processed = value

    @property
    def num_oracle_queries(self) -> int:
        """Return the number of Grover oracle queries."""
        return self._num_oracle_queries

    @num_oracle_queries.setter
    def num_oracle_queries(self, value: int) -> None:
        """Set the number of Grover oracle queries."""
        self._num_oracle_queries = value

    @property
    def post_processing(self) -> Callable[[float], float]:
        """Return a handle to the post processing function."""
        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float]) -> None:
        """Set a handle to the post processing function."""
        self._post_processing = post_processing

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Return the confidence interval for the amplitude (95% interval by default)."""
        return self._confidence_interval

    @confidence_interval.setter
    def confidence_interval(self, confidence_interval: tuple[float, float]) -> None:
        """Set the confidence interval for the amplitude (95% interval by default)."""
        self._confidence_interval = confidence_interval

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the post-processed confidence interval (95% interval by default)."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, confidence_interval: tuple[float, float]) -> None:
        """Set the post-processed confidence interval (95% interval by default)."""
        self._confidence_interval_processed = confidence_interval
