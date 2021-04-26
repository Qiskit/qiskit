# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Amplitude Estimation interface."""

from abc import abstractmethod

from .estimation_problem import EstimationProblem
from ..algorithm_result import AlgorithmResult


class AmplitudeEstimator:
    """The Amplitude Estimation interface."""

    @abstractmethod
    def estimate(self, estimation_problem: EstimationProblem) -> 'AmplitudeEstimatorResult':
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
        self.circuit_results = None
        self.shots = None
        self.estimation = None
        self.estimation_processed = None
        self.num_oracle_queries = None
        self.post_processing = None
        self.confidence_interval = None
        self.confidence_interval_processed = None
