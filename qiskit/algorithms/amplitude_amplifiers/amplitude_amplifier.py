# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The interface for amplification algorithms and results."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Dict, List

import numpy as np

from .amplification_problem import AmplificationProblem
from ..algorithm_result import AlgorithmResult


class AmplitudeAmplifier(ABC):
    """The interface for amplification algorithms."""

    @abstractmethod
    def amplify(self, amplification_problem: AmplificationProblem) -> "AmplificationResult":
        """Run the amplification algorithm.

        Args:
            amplification_problem: The amplification problem.

        Returns:
            The result as a ``AmplificationResult``, where e.g. the most likely state can be queried
            as ``result.top_measurement``.
        """
        raise NotImplementedError


class AmplitudeAmplifierResult(AlgorithmResult):
    """The amplification result base class."""

    def __init__(self) -> None:
        super().__init__()
        self._top_measurement = None
        self._assignment = None
        self._oracle_evaluation = None

    @property
    def top_measurement(self) -> Optional[str]:
        """The most frequently measured output as bitstring.

        Returns:
            The most frequently measured output state.
        """
        return self._top_measurement

    @top_measurement.setter
    def top_measurement(self, value: str) -> None:
        """Set the most frequently measured bitstring.

        Args:
            value: A new value for the top measurement.
        """
        self._top_measurement = value

    @property
    def assignment(self) -> Any:
        """The post-processed value of the most likely bitstring.

        Returns:
            The output of the ``post_processing`` function of the respective
            ``AmplificationProblem``, where the input is the ``top_measurement``. The type
            is the same as the return type of the post-processing function.
        """
        return self._assignment

    @assignment.setter
    def assignment(self, value: Any) -> None:
        """Set the value for the assignment.

        Args:
            value: A new value for the assignment/solution.
        """
        self._assignment = value

    @property
    def oracle_evaluation(self) -> bool:
        """Whether the classical oracle evaluation of the top measurement was True or False.

        Returns:
            The classical oracle evaluation of the top measurement.
        """
        return self._oracle_evaluation

    @oracle_evaluation.setter
    def oracle_evaluation(self, value: bool) -> None:
        """Set the classical oracle evaluation of the top measurement.

        Args:
            value: A new value for the classical oracle evaluation.
        """
        self._oracle_evaluation = value

    @property
    def circuit_results(self) -> Optional[Union[List[np.ndarray], List[Dict[str, int]]]]:
        """Return the circuit results. Can be a statevector or counts dictionary."""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(self, value: Union[List[np.ndarray], List[Dict[str, int]]]) -> None:
        """Set the circuit results."""
        self._circuit_results = value

    @property
    def max_probability(self) -> float:
        """Return the maximum sampling probability."""
        return self._max_probability

    @max_probability.setter
    def max_probability(self, value: float) -> None:
        """Set the maximum sampling probability."""
        self._max_probability = value
