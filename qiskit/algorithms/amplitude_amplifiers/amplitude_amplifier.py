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

from .amplification_problem import AmplificationProblem
from ..algorithm_result import AlgorithmResult


class AmplitudeAmplifier(ABC):
    """The interface for amplification algorithms."""

    @abstractmethod
    def amplify(self, amplification_problem: AmplificationProblem) -> 'AmplificationResult':
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
        self.top_measurement = None
        self.assignment = None
        self.oracle_evaluation = None
