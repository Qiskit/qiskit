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
from typing import Union, Optional, Dict, Callable, Tuple
import numpy as np
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import AlgorithmResult

from .estimation_problem import EstimationProblem


class AmplitudeEstimator:
    """The Amplitude Estimation interface."""

    def __init__(self,
                 quantum_instance: Optional[Union[Backend, BaseBackend, QuantumInstance]] = None
                 ) -> None:
        """
        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        self.quantum_instance = quantum_instance

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance,
                                                       BaseBackend, Backend]) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    @abstractmethod
    def estimate(self, estimation_problem: EstimationProblem) -> 'AmplitudeEstimatorResult':
        """Run the amplitude estimation algorithm.

        Args:
            estimation_problem: An ``EstimationProblem`` describing
        """
        raise NotImplementedError


class AmplitudeEstimatorResult(AlgorithmResult):
    """The results object for amplitude estimation algorithms."""

    @property
    def circuit_results(self) -> Optional[Union[np.ndarray, Dict[str, int]]]:
        """ return circuit result """
        return self.get('circuit_results')

    @circuit_results.setter
    def circuit_results(self, value: Union[np.ndarray, Dict[str, int]]) -> None:
        """ set circuit result """
        self.data['circuit_results'] = value

    @property
    def shots(self) -> int:
        """ return shots """
        return self.get('shots')

    @shots.setter
    def shots(self, value: int) -> None:
        """ set shots """
        self.data['shots'] = value

    @property
    def estimation(self) -> float:
        r"""Return the estimation for the amplitude in :math:`[0, 1]`."""
        return self.get('estimation')

    @estimation.setter
    def estimation(self, value: float) -> None:
        """ set estimation """
        self.data['estimation'] = value

    @property
    def estimation_processed(self) -> float:
        """Return the estimation for the amplitude after the post-processing has been applied."""
        return self.get('estimation_processed')

    @estimation_processed.setter
    def estimation_processed(self, value: float) -> None:
        """ set estimation """
        self.data['estimation_processed'] = value

    @property
    def num_oracle_queries(self) -> int:
        """ return num_oracle_queries """
        return self.get('num_oracle_queries')

    @num_oracle_queries.setter
    def num_oracle_queries(self, value: int) -> None:
        """ set num_oracle_queries """
        self.data['num_oracle_queries'] = value

    @property
    def post_processing(self) -> Callable[[float], float]:
        """ returns post_processing """
        return self._post_processing
        # return self.get('post_processing')

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float]) -> None:
        """ sets post_processing """
        self._post_processing = post_processing
        # self.data['post_processing'] = post_processing

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """ returns the confidence interval (95% by default) """
        return self.get('confidence_interval')

    @confidence_interval.setter
    def confidence_interval(self, confidence_interval: Tuple[float, float]) -> None:
        """ sets confidence interval """
        self.data['confidence_interval'] = confidence_interval

    @property
    def confidence_interval_processed(self) -> Tuple[float, float]:
        """ returns the confidence interval (95% by default) """
        return self.get('confidence_interval_processed')

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, confidence_interval: Tuple[float, float]) -> None:
        """ sets confidence interval """
        self.data['confidence_interval_processed'] = confidence_interval

    @staticmethod
    def from_dict(a_dict: Dict) -> 'AmplitudeEstimationAlgorithmResult':
        """ create new object from a dictionary """
        return AmplitudeEstimatorResult(a_dict)
