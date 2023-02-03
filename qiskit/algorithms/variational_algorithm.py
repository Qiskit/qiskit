# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Algorithm Base Class.

This class can be used an interface for working with Variation Algorithms, such as VQE,
QAOA, or QSVM, and also provides helper utilities for implementing new variational algorithms.
Writing a new variational algorithm is a simple as extending this class, implementing a cost
function for the new algorithm to pass to the optimizer, and running :meth:`find_minimum` method
of this class to carry out the optimization. Alternatively, all of the functions below can be
overridden to opt-out of this infrastructure but still meet the interface requirements.

.. note::

    This component has some function that is normally random. If you want to reproduce behavior
    then you should set the random number generator seed in the algorithm_globals
    (``qiskit.utils.algorithm_globals.random_seed = seed``).
"""

from typing import Optional, Dict
from abc import ABC, abstractmethod
import numpy as np

from qiskit.circuit import QuantumCircuit

from .algorithm_result import AlgorithmResult
from .optimizers import OptimizerResult


class VariationalAlgorithm(ABC):
    """The Variational Algorithm Base Class."""

    @property
    @abstractmethod
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point."""
        pass

    @initial_point.setter
    @abstractmethod
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets initial point."""
        pass


class VariationalResult(AlgorithmResult):
    """Variation Algorithm Result."""

    def __init__(self) -> None:
        super().__init__()
        self._optimizer_evals = None
        self._optimizer_time = None
        self._optimal_value = None
        self._optimal_point = None
        self._optimal_parameters = None
        self._optimizer_result = None
        self._optimal_circuit = None

    @property
    def optimizer_evals(self) -> Optional[int]:
        """Returns number of optimizer evaluations"""
        return self._optimizer_evals

    @optimizer_evals.setter
    def optimizer_evals(self, value: int) -> None:
        """Sets number of optimizer evaluations"""
        self._optimizer_evals = value

    @property
    def optimizer_time(self) -> Optional[float]:
        """Returns time taken for optimization"""
        return self._optimizer_time

    @optimizer_time.setter
    def optimizer_time(self, value: float) -> None:
        """Sets time taken for optimization"""
        self._optimizer_time = value

    @property
    def optimal_value(self) -> Optional[float]:
        """Returns optimal value"""
        return self._optimal_value

    @optimal_value.setter
    def optimal_value(self, value: int) -> None:
        """Sets optimal value"""
        self._optimal_value = value

    @property
    def optimal_point(self) -> Optional[np.ndarray]:
        """Returns optimal point"""
        return self._optimal_point

    @optimal_point.setter
    def optimal_point(self, value: np.ndarray) -> None:
        """Sets optimal point"""
        self._optimal_point = value

    @property
    def optimal_parameters(self) -> Optional[Dict]:
        """Returns the optimal parameters in a dictionary"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: Dict) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value

    @property
    def optimizer_result(self) -> Optional[OptimizerResult]:
        """Returns the optimizer result"""
        return self._optimizer_result

    @optimizer_result.setter
    def optimizer_result(self, value: OptimizerResult) -> None:
        """Sets optimizer result"""
        self._optimizer_result = value

    @property
    def optimal_circuit(self) -> QuantumCircuit:
        """The optimal circuits. Along with the optimal parameters,
        these can be used to retrieve the minimum eigenstate.
        """
        return self._optimal_circuit

    @optimal_circuit.setter
    def optimal_circuit(self, optimal_circuit: QuantumCircuit) -> None:
        self._optimal_circuit = optimal_circuit
