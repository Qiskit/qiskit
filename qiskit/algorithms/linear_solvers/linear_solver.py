# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for linear systems solvers."""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .observables.linear_system_observable import LinearSystemObservable
from ..algorithm_result import AlgorithmResult


class LinearSolverResult(AlgorithmResult):
    """A base class for linear systems results.

    The linear systems algorithms return an object of the type ``LinearSystemsResult``
    with the information about the solution obtained.
    """

    def __init__(self) -> None:
        super().__init__()

        # Set the default to None, if the algorithm knows how to calculate it can override it.
        self._state = None
        self._observable = None
        self._euclidean_norm = None
        self._circuit_results = None

    @property
    def observable(self) -> Union[float, List[float]]:
        """return the (list of) calculated observable(s)"""
        return self._observable

    @observable.setter
    def observable(self, observable: Union[float, List[float]]) -> None:
        """Set the value(s) of the observable(s).

        Args:
            observable: The new value(s) of the observable(s).
        """
        self._observable = observable

    @property
    def state(self) -> Union[QuantumCircuit, np.ndarray]:
        """return either the circuit that prepares the solution or the solution as a vector"""
        return self._state

    @state.setter
    def state(self, state: Union[QuantumCircuit, np.ndarray]) -> None:
        """Set the solution state as either the circuit that prepares it or as a vector.

        Args:
            state: The new solution state.
        """
        self._state = state

    @property
    def euclidean_norm(self) -> float:
        """return the euclidean norm if the algorithm knows how to calculate it"""
        return self._euclidean_norm

    @euclidean_norm.setter
    def euclidean_norm(self, norm: float) -> None:
        """Set the euclidean norm of the solution.

        Args:
            norm: The new euclidean norm of the solution.
        """
        self._euclidean_norm = norm

    @property
    def circuit_results(self) -> Union[List[float], List[Result]]:
        """return the results from the circuits"""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(self, results: Union[List[float], List[Result]]):
        self._circuit_results = results


class LinearSolver(ABC):
    """An abstract class for linear system solvers in Qiskit."""

    @abstractmethod
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                BaseOperator,
                List[LinearSystemObservable],
                List[BaseOperator],
            ]
        ] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> LinearSolverResult:
        """Solve the system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is ``None``.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Returns:
            The result of the linear system.
        """
        raise NotImplementedError
