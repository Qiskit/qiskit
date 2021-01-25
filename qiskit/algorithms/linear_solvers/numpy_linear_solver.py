# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The Numpy LinearSolver algorithm (classical)."""

from typing import List, Union, Optional
import logging
import numpy as np

from qiskit import QuantumCircuit
from .linear_solver import LinearSolverResult, LinearSolver
from .observables.linear_system_observable import LinearSystemObservable

logger = logging.getLogger(__name__)


class NumPyLinearSolver(LinearSolver):
    """
    The Numpy LinearSolver algorithm (classical).

    This linear system solver computes the exact value of the given observable(s) or the full
    solution vector if no observable is specified.
    """

    def __init__(self) -> None:
        super().__init__()
        # self._solution = LinearSolverResult()

    def solve(self, matrix: Union[np.ndarray, QuantumCircuit],
              vector: Union[np.ndarray, QuantumCircuit],
              observable: Optional[Union[LinearSystemObservable, List[LinearSystemObservable]]]
              = None) -> LinearSolverResult:
        """
        solve the system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Information to be extracted from the solution.

        Returns:
            LinearSolverResult

        Raises:
            TODO
        """
        # Raise a warning if the matrix or vector are given as a circuit
        # TODO: maybe it's possible to extract the matrix from the circuit
        if isinstance(matrix, QuantumCircuit) or isinstance(vector, QuantumCircuit):
            raise ValueError("The classical algorithm does not accept a QuantumCircuit as an input.")

        solution_vector = np.linalg.solve(matrix, vector)
        solution = LinearSolverResult()
        solution.state = solution_vector
        if observable is not None:
            if isinstance(observable, list):
                solution.observable = []
                for obs in observable:
                    solution.observable.append(obs.evaluate(solution_vector))
            else:
                solution.observable = observable.evaluate(solution_vector)
        else:
            solution.observable = solution_vector
        solution.euclidean_norm = np.linalg.norm(solution_vector)
        return solution
