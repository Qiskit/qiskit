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
"""The Numpy LinearSolver algorithm (classical)."""

from typing import List, Union, Optional, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .linear_solver import LinearSolverResult, LinearSolver
from .observables.linear_system_observable import LinearSystemObservable


class NumPyLinearSolver(LinearSolver):
    """The Numpy Linear Solver algorithm (classical).

    This linear system solver computes the exact value of the given observable(s) or the full
    solution vector if no observable is specified.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit.algorithms import NumPyLinearSolver
            from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
            from qiskit.algorithms.linear_solvers.observables import MatrixFunctional

            matrix = TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2)
            right_hand_side = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, 1 / 2)
            rhs = right_hand_side / np.linalg.norm(right_hand_side)

            np_solver = NumPyLinearSolver()
            solution = np_solver.solve(matrix, rhs, observable)
            result = solution.observable
    """

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[LinearSystemObservable, BaseOperator, List[BaseOperator]]
        ] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]]], Union[float, List[float]]]
        ] = None,
    ) -> LinearSolverResult:
        """Solve classically the linear system and compute the observable(s)

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
        # Check if either matrix or vector are QuantumCircuits and get the array from them
        if isinstance(vector, QuantumCircuit):
            vector = Statevector(vector).data
        if isinstance(matrix, QuantumCircuit):
            if hasattr(matrix, "matrix"):
                matrix = matrix.matrix
            else:
                matrix = Operator(matrix).data

        solution_vector = np.linalg.solve(matrix, vector)
        solution = LinearSolverResult()
        solution.state = solution_vector
        if observable is not None:
            if isinstance(observable, list):
                solution.observable = []
                for obs in observable:
                    solution.observable.append(obs.evaluate_classically(solution_vector))
            else:
                solution.observable = observable.evaluate_classically(solution_vector)
        solution.euclidean_norm = np.linalg.norm(solution_vector)
        return solution
