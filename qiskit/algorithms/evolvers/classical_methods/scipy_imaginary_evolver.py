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

"""Interface for Classical Quantum Real Time Evolution."""
from logging import raiseExceptions
from typing import Tuple, Optional, Union
from scipy.sparse.linalg import expm_multiply
import numpy as np
from qiskit.algorithms.list_or_dict import ListOrDict, List
from qiskit import QuantumCircuit
from qiskit.opflow import StateFn


from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..imaginary_evolver import ImaginaryEvolver
from ...list_or_dict import ListOrDict


class ScipyImaginaryEvolver(ImaginaryEvolver):
    """Classical Evolver for imaginary time evolution."""

    def __init__(
        self,
        timesteps: Optional[int] = None,
    ):
        """
        Args:
            timesteps: The number of timesteps to get data from the observables. At the end of each
            step the state will be renormalized. Note that increassing the ammount of timesteps
            will not increase the accuracy of the integration but can avoid overflow errors for
            hamiltonians with high eigenvalues.
        """

        self.timesteps = timesteps

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform imaginary time evolution :math:`\exp(- \tau H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`\tau = it`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        For each timestep we use scipy's expm_multiply function to evolve the state. Internally,
        this function will perform integration steps as well. The number of timesteps specified
        by the user is just to renormalize the state to avoid overflow errors and to get the
        values of the observables at those times. Note that increasing the number of timesteps
        will not direclty increase the accuracy of the integration.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.

        Raises:
            ValueError: If a NaN is encountered in the final state of the system.
        """

        # Get the initial state, hamiltonian and the auxiliary operators.
        state, hamiltonian, aux_operators, aux_operators_squared, timestep = self._start(
            evolution_problem=evolution_problem
        )

        if self.timesteps is None:
            return self._simple_evolution(
                aux_operators=aux_operators,
                hamiltonian=hamiltonian,
                evolution_problem=evolution_problem,
            )

        # Create empty arrays to store the time evolution of the aux operators.
        operator_number = 0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
        aux_operators_time_evolution_mean = np.empty(
            shape=(operator_number, self.timesteps + 1), dtype=float
        )

        aux_operators_time_evolution_std = np.empty(
            shape=(operator_number, self.timesteps + 1), dtype=float
        )

        for ts in range(self.timesteps):
            (
                aux_operators_time_evolution_mean[:, ts],
                aux_operators_time_evolution_std[:, ts],
            ) = self._evaluate_aux_operators(
                aux_operators,
                aux_operators_squared,
                state,
            )
            state = expm_multiply(A=-hamiltonian * timestep, B=state)
            if np.nan in state:
                raise ValueError(
                    "An overflow has probably occured. Try increasing the ammount of timesteps."
                )
            state = self._renormalize(state)

        (
            aux_operators_time_evolution_mean[:, self.timesteps],
            aux_operators_time_evolution_std[:, self.timesteps],
        ) = self._evaluate_aux_operators(
            aux_operators,
            aux_operators_squared,
            state,
        )

        aux_ops_history = self._create_observable_output(
            aux_operators_time_evolution_mean,
            aux_operators_time_evolution_std,
            evolution_problem.aux_operators,
        )
        aux_ops = self._create_observable_output(aux_operators_time_evolution_mean[:, self.timesteps],
            aux_operators_time_evolution_std[:, self.timesteps],
            evolution_problem.aux_operators,
        )

        return EvolutionResult(evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables = aux_ops_history)


    def _create_observable_output(
        self,
        aux_operators_time_evolution_mean: np.ndarray,
        aux_operators_time_evolution_std: np.ndarray,
        aux_operators: ListOrDict,
    ) -> ListOrDict[Union[Tuple[np.ndarray, np.ndarray], Tuple[complex,complex]]:
        """Creates the right output format for the evaluated auxiliary operators."""
        operator_number = 0 if aux_operators is None else len(aux_operators)
        observable_evolution = [(aux_operators_time_evolution_mean[i],  aux_operators_time_evolution_std[i]) for i in range(operator_number)]

        if isinstance(aux_operators, dict):
            observable_evolution = {key: value for key, value in zip(aux_operators.keys(), observable_evolution)}

        return observable_evolution

    def _evaluate_aux_operators(
        self,
        aux_operators: List[sp.csr_matrix],
        aux_operators_squared: List[sp.csr_matrix],
        state: np.ndarray,
    ):
        """Evaluate the aux operators if they are provided and stores their value."""
        op_mean = np.array([np.real(state.conjugate().dot(op.dot(state))) for op in aux_operators])
        op_std = np.sqrt(
            np.array(
                [np.real(state.conjugate().dot(op2.dot(state))) for op2 in aux_operators_squared]
            )
            - op_mean**2
        )
        return op_mean, op_std

    def _simple_evolution(
        self,
        aux_operators: List[sp.csr_matrix],
        hamiltonian: sp.csr_matrix,
        evolution_problem: EvolutionProblem,
    ):
        state = expm_multiply(A=-hamiltonian * evolution_problem.time, B=state)
        aux_ops_evaluated = [self._evaluate_operator(state, aux_op) for aux_op in aux_operators]
        aux_ops_evaluated = self._create_aux_ops_evaluated(
            aux_ops_evaluated, evolution_problem.aux_operators
        )
        return EvolutionResult(evolved_state=state, aux_ops_evaluated=aux_ops_evaluated)

    def _create_aux_ops_evaluated(self, observable_evolution: List, aux_operators: ListOrDict):
        """Creates the right output format for the evaluated auxiliary operators."""

        if isinstance(aux_operators, dict):
            observable_evolution = dict(zip(aux_operators.keys(), observable_evolution))
        else:
            observable_evolution = [observable_evolution[i] for i in range(len(aux_operators))]

        return observable_evolution

    def _renormalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize the state."""
        return state / np.linalg.norm(state)

    def _evaluate_operator(self, state: np.ndarray, aux_operator: sp.csr_matrix) -> np.ndarray:
        """Evaluate the operator at the current time evolved state."""
        return np.real(state.conj().T @ aux_operator @ state)

    def _start(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, List[sp.csr_matrix], float]:
        """Returns a tuple with the initial state as an array and the operator needed for time
            evolution as a sparse matrix.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array, the operator needed for time
            evolution as a sparse matrix and the number in which to divide the time evolution.
        """

        # Convert the initial state and hamiltonian into sparse matrices.
        if isinstance(evolution_problem.initial_state, QuantumCircuit):
            initial_state = Statevector(evolution_problem.initial_state).data.T
        else:
            initial_state = evolution_problem.initial_state.to_matrix(massive=True).transpose()

        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

        # Get the auxiliary operators as sparse matrices.
        if isinstance(evolution_problem.aux_operators, list):
            aux_operators = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_operators = [
                aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()
            ]
        else:
            aux_operators = []

        aux_operators_squared = [aux_op.dot(aux_op) for aux_op in aux_operators]


        timestep = evolution_problem.time / self.timesteps

        return initial_state, hamiltonian, aux_operators, aux_operators_squared, timestep
