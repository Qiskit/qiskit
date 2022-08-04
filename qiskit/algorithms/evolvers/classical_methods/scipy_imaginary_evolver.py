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

"""Classical Quantum Real Time Evolution."""
from typing import Tuple,List
from scipy.sparse.linalg import expm_multiply
import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.opflow import StateFn
from qiskit.quantum_info.states import Statevector


from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..imaginary_evolver import ImaginaryEvolver
from .scipy_evolver import SciPyEvolver


class SciPyImaginaryEvolver(ImaginaryEvolver, SciPyEvolver):
    """Classical Evolver for imaginary time evolution."""

    def __init__(
        self,
        timesteps: int = 1,
    ):
        """
        Args:
            timesteps: The number of timesteps to get data from the observables. At the end of each
                step the state will be renormalized. Note that increassing the amount of
                timesteps will not increase the accuracy of the integration but can avoid
                overflow errors for Hamiltonians with large eigenvalues.

        Raises:
            ValueError: If `timesteps` is not a positive integer.
        """
        if timesteps < 1 or not isinstance(timesteps, int):
            raise ValueError("`timesteps` must be a positive integer.")

        self.timesteps = timesteps

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform imaginary time evolution :math:`\exp(- \tau H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for an imaginary time :math:`\tau = it`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        For each timestep we use SciPy's `expm_multiply` function to evolve the state. Internally,
        this function performs integration steps as well. The number of timesteps specified
        by the user is just to renormalize the state to avoid overflow errors and to get the
        values of the observables at those times. Note that increasing the number of timesteps
        will not direclty increase the accuracy of the integration.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.

        Raises:
            ValueError: If a `NaN` is encountered in the final state of the system.
            ValueError: If the Hamiltonian in the evolution problem is time dependent.
        """

        if evolution_problem.t_param is not None:
            raise ValueError("Time dependent Hamiltonians are not currently supported.")

        # Get the initial state, Hamiltonian and the auxiliary operators.
        state, hamiltonian, aux_ops, aux_ops_2, timestep = self._start(
            evolution_problem=evolution_problem
        )

        # Create empty arrays to store the time evolution of the aux operators.
        operator_number = (
            0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
        )
        ops_ev_mean = np.empty(shape=(operator_number, self.timesteps + 1), dtype=float)

        ops_ev_std = np.empty(shape=(operator_number, self.timesteps + 1), dtype=float)

        for ts in range(self.timesteps):
            ops_ev_mean[:, ts], ops_ev_std[:, ts] = self._evaluate_aux_ops(
                aux_ops,
                aux_ops_2,
                state,
            )
            state = expm_multiply(A=-hamiltonian * timestep, B=state)
            if np.nan in state:
                raise ValueError(
                    "An overflow has probably occured. Try increasing the amount of timesteps."
                )
            state = state / np.linalg.norm(state)

        ops_ev_mean[:, self.timesteps], ops_ev_std[:, self.timesteps] = self._evaluate_aux_ops(
            aux_ops,
            aux_ops_2,
            state,
        )

        aux_ops_history = self._create_observable_output(
            ops_ev_mean,
            ops_ev_std,
            evolution_problem.aux_operators,
        )
        aux_ops = self._create_observable_output(
            ops_ev_mean[:, -1],
            ops_ev_std[:, -1],
            evolution_problem.aux_operators,
        )

        return EvolutionResult(
            evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables=aux_ops_history
        )

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

        # Convert the initial state and Hamiltonian into sparse matrices.
        if isinstance(evolution_problem.initial_state, QuantumCircuit):
            initial_state = Statevector(evolution_problem.initial_state).data.T
        else:
            initial_state = evolution_problem.initial_state.to_matrix(massive=True).transpose()

        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

        # Get the auxiliary operators as sparse matrices.
        if isinstance(evolution_problem.aux_operators, list):
            aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()]
        else:
            aux_ops = []

        aux_ops_2 = [aux_op.dot(aux_op) for aux_op in aux_ops]

        timestep = evolution_problem.time / self.timesteps

        return initial_state, hamiltonian, aux_ops, aux_ops_2, timestep
