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
from typing import Tuple, List, Optional
from scipy.sparse.linalg import expm_multiply, norm
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
    r"""Classical Evolver for imaginary time evolution.

    Evolves an initial state :math:`|\Psi\rangle` for an imaginary time :math:`\tau = it`
    under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

    For each timestep we use SciPy's `expm_multiply` function to evolve the state. Internally,
    this function performs integration steps as well. The number of timesteps specified
    by the user is just to renormalize the state to avoid overflow errors and to get the
    values of the observables at those times. Note that increasing the number of timesteps
    will not direclty increase the accuracy of the integration.

    """

    def __init__(
        self,
        timesteps: Optional[int] = 1,
    ):
        """
        Args:
            timesteps: The number of timesteps in the simulation.

        Raises:
            ValueError: If `timesteps` is not a positive integer.
        """
        if isinstance(timesteps, int) and timesteps < 1:
            raise ValueError("`timesteps` must be a positive integer.")

        self.timesteps = timesteps

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform imaginary time evolution :math:`\exp(- \tau H)|\Psi\rangle`.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.

        Raises:
            ValueError: If a `NaN` is encountered in the final state of the system.
            RuntimeError: If the Hamiltonian in the evolution problem is time dependent.
            RuntimeError: If the state norm goes to 0 during time evolution
        """

        if evolution_problem.t_param is not None:
            raise ValueError("Time dependent Hamiltonians are not currently supported.")

        # Get the initial state, Hamiltonian and the auxiliary operators.
        state, hamiltonian, aux_ops = self._sparsify(evolution_problem=evolution_problem)

        # Overflow errors with numpy floats happen at arround exp(700). Therefore we need to keep
        # H*tau < 700.

        timesteps = max(
            self.timesteps, int(norm(hamiltonian, ord=np.inf) * evolution_problem.time / 700) + 1
        )

        timestep = evolution_problem.time / timesteps
        # Create empty arrays to store the time evolution of the auxiliary operators.
        operators_number = (
            0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
        )
        ops_ev_mean = np.empty(shape=(operators_number, timesteps + 1), dtype=complex)

        for ts in range(timesteps):
            ops_ev_mean[:, ts] = self._evaluate_aux_ops(aux_ops, state)
            state = expm_multiply(A=-hamiltonian * timestep, B=state)

            if np.nan in state:
                raise RuntimeError(
                    "An overflow has probably occured. Try increasing the amount of timesteps."
                )
            # We need to normalize first with respect to the max value to avoid having a norm
            # with a value that overflows.
            state = state / state.max()
            # We then check that the state hasn't gone to zero.
            state_norm = np.linalg.norm(state)
            if state_norm == 0:
                raise RuntimeError("The norm of the state went to 0.")
            # Finally we normalize with respect to the norm.
            state = state / state_norm

        ops_ev_mean[:, timesteps] = self._evaluate_aux_ops(aux_ops, state)

        aux_ops_history = self._create_observable_output(ops_ev_mean, evolution_problem)
        aux_ops = self._create_obs_final(ops_ev_mean[:, -1], evolution_problem)

        return EvolutionResult(
            evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables=aux_ops_history
        )

    def _sparsify(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, List[sp.csr_matrix], float]:
        """Returns the operators needed for the evolution.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array, the hamiltonian as a sparse matrix, a list of
            the operators to evaluate at each timestep as sparse matrices and the lenght of a timestep.

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

        return initial_state, hamiltonian, aux_ops
