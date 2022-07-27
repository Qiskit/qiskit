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
from typing import Tuple, Optional
from pyrsistent import s
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import numpy as np
from qiskit.algorithms.list_or_dict import ListOrDict, List
from qiskit.quantum_info.states import Statevector
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
        # threshold: Optional[float] = None,
        # order: str = "first",
    ):
        """
        Args:
            timesteps: The number of timesteps to perform for the time evolution.
            threshold: The threshold for the error. If timesteps is `None` this will be used
            to estimate the necessary number of timesteps to reach a threshold error.
            order: The order of the taylor expansion. Either 'first' or 'second'.

        Raises:
            ValueError: If timesteps is `None`.
        """
        if timesteps is None:
            raise ValueError("timesteps must be specified.")
        # if order not in ("first", "second"):
        #     raise ValueError("Order must be either 'first' or 'second'.")

        self.timesteps = timesteps
        # self.threshold = threshold
        # self.order = order
        # self.aux_operators_time_evolution: Optional[ListOrDict[np.ndarray]] = None

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform imaginary time evolution :math:`\exp(- \tau H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`\tau = it`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        In order to perform one timestep :math:`\delta \tau` of the evolution, we need to implement

        .. math::

            |\Psi \left( \tau + \delta \tau \right) \rangle =
             \exp(-\delta \tau H) |\Psi \left( \tau \right) \rangle

        Which can be approximated as:

        .. math::

            \exp(- \delta \tau H) = 1 - \delta \tau H + O \left( \delta \tau^2 \right)

        Note that this operator is not unitary, and thus we will need to renormalize the state at
        each step.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.
        """

        # Get the initial state, hamiltonian and the auxiliary operators.
        state, hamiltonian, aux_operators, timestep = self._start(evolution_problem = evolution_problem)

        # Initialize empty array to store observable evolution.
        observable_evolution = np.empty((len(aux_operators),self.timesteps+1),dtype = float)

        for ts in range(self.timesteps):
            for i,aux_operator in enumerate(aux_operators):
                observable_evolution[i,ts] = self._evaluate_operator(state, aux_operator)
            state = expm_multiply(A= - hamiltonian * timestep, B = state)
            state = self._renormalize(state)


        for i,aux_operator in enumerate(aux_operators):
            observable_evolution[i,self.timesteps] = self._evaluate_operator(state, aux_operator)

        # Creates the right output format for the evaluated auxiliary operators.
        if isinstance(evolution_problem.aux_operators, dict):
            observable_evolution = dict(
                zip(evolution_problem.aux_operators.keys(), observable_evolution)
            )
        else:
            observable_evolution = [observable_evolution[i] for i in range(len(aux_operators))]

        return EvolutionResult(
            evolved_state=StateFn(state),
            aux_ops_evaluated=observable_evolution
        )

    def _renormalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize the state."""
        return state / np.linalg.norm(state)

    def _evaluate_operator(
            self, state: np.ndarray, aux_operator: sp.csr_matrix
        ) -> np.ndarray:
            """Evaluate the operator at the current time evolved state."""
            return state.conj().T @ aux_operator @ state

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

        timestep = evolution_problem.time / self.timesteps

        return initial_state, hamiltonian, aux_operators, timestep