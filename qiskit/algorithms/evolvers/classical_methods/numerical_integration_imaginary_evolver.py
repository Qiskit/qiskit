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

import scipy.sparse as sp
from scipy.sparse.linalg import norm
import numpy as np
from typing import Tuple, Optional
from qiskit.algorithms.list_or_dict import ListOrDict, List

from qiskit.opflow import StateFn


from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..imaginary_evolver import ImaginaryEvolver
from ...list_or_dict import ListOrDict


class NumericalIntegrationImaginaryEvolver(ImaginaryEvolver):
    """Classical Evolver for imaginary time evolution."""

    def __init__(
        self,
        timesteps: Optional[int] = None,
        threshold: Optional[float] = None,
        order: str = "first",
    ):
        """
        Args:
            timesteps: The number of timesteps to perform for the time evolution.
            threshold: The threshold for the error. If timesteps is `None` this will be used
            to estimate the necessary number of timesteps to reach a threshold error.
            order: The order of the taylor expansion. Either 'first' or 'second'.

        Raises:
            ValueError: If timesteps and threshold are `None` or if the order of the taylor expansion
            is not propely specified.
        """
        if timesteps is None and threshold is None:
            raise ValueError("Either timesteps or threshold must be specified.")

        if order != "first" and order != "second":
            raise ValueError("Order must be either 'first' or 'second'.")

        self.timesteps = timesteps
        self.threshold = threshold
        self.order = order
        self.aux_operators_time_evolution: Optional[ListOrDict[np.ndarray]] = None

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform imaginary time evolution :math:`\exp(- \tau H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`\tau = it`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        In order to perform one timestep :math:`\delta \tau` of the evolution, we need to implement

        .. math::

            |\Psi \left( \tau + \delta \tau \right) \rangle = \exp(-\delta \tau H) |\Psi \left( \tau \right) \rangle

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

        state, step_evolution_operator, aux_operators, timesteps = self._start(
            evolution_problem=evolution_problem
        )

        #Perform the time evolution and stores the value of the operators at each timestep.
        for t in range(timesteps):
            state = self._step(state, step_evolution_operator)
            self._evaluate_aux_operators(aux_operators, state,t)

        #Creates the right output format for the evaluated auxiliary operators.
        if isinstance(evolution_problem.aux_operators, dict):
            aux_ops_evaluated = dict(
                zip(evolution_problem.aux_operators.keys(), self.aux_operators_time_evolution)
            )
        else:
            aux_ops_evaluated = self.aux_operators_time_evolution

        return EvolutionResult(evolved_state=StateFn(state), aux_ops_evaluated=aux_ops_evaluated)

    def _evaluate_aux_operators(self, aux_operators: List[sp.csr_matrix], state: np.ndarray, t:int):
        """Evaluate the aux operators if they are provided and stores their value."""
        for n,op in enumerate(aux_operators):
            self.aux_operators_time_evolution[n][t] = state.conjugate().dot(op.dot(state))


    def _start(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix,List[sp.csr_matrix],float]:
        """Returns a tuple with the initial state as an array and the operator needed for time
            evolution as a sparse matrix.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array, the operator needed for time
            evolution as a sparse matrix and the number in which to divide the time evolution.
        """

        # Convert the initial state and hamiltonian into sparse matrices.
        state = evolution_problem.initial_state.to_matrix(massive=True).transpose()
        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

        # Determine the number of timesteps.
        if self.timesteps is None:
            timesteps = self._ntimesteps(
                time=evolution_problem.time, hamiltonian=hamiltonian, threshold=self.threshold
            )
            print(timesteps)
        else:
            timesteps = self.timesteps

        timestep = evolution_problem.time / timesteps

        # Create the operator for the time evolution.
        if self.order == "first":
            step_evolution_operator = -hamiltonian * timestep
        elif self.order == "second":
            step_evolution_operator = -hamiltonian * timestep + timestep / 2 * hamiltonian.dot(
                hamiltonian
            )

        # Create empty arrays to store the time evolution of the aux operators.
        if evolution_problem.aux_operators is not None:
            self.aux_operators_time_evolution = [
                np.empty(shape=(timesteps,),dtype=np.complex128) for _ in evolution_problem.aux_operators
            ]
        if isinstance(evolution_problem.aux_operators, list):
            aux_operators = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_operators = [
                aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()
            ]
        else:
            aux_operators = []

        return state, step_evolution_operator, aux_operators, timesteps

    def minimal_number_steps(self, norm_hamiltonian: float, time: float, threshold=1e-4) -> int:
        r"""Calculate the timestep for the given time and threshold.

        we use the fact that the taylor expansion term of third order in our expansion would be
        :math:`\frac{\left( - H \delta \tau \right) ^3}{6} ` to compute an approximation for how
        many timesteps we need to reach a certain precision.

        Args:
            time: The time to evolve.
            norm_hamiltonian: The operator norm of the Hamiltonian if known or an estimation of
                             it (For example the infinity norm of the Hamiltonian). This value
                             will be associated with the error of the Hamiltonian.
            threshold: The threshold for the error.

        Returns:
            The number of timesteps needed to reach the error threshold.
        """
        if self.order == "first":
            return int(np.power(norm_hamiltonian * time, 2) * np.power(2 * threshold, -1)) + 1
        elif self.order == "second":
            return (
                int(np.power(norm_hamiltonian * time, 3 / 2) * np.power(6 * threshold, -1 / 2)) + 1
            )

    def _ntimesteps(self, time: float, hamiltonian: sp.csr_matrix, threshold: float = 1e-4) -> int:
        """Calculate the number of timesteps needed to reach the threshold error if the user doesn't
        indicate the number of timesteps.

        Uses the infinity norm to estimate the operator norm of the Hamiltonian.
        """
        hnorm = norm(hamiltonian, ord=np.inf)
        return self.minimal_number_steps(norm_hamiltonian=hnorm, time=time, threshold=threshold)

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize the state."""
        return state / np.linalg.norm(state)

    def _step(self, state: np.ndarray, step_evolution_operator: sp.csr_matrix) -> np.ndarray:
        """ "Perform one timestep of the evolution.

        Args:
            state: The initial state.
            step_evolution_operator: Operator for performing one timestep.

        Returns:
            The evolved state.

        """

        return self._normalize(state + step_evolution_operator.dot(state))
