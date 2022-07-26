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

from typing import Tuple, Optional, List
import scipy.sparse as sp
from scipy.sparse.linalg import bicg, norm
import numpy as np


from qiskit.quantum_info.states import Statevector
from qiskit import QuantumCircuit
from qiskit.opflow import StateFn


from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..real_evolver import RealEvolver
from ...list_or_dict import ListOrDict


class ScipyRealEvolver(RealEvolver):
    """Classical Evolver for real time evolution."""

    def __init__(self, timesteps: Optional[int] = None, threshold: Optional[float] = None):
        """
        Args:
            timesteps: The number of timesteps to perform for the time evolution.
            threshold: The threshold for the error. If timesteps is `None` this will be used
            to estimate the necessary number of timesteps to reach a threshold error.

        Raises:
            ValueError: If timesteps and threshold are `None`.
        """
        if timesteps is None and threshold is None:
            raise ValueError("Either timesteps or threshold must be specified.")

        self.timesteps = timesteps
        self.threshold = threshold
        self.aux_operators_time_evolution: Optional[ListOrDict[np.ndarray]] = None

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        In order to perform one timestep :math:`\delta t` of the evolution, we need to implement

        .. math::

            |\Psi \left( t + \delta t \right) \rangle = \exp(-i \delta t H) |\Psi \left(t \right) \rangle

        Which can be approximated as:

        .. math::

            \exp(-i \delta t H) = \left(1+ i \frac{\delta t}{2} H \right) ^{-1}
                                  \left( 1 - i \frac{\delta t}{2} H \right)
                                  + O\left( \delta t^3 \right)

        Note that this operator is unitary, and thus we won't need to renormalize the state at
        each step. In order to find :math:`|\Psi \left( t + \delta t \right) \rangle` we then
        need to solve a linear system of equations:

        .. math::

            \left( 1+ i \frac{\delta t}{2} H \right) |\Psi \left( t + \delta t \right) \rangle
                    = \left( 1 - i \frac{\delta t}{2} H \right) |\Psi\left( t \right) \rangle

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.
        """

        if evolution_problem.t_param is not None:
            raise ValueError("Time dependent hamiltonians are not currently supported.")

        state, lhs_operator, rhs_operator, aux_operators, timesteps = self._start(
            evolution_problem=evolution_problem
        )

        # Perform the time evolution and stores the value of the operators at each timestep.
        for ts in range(timesteps):
            self._evaluate_aux_operators(aux_operators, state, ts)
            state = self._step(state, lhs_operator, rhs_operator)

        self._evaluate_aux_operators(aux_operators, state, timesteps)

        # Creates the right output format for the evaluated auxiliary operators.
        if isinstance(evolution_problem.aux_operators, dict):
            aux_ops_evaluated = dict(
                zip(evolution_problem.aux_operators.keys(), self.aux_operators_time_evolution)
            )
        else:
            aux_ops_evaluated = self.aux_operators_time_evolution

        return EvolutionResult(evolved_state=StateFn(state), aux_ops_evaluated=aux_ops_evaluated)

    def _evaluate_aux_operators(
        self, aux_operators: List[sp.csr_matrix], state: np.ndarray, current_timestep: int
    ):
        """Evaluate the aux operators if they are provided and stores their value."""
        for n, op in enumerate(aux_operators):
            self.aux_operators_time_evolution[n][current_timestep] = state.conjugate().dot(
                op.dot(state)
            )

    def _start(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, List[sp.csr_matrix], float]:
        """Returns a tuple with the initial state as an array, the operators needed for time
         evolution as sparse matrices and the number of timesteps in which to divide the time evolution.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array and the operators needed for time
            evolution as sparse matrices.
        """
        # Convert the initial state and hamiltonian into sparse matrices.

        if isinstance(evolution_problem.initial_state, QuantumCircuit):
            state = Statevector(evolution_problem.initial_state).data.T
        else:
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

        # Create the operators for the time evolution.
        idnty = sp.identity(hamiltonian.shape[0], format="csr")

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian

        # Create empty arrays to store the time evolution of the aux operators.
        if evolution_problem.aux_operators is not None:
            self.aux_operators_time_evolution = [
                np.empty(shape=(timesteps+1,), dtype=float)
                for _ in evolution_problem.aux_operators
            ]
        if isinstance(evolution_problem.aux_operators, list):
            aux_operators = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_operators = [
                aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()
            ]
        else:
            aux_operators = []

        return state, lhs_operator, rhs_operator, aux_operators, timesteps

    def minimal_number_steps(
        self, norm_hamiltonian: float, time: float, threshold: float = 1e-4
    ) -> int:
        r"""Calculate the timestep for the given time and threshold.

        we use the fact that the taylor expansion term of third order in our expansion would be
        :math:`\frac{\left( -i H \delta t \right) ^3}{12} ` to compute an approximation for how
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
        return int(np.power(norm_hamiltonian * time, 3 / 2) * np.power(12 * threshold, -1 / 2)) + 1

    def _ntimesteps(self, time: float, hamiltonian: sp.csr_matrix, threshold: float = 1e-4) -> int:
        """Calculate the number of timesteps needed to reach the threshold error if the user doesn't
        indicate the number of timesteps.

        Uses the infinity norm to estimate the operator norm of the Hamiltonian.
        """
        hnorm = norm(hamiltonian, ord=np.inf)
        return self.minimal_number_steps(norm_hamiltonian=hnorm, time=time, threshold=threshold)

    def  _step(
        self, state: np.ndarray, lhs_operator: sp.csr_matrix, rhs_operator: sp.csr_matrix
    ) -> np.ndarray:
        """ "Perform one timestep of the evolution.

        Args:
            state: The initial state.
            timestep: The timestep to evolve.
            hamiltonian: The Hamiltonian to evolve under.

        Returns:
            The evolved state.

        Raises:
            RuntimeError: If the biconjugate gradient solver fails.

        """

        rhs = rhs_operator.dot(state)

        atol = 1e-8 / self.timesteps if self.threshold is None else self.threshold/self.timesteps
        tol = atol
        ev_state, exitcode = bicg(A=lhs_operator, b=rhs, x0=state, atol= atol, tol =tol)
        if exitcode != 0:
            raise RuntimeError("The biconjugate gradient solver has falied.")
        return ev_state
