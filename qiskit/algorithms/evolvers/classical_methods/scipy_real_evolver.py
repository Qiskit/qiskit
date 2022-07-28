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

from ast import operator
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

    def __init__(
        self, threshold: float = 1e-3, coeff_a: float = 0.1, max_iterations: Optional[int] = np.inf
    ):
        """
        Args:
            threshold: The threshold for the error. If timesteps is `None` this will be used
            to estimate the necessary number of timesteps to reach a threshold error.
            coeff_a: Needs to be a value between 0 and 1. `coeff_a * threshold` will be the error that
            comes from solving the linear system of equation with BiCG and `(1 - coeff_a) * threshold`
            will be the error that comes from the taylor expansion.
            max_iterations: The maximum number of iterations to perform.
        """

        self.max_iterations = max_iterations
        self.threshold = threshold
        self.coeff_a = coeff_a

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

        (
            state,
            lhs_operator,
            rhs_operator,
            aux_operators,
            aux_operators_squared,
            timesteps,
            bicg_tol,
        ) = self._start(evolution_problem=evolution_problem)

        # Create empty arrays to store the time evolution of the aux operators.
        operator_number = 0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
        aux_operators_time_evolution_mean = np.empty(
            shape=(operator_number, timesteps + 1), dtype=float
        )

        aux_operators_time_evolution_std = np.empty(
            shape=(operator_number, timesteps + 1), dtype=float
        )

        # Perform the time evolution and stores the value of the operators at each timestep.
        for ts in range(timesteps):
            (
                aux_operators_time_evolution_mean[:, ts],
                aux_operators_time_evolution_std[:, ts],
            ) = self._evaluate_aux_operators(
                aux_operators,
                aux_operators_squared,
                state,
            )

            state = self._step(state, lhs_operator, rhs_operator, bicg_tol)

        (
            aux_operators_time_evolution_mean[:,timesteps],
            aux_operators_time_evolution_std[:,timesteps],
        ) = self._evaluate_aux_operators(aux_operators, aux_operators_squared, state)

        aux_ops_history = self._create_observable_output(
            aux_operators_time_evolution_mean,
            aux_operators_time_evolution_std,
            evolution_problem.aux_operators,
        )
        aux_ops = self._create_observable_output(aux_operators_time_evolution_mean[:, timesteps],
            aux_operators_time_evolution_std[:, timesteps],
            evolution_problem.aux_operators,
        )

        return EvolutionResult(evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables = aux_ops_history)

    def _create_observable_output(
        self,
        aux_operators_time_evolution_mean: np.ndarray,
        aux_operators_time_evolution_std: np.ndarray,
        aux_operators: ListOrDict,
    ) -> ListOrDict[Tuple[np.ndarray,np.ndarray]]:
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
    ) -> Tuple[np.ndarray,np.ndarray]:
        """Evaluate the operators at the current state.

        Returns:
            A tuple of the mean and standard deviation of the auxiliary operators.
        """
        op_mean = np.array([np.real(state.conjugate().dot(op.dot(state))) for op in aux_operators])
        op_std = np.sqrt(
            np.array(
                [np.real(state.conjugate().dot(op2.dot(state))) for op2 in aux_operators_squared]
            )
            - op_mean**2
        )
        return op_mean, op_std

    def _start(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, List[sp.csr_matrix], List[sp.csr_matrix], int, float]:
        """Returns a tuple with the initial state as an array, the operators needed for time
         evolution as sparse matrices, the number of timesteps in which to divide the time evolution
         and the tolerance for the BiCG method.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array, the operators needed for time
            evolution as sparse matrices, the operators to evaluate at each timestep as a list
            of sparse matrices as well as a list of the squared operators, the number of timesteps
             in which to divide the time evolution and the tolerance for the BiCG method.
        """
        # Convert the initial state and hamiltonian into sparse matrices.
        if isinstance(evolution_problem.initial_state, QuantumCircuit):
            state = Statevector(evolution_problem.initial_state).data.T
        else:
            state = evolution_problem.initial_state.to_matrix(massive=True).transpose()

        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

        # Determine the number of timesteps.
        timesteps = min(self._ntimesteps(
            time=evolution_problem.time, hamiltonian=hamiltonian, threshold=self.threshold
        ), self.max_iterations)
        timestep = evolution_problem.time / timesteps

        # Create the operators for the time evolution.
        idnty = sp.identity(hamiltonian.shape[0], format="csr")

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian

        if isinstance(evolution_problem.aux_operators, list):
            aux_operators = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_operators = [
                aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()
            ]
        else:
            aux_operators = []

        aux_operators_squared = [aux_op.dot(aux_op) for aux_op in aux_operators]

        return (
            state,
            lhs_operator,
            rhs_operator,
            aux_operators,
            aux_operators_squared,
            timesteps,
            self._bicg_tol(timesteps),
        )

    def _bicg_tol(self, timesteps: int) -> float:
        """Returns the tolerance for the BiCG solver.

        Args:
            timesteps: The number of timesteps in the evolution.

        Returns:
            The tolerance for the BiCG solver.
        """
        return self.threshold * self.coeff_a / timesteps

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
        return (
            int(
                np.power(norm_hamiltonian * time, 3 / 2)
                * np.power(12 * threshold * (1 - self.coeff_a), -1 / 2)
            )
            + 1
        )

    def _ntimesteps(self, time: float, hamiltonian: sp.csr_matrix, threshold: float = 1e-4) -> int:
        """Calculate the number of timesteps needed to reach the threshold error if the user doesn't
        indicate the number of timesteps.

        Uses the infinity norm to estimate the operator norm of the Hamiltonian.
        Returns:
            The number of timesteps needed to reach the error threshold.
        """
        hnorm = norm(hamiltonian, ord=np.inf)
        return self.minimal_number_steps(norm_hamiltonian=hnorm, time=time, threshold=threshold)

    def _step(
        self,
        state: np.ndarray,
        lhs_operator: sp.csr_matrix,
        rhs_operator: sp.csr_matrix,
        bicg_tol: float,
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
        ev_state, exitcode = bicg(A=lhs_operator, b=rhs, x0=state, atol=bicg_tol, tol=bicg_tol)
        if exitcode != 0:
            raise RuntimeError("The biconjugate gradient solver has falied.")
        return ev_state
