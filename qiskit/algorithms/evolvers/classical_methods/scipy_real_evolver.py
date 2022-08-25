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
from typing import Tuple, List
import scipy.sparse as sp
from scipy.sparse.linalg import bicg, norm
import numpy as np

from qiskit.quantum_info.states import Statevector
from qiskit import QuantumCircuit
from qiskit.opflow import StateFn

from .scipy_evolver import SciPyEvolver
from ..evolution_problem import EvolutionProblem
from ..evolution_result import EvolutionResult
from ..real_evolver import RealEvolver


class SciPyRealEvolver(RealEvolver, SciPyEvolver):
    r"""Classical Evolver for real time evolution.

    Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
    under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

    In order to perform one timestep :math:`\delta t` of the evolution, we need to implement:

    .. math::

        |\Psi ( t + \delta t ) \rangle = \exp(-i \delta t H) |\Psi (t ) \rangle

    Which can be approximated as:

    .. math::

        \exp(-i \delta t H) = \left(1+ i \frac{\delta t}{2} H \right) ^{-1}
                                \left( 1 - i \frac{\delta t}{2} H \right)
                                + O\left( \delta t^3 \right)

    Note that this operator is unitary, and thus we won't need to renormalize the state at
    each step. In order to find :math:`|\Psi ( t + \delta t ) \rangle` we then
    need to solve a linear system of equations:

    .. math::

        \left( 1+ i \frac{\delta t}{2} H \right) |\Psi ( t + \delta t ) \rangle
                = \left( 1 - i \frac{\delta t}{2} H \right) |\Psi( t ) \rangle

    """

    def __init__(
        self, threshold: float = 1e-3, bicg_err: float = 0.1, max_iterations: int = np.inf
    ):
        r"""
        Args:
            threshold: The threshold for the error. If timesteps is `None` this will be used
                to estimate the necessary number of timesteps to reach a threshold error.
            bicg_err: Needs to be a value between 0 and 1. `bicg_err` will be the percentage of error
                that comes from solving the linear system of equation with BiCG and
                the rest will be the error that comes from the Taylor expansion.
            max_iterations: The maximum number of iterations to perform.

        Raises:
            ValueError: If `bicg_err` is not between 0 and 1.
            ValueError: If `threshold` is not positive.
            ValueError: If `max_iterations` is not a positive integer.

        """

        if max_iterations < 1:
            raise ValueError("`max_itertations` must be a positive integer.")

        if bicg_err < 0 or bicg_err > 1:
            raise ValueError("`bicg_err` must be between 0 and 1.")

        if threshold < 0:
            raise ValueError("`threshold` must be positive.")

        self.max_iterations = max_iterations
        self.threshold = threshold
        self.bicg_err = bicg_err

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.

        Raises:
            ValueError: If the Hamiltonian is time dependent.

        """

        if evolution_problem.t_param is not None:
            raise ValueError("Time dependent Hamiltonians are not currently supported.")

        (state, lhs_operator, rhs_operator, aux_ops, timesteps, bicg_tol) = self._sparsify(
            evolution_problem=evolution_problem
        )

        # Create empty arrays to store the time evolution of the aux operators.
        operator_numbers = (
            0 if evolution_problem.aux_operators is None else len(evolution_problem.aux_operators)
        )
        ops_ev_mean = np.empty(shape=(operator_numbers, timesteps + 1), dtype=complex)

        # Perform the time evolution and stores the value of the operators at each timestep.
        for ts in range(timesteps):
            ops_ev_mean[:, ts] = self._evaluate_aux_ops(aux_ops, state)

            state = self._step(state, lhs_operator, rhs_operator, bicg_tol)

        ops_ev_mean[:, timesteps] = self._evaluate_aux_ops(aux_ops, state)

        aux_ops_history = self._create_observable_output(ops_ev_mean, evolution_problem)

        aux_ops = self._create_obs_final(ops_ev_mean[:, -1], evolution_problem)

        return EvolutionResult(
            evolved_state=StateFn(state), aux_ops_evaluated=aux_ops, observables=aux_ops_history
        )

    def _sparsify(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix, List[sp.csr_matrix], int, float]:
        """Returns the matrices and parameters needed for time evolution in the appropiate format.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array, the operators needed for time
         evolution as sparse matrices, the number of timesteps in which to divide the time evolution
         and the tolerance for the BiCG method.
        """
        # Convert the initial state and Hamiltonian into sparse matrices.
        if isinstance(evolution_problem.initial_state, QuantumCircuit):
            state = Statevector(evolution_problem.initial_state).data.T
        else:
            state = evolution_problem.initial_state.to_matrix(massive=True).transpose()

        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()

        # Determine the number of timesteps.
        # We use the infinity norm to estimate the norm of the hamiltonian
        timesteps = self.minimal_number_steps(
            norm_hamiltonian=norm(hamiltonian, ord=np.inf),
            time=evolution_problem.time,
            threshold=self.threshold,
        )

        timesteps = min(
            timesteps,
            self.max_iterations,
        )
        timestep = evolution_problem.time / timesteps

        # Create the operators for the time evolution.
        idnty = sp.identity(hamiltonian.shape[0], format="csr")

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian

        if isinstance(evolution_problem.aux_operators, list):
            aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators]
        elif isinstance(evolution_problem.aux_operators, dict):
            aux_ops = [aux_op.to_spmatrix() for aux_op in evolution_problem.aux_operators.values()]
        else:
            aux_ops = []

        return (
            state,
            lhs_operator,
            rhs_operator,
            aux_ops,
            timesteps,
            self.threshold * self.bicg_err / timesteps,
        )

    def minimal_number_steps(
        self, norm_hamiltonian: float, time: float, threshold: float = 1e-4
    ) -> int:
        r"""Calculate the timestep for the given time and threshold.

        We use the fact that the Taylor expansion term of third order in our expansion would be
        :math:`\frac{ ( -i H \delta t ) ^3}{12}` to compute an approximation for how
        many timesteps we need to reach a certain precision.

        Args:
            time: The time to evolve.
            norm_hamiltonian: The operator norm of the Hamiltonian if known or an estimation of
                it (For example the infinity norm of the Hamiltonian). This value will be associated
                with the error of the Hamiltonian.
            threshold: The threshold for the error.

        Returns:
            The number of timesteps needed to reach the error threshold.
        """
        return int(
            np.power(norm_hamiltonian * time, 3 / 2)
            * np.power(12 * threshold * (1 - self.bicg_err), -1 / 2)
            + 1
        )

    def _step(
        self,
        state: np.ndarray,
        lhs_operator: sp.csr_matrix,
        rhs_operator: sp.csr_matrix,
        bicg_tol: float,
    ) -> np.ndarray:
        """Perform one timestep of the evolution.

        # Args:
        #     state: The initial state.
        #     timestep: The timestep to evolve.
        #     hamiltonian: The Hamiltonian to evolve under.

        Returns:
            The evolved state.

        Raises:
            RuntimeError: If the biconjugate gradient solver fails.

        """

        rhs = rhs_operator.dot(state)
        ev_state, exitcode = bicg(A=lhs_operator, b=rhs, x0=state, atol=bicg_tol, tol=bicg_tol)
        if exitcode != 0:
            raise RuntimeError(
                f"The biconjugate gradient solver has falied with exitcode: {exitcode}"
            )
        return ev_state
