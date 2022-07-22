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

from multiprocessing.sharedctypes import Value
import scipy.sparse as sp
from scipy.sparse.linalg import bicg, norm
import numpy as np
from typing import Tuple, Optional

from qiskit.opflow import StateFn


from .evolution_problem import EvolutionProblem
from .evolution_result import EvolutionResult
from .real_evolver import RealEvolver


class ClassicalRealEvolver(RealEvolver):
    """Interface for Quantum Real Time Evolution."""

    def __init__(self, timesteps: Optional[int]  = None, threshold: Optional[float] = 1e-4):
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

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        In order to perform one timestep :math:`\delta t` of the evolution, we need to implement

        .. math::

            |\Psi(t + \delta t)\rangle = \exp(-i \delta t H) |\Psi(t)\rangle

        Which can be approximated as:

        .. math::

            \exp(-i \delta t H) = (1+ i \frac{\delta t}{2} H)^{-1} (1 - i \frac{\delta t}{2} H) + O(\delta t^3)

        Note that this operator is unitary, and thus we won't need to renormalize the state at
        each step. In order to find :math:`|\Psi(t + \delta t)\rangle` we then need to solve a
        linear system of equations:

        .. math::

            (1+ i \frac{\delta t}{2} H) |\Psi(t + \delta t)\rangle = (1 - i \frac{\delta t}{2} H) |\Psi(t)\rangle

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.
        """

        state, lhs_operator, rhs_operator, timesteps = self._start(evolution_problem=evolution_problem)

        for _ in range(timesteps):
            state = self._step(state, lhs_operator, rhs_operator)

        return EvolutionResult(evolved_state=StateFn(state), aux_ops_evaluated=None)

    def _start(
        self, evolution_problem: EvolutionProblem
    ) -> Tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
        """Returns a tuple with the initial state as an array and the operators needed for time
         evolution as sparse matrices.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            A tuple with the initial state as an array and the operators needed for time
            evolution as sparse matrices.
        """
        state = evolution_problem.initial_state.to_matrix(massive=True).transpose()
        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()
        if self.timesteps is None:
            timesteps = self._ntimesteps(time = evolution_problem.time, hamiltonian = hamiltonian, threshold = self.threshold)
            print(timesteps)
        else:
            timesteps = self.timesteps
        timestep = evolution_problem.time / timesteps

        idnty = sp.identity(hamiltonian.shape[0], format="csr")

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian

        return state, lhs_operator, rhs_operator, timesteps

    def minimal_number_steps(self, norm_hamiltonian: float, time: float, threshold = 1e-4) -> int:
        """Calculate the timestep for the given time and threshold.

        we use the fact that the taylor expansion term of third order in our expansion would be
        :math:`\frac{(-i H \delta t)^3}{12} ` to compute an approximation for how many timesteps
        we need to reach a certain precision.

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

    def _step(self, state: np.ndarray , lhs_operator: sp.csr_matrix, rhs_operator: sp.csr_matrix) -> np.ndarray:
        """ "Perform one timestep of the evolution.

        Args:
            state: The initial state.
            timestep: The timestep to evolve.
            hamiltonian: The Hamiltonian to evolve under.

        Returns:
            The evolved state.
        """

        rhs = rhs_operator.dot(state)
        ev_state, exitcode = bicg(A=lhs_operator, b=rhs, atol=1e-8, x0=state)
        if exitcode != 0:
            raise RuntimeError("Failure!")
        return ev_state
