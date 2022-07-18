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
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow import StateFn


from .evolution_problem import EvolutionProblem
from .evolution_result import EvolutionResult
from .real_evolver import RealEvolver


class ClassicalRealEvolver(RealEvolver):
    """Interface for Quantum Real Time Evolution."""

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
        state = evolution_problem.initial_state.to_matrix().transpose()
        hamiltonian = evolution_problem.hamiltonian.to_spmatrix()
        ntimesteps = self._ntimesteps(
            evolution_problem.time,
            evolution_problem.hamiltonian,
            evolution_problem.truncation_threshold,
        )
        timestep = evolution_problem.time / ntimesteps

        for t in range(ntimesteps):
            state = self._step(state, timestep, hamiltonian)

        return EvolutionResult(
            evolved_state=StateFn(state),
        )

    def _ntimesteps(self, time, hamiltonian, threshold):
        """Calculate the timestep for the given time and threshold.

        we use the fact that the taylor expansion term of third order in our expansion would be
        :math:`\frac{(-i H \delta t)^3}{12} ` to compute an approximation for how many timesteps
        we need to reach a certain precision.
        """
        hnorm = 1  # Frobenius norm.
        return int(time * hnorm / np.power(12 * threshold, 1 / 3))

    def _step(self, state, timestep, hamiltonian):
        """ "Perform one timestep of the evolution.

        Args:
            state: The initial state.
            timestep: The timestep to evolve.
            hamiltonian: The Hamiltonian to evolve under.

        Returns:
            The evolved state.
        """
        idnty = sp.identity(
            hamiltonian.shape[0], format="csr"
        )  # What would be the best format for this?

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian
        rhs = rhs_operator.dot(state)
        ev_state, exitcode = sp.linalg.bicg(A=lhs_operator, b=rhs, atol=1e-8)

        if exitcode != 0:
            raise RuntimeError("Failure!")
        return ev_state
