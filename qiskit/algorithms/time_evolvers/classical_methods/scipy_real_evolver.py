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
from .evolve import _evolve
from ..time_evolution_problem import TimeEvolutionProblem
from ..time_evolution_result import TimeEvolutionResult
from ..real_time_evolver import RealTimeEvolver


class SciPyRealEvolver(RealTimeEvolver):
    r"""Classical Evolver for real time evolution.

    Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
    under a Hamiltonian :math:`H`, as provided in the ``evolution_problem``.
    Note that the precision of the evolver does not depend on the number of
    timesteps taken.
    """

    def __init__(self, num_timesteps: int):
        """
        Args:
            num_timesteps: The number of timesteps in the simulation.
        Raises:
            ValueError: If `steps` is not a positive integer.
        """
        self.num_timesteps = num_timesteps

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:
        r"""Perform real time evolution :math:`\exp(-i t H)|\Psi\rangle`.

        Evolves an initial state :math:`|\Psi\rangle` for a time :math:`t`
        under a Hamiltonian  :math:`H`, as provided in the ``evolution_problem``.

        Args:
            evolution_problem: The definition of the evolution problem.

        Returns:
            Evolution result which includes an evolved quantum state.
        """
        return _evolve(evolution_problem, self.num_timesteps, real_time=True)
