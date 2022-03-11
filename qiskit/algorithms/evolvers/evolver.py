# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Interface for quantum time evolution."""

from abc import ABC, abstractmethod

from . import EvolutionProblem
from . import EvolutionResult


class Evolver(ABC):
    """Interface class for quantum time evolution."""

    @abstractmethod
    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Evolves an initial state in the evolution_problem according to a Hamiltonian provided.

        Args:
            evolution_problem: ``EvolutionProblem`` instance that includes definition of an evolution
                problem.

        Returns:
            Evolution result which includes an evolved quantum state.
        """
        raise NotImplementedError()
