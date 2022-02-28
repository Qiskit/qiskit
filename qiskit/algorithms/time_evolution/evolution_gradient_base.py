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

"""Base class for quantum time evolution and gradient evolution."""

from abc import ABC, abstractmethod

from qiskit.algorithms.time_evolution.evolution_result import EvolutionResult
from qiskit.algorithms.time_evolution.problems.gradient_evolution_problem import (
    GradientEvolutionProblem,
)
from qiskit.opflow import EvolutionBase


class GradientEvolutionBase(EvolutionBase, ABC):
    """Base class for quantum time evolution and gradient evolution."""

    @abstractmethod
    def gradient(self, gradient_evolution_problem: GradientEvolutionProblem) -> EvolutionResult:
        """
        Performs Quantum Time Evolution of gradient expressions.

        Args:
            gradient_evolution_problem: GradientEvolutionProblem instance that includes definition
                of a gradient evolution problem.

        Returns:
            Evolution result which includes an evolved gradient of quantum state or an observable
                and metadata.
        """
        raise NotImplementedError()
