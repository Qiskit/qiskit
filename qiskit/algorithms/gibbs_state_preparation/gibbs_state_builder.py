# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Interface for building Gibbs States."""
from abc import abstractmethod

from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.opflow import OperatorBase


class GibbsStateBuilder:
    """Interface for building Gibbs States."""

    BOLTZMANN_CONSTANT = 1.38064852e-2

    @abstractmethod
    def build(self, problem_hamiltonian: OperatorBase, temperature: float) -> GibbsState:
        raise NotImplementedError
