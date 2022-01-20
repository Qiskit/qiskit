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
from typing import Union, Dict, Optional

from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase


class GibbsStateBuilder:
    """Interface for building Gibbs States."""

    BOLTZMANN_CONSTANT = 1.38064852e-2

    @abstractmethod
    def build(
        self,
        problem_hamiltonian: OperatorBase,
        temperature: float,
        problem_hamiltonian_param_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
    ) -> GibbsState:
        """
        Creates a Gibbs state from given parameters.
        Args:
            problem_hamiltonian: Hamiltonian that defines a desired Gibbs state.
            temperature: Temperature of a desired Gibbs state.
            problem_hamiltonian_param_dict: If a problem Hamiltonian is parametrized, a dictionary
                                            that maps all of its parameters to certain values.
        Returns: GibbsState object that includes a relevant quantum state functions as well as
                metadata.
        """
        raise NotImplementedError
