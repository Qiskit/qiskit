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
"""Class representing a quantum state of a Gibbs State along with metadata and gradient
calculation methods."""
from typing import List, Optional

from qiskit.opflow import StateFn, OperatorBase


class GibbsState:
    """Class representing a quantum state of a Gibbs State along with metadata and gradient
    calculation methods."""

    def __init__(
        self,
        gibbs_state_function: StateFn,
        hamiltonian: Optional[OperatorBase],
        temperature: Optional[float],
        gradients: Optional[List] = None,
        gradient_params: Optional[List] = None,
    ):
        """
        Args:
            gibbs_state_function: Quantum state function of a Gibbs state.
            hamiltonian: Hamiltonian used to build a Gibbs state.
            temperature: Temperature used to build a Gibbs state.
            gradients: Optional values of gradients, obtained, e.g., from a VarQite algorithm.
            gradient_params: Optional list of parameters present in the gibbs_state_function in
                            respect to which gradients shall be calculated.
        """
        self._hamiltonian = hamiltonian
        self._temperature = temperature
        self._gibbs_state_function = gibbs_state_function
        self._gradients = gradients
        self._gradient_params = gradient_params

    @property
    def hamiltonian(self):
        """Returns a hamiltonian."""
        return self._hamiltonian

    @property
    def temperature(self):
        """Returns a temperature."""
        return self._temperature

    @property
    def gibbs_state_function(self):
        """Returns a Gibbs state function."""
        return self._gibbs_state_function

    @property
    def gradients(self):
        """Returns gradients values."""
        return self._gradients

    @gradients.setter
    def gradients(self, gradients: List):
        """Sets gradients values."""
        self._gradients = gradients

    @property
    def gradient_params(self):
        """Returns parameters for gradient calculation."""
        return self._gradient_params

    @gradient_params.setter
    def gradient_params(self, gradient_params: List):
        """Sets parameters for gradient calculation."""
        self._gradient_params = gradient_params

    def calc_gradients(self):
        """Calculates gradients w.r.t. gradient_params, using gradients values."""
        if not self._gradient_params or not self._gradients:
            raise ValueError(
                "Could not calculate gradients because gradient_params and/or gradients not set."
            )
        raise NotImplementedError
