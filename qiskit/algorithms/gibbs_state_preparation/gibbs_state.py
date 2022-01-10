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
from typing import List, Optional, Union, Dict

import numpy as np
import numpy.typing

from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, Gradient, CircuitStateFn


class GibbsState:
    """Class representing a quantum state of a Gibbs State along with metadata and gradient
    calculation methods."""

    def __init__(
        self,
        gibbs_state_function: StateFn,
        hamiltonian: Optional[OperatorBase],
        temperature: Optional[float],
        ansatz: Optional[OperatorBase] = None,
        ansatz_params_dict: Optional[Dict[Parameter, Union[complex, float]]] = None,
        ansatz_hamiltonian_gradients: Optional[
            Dict[Parameter, np.typing.NDArray[Union[complex, float]]]
        ] = None,
    ):
        """
        Args:
            gibbs_state_function: Quantum state function of a Gibbs state.
            hamiltonian: Hamiltonian used to build a Gibbs state.
            temperature: Temperature used to build a Gibbs state.
            ansatz: Ansatz that gave rise to a Gibbs state.
            ansatz_params_dict: Dictionary that maps ansatz parameters to values optimal for a
                                Gibbs state.
            ansatz_hamiltonian_gradients: Gradients of ansatz parameters w.r.t. Hamiltonian
                                          parameters. The dictionary key is assumed to be a
                                          Hamiltonian parameter. It gives access to a numpy array
                                          of ansatz parameters gradients w.r.t. that Hamiltonian
                                          parameter. The numpy array is assumed to be ordered in the
                                          same way as parameters are ordered in an ansatz (i.e.
                                          according to ansatz.ordered_parameters). Needed for
                                          Variational Quantum Boltzmann Machine algorithm for
                                          example. Might be obtained from a Variational Quantum
                                          Imaginary Time Evolution algorithm for example.
        """
        self._gibbs_state_function = gibbs_state_function
        self._hamiltonian = hamiltonian
        self._temperature = temperature
        self._ansatz = ansatz
        self._ansatz_params_dict = ansatz_params_dict
        self._ansatz_hamiltonian_gradients = ansatz_hamiltonian_gradients

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
    def ansatz(self):
        """Returns an ansatz that gave rise to a Gibbs state."""
        return self._ansatz

    @property
    def ansatz_params_dict(self):
        """Returns a dictionary that maps ansatz parameters to values that gave rise to a Gibbs
        state."""
        return self._ansatz_params_dict

    def calc_ansatz_gradients(
        self,
        gradient_params: List[Parameter],
        measurement_op: OperatorBase,
        gradient_method: str = "param_shift",
    ) -> numpy.typing.NDArray[Union[complex, float]]:
        """
        Calculates gradients of the visible part of a Gibbs state w.r.t. desired
        gradient_params that parametrize the Gibbs state.
        Args:
            gradient_params: List of parameters present in the gibbs_state_function in
                            respect to which gradients shall be calculated. It is assumed to be
                            ordered in the same way as ansatz.ordered_parameters.
            measurement_op: A projective measurement operator constructed in a way that it measures
                            only qubits of interest in a Gibbs state (e.g. in the Quantum Boltzmann
                            Machine context one might want to measure qubits corresponding to
                            visible nodes only).
            gradient_method: A desired gradient method chosen from the Qiskit Gradient Framework.
        Returns:
            Calculated gradients with respect to each parameter indicated in gradient_params
            with bound parameter values.
        Raises:
            ValueError: If gradient_params not set or if ansatz and ansatz_params_dict are not both
                        provided or if any of the provided gradient parameters is not present in an
                        ansatz that gave rise to a Gibbs state.
        """
        self._validate_arguments(gradient_params)
        op = ~StateFn(measurement_op) @ CircuitStateFn(self._ansatz)

        state_grad = Gradient(grad_method=gradient_method).convert(
            operator=op, params=gradient_params
        )

        return state_grad.assign_parameters(self._ansatz_params_dict).eval()

    def calc_hamiltonian_gradients(
        self,
        gradient_params: List,
        measurement_op: OperatorBase,
        gradient_method: str = "param_shift",
    ) -> Dict[Parameter, Union[complex, float]]:
        """
        Calculates gradients of the visible part of a Gibbs state w.r.t. parameters of a
        Hamiltonian that gave rise to the Gibbs state.
        Args:
            gradient_params: List of parameters present in the gibbs_state_function in
                            respect to which gradients shall be calculated.
            measurement_op: A projective measurement operator constructed in a way that it measures
                            only qubits of interest in a Gibbs state (e.g. in the Quantum Boltzmann
                            Machine context one might want to measure qubits corresponding to
                            visible nodes only).
            gradient_method: A desired gradient method chosen from the Qiskit Gradient Framework.
        Returns:
            Calculated gradients of the visible part of a Gibbs state w.r.t. parameters of a
            Hamiltonian that gave rise to the Gibbs state,
            with bound parameter values.
        Raises:
            ValueError: If gradient_params not set or if ansatz and ansatz_params_dict are not both
                        provided or if any of the provided gradient parameters is not present in an
                        ansatz that gave rise to a Gibbs state.
        """
        self._validate_arguments(gradient_params)
        ansatz_gradients = self.calc_ansatz_gradients(
            gradient_params, measurement_op, gradient_method
        )

        gibbs_state_hamiltonian_gradients = {}
        for hamiltonian_parameter in self._hamiltonian.parameters:
            summed_gradient = np.dot(
                ansatz_gradients, self._ansatz_hamiltonian_gradients[hamiltonian_parameter]
            )

            gibbs_state_hamiltonian_gradients[hamiltonian_parameter] = summed_gradient

        return gibbs_state_hamiltonian_gradients

    def _validate_arguments(self, gradient_params):
        if not gradient_params:
            raise ValueError("Could not calculate gradients because gradient_params not set.")
        if not self._ansatz or not self._ansatz_params_dict:
            raise ValueError(
                "Both ansatz and ansatz_params_dict must be present in the class to compute "
                "gradients."
            )
        for param in gradient_params:
            if param not in self._ansatz.parameters:
                raise ValueError(
                    f"Provided parameter {param} not present in an ansatz that gave rise to the "
                    f"Gibbs state. "
                )
