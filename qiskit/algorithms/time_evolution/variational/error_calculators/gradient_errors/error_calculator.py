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

"""Abstract class for calculating gradient errors for Variational Quantum Time Evolution."""

from abc import abstractmethod, ABC
from typing import Union, List, Tuple, Any, Dict, Optional

import numpy as np

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class ErrorCalculator(ABC):
    """Abstract class for calculating gradient errors for Variational Quantum Time Evolution."""

    def __init__(
        self,
        h_squared: OperatorBase,
        operator: OperatorBase,
        h_squared_sampler: Optional[CircuitSampler] = None,
        operator_sampler: Optional[CircuitSampler] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        """
        Args:
            h_squared: Squared Hamiltonian.
            operator: Operator composed of a Hamiltonian and a quantum state.
            h_squared_sampler: CircuitSampler for a squared Hamiltonian.
            operator_sampler: CircuitSampler for an operator.
            backend: Optional backend tht enables the use of circuit samplers.
        """
        self._h_squared = h_squared
        self._operator = operator
        self._h_squared_sampler = h_squared_sampler
        self._operator_sampler = operator_sampler
        self._backend = backend

    @classmethod
    def _bind_or_sample_operator(
        cls,
        operator: OperatorBase,
        param_dict: Dict[Parameter, Union[float, complex]],
        operator_circuit_sampler: Optional[CircuitSampler] = None,
    ) -> Union[np.ndarray, int, float, complex]:
        """
        Method accepts an OperatorBase and potentially a respective CircuitSampler. If the
        CircuitSampler is provided, it uses it to sample the operator using a dictionary of
        parameters and values. Otherwise, assign_parameters method is used to bind parameters in
        an operator. A bound operator is then executed and a real part is returned because TODO.
        Args:
            operator: Operator to be bound with parameter values.
            operator_circuit_sampler: CircuitSampler for the operator.
            param_dict: Dictionary which relates parameter values to the parameters in the operator.
        Returns:
            Real part of an operator after binding parameters and executing associated quantum
            circuits.
        """
        if operator_circuit_sampler is not None:
            operator = operator_circuit_sampler.convert(operator, params=param_dict)
        else:
            operator = operator.assign_parameters(param_dict)

        return np.real(operator.eval())

    @abstractmethod
    def _calc_single_step_error(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any],]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQTE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉.
            metric: Fubini-Study Metric.
            param_dict: Dictionary of parameters to be bound.
        Returns:
            Square root of the l2 norm of the error.
        """
        pass

    @abstractmethod
    def _calc_single_step_error_gradient(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> float:
        """
        Evaluate the gradient of the l2 norm for a single time step of VarQRTE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉.
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        """
        pass

    @classmethod
    def _validate_epsilon_squared(cls, eps_squared: float) -> float:
        """
        Sets an epsilon provided to 0 if it is small enough and negative (smaller than 1e-7 in
        absolute value).
        Args:
            eps_squared: Value to be validated.
        Returns:
            Modified (or not) value provided.
        Raises:
            ValueError: If the value provided is too large.
        """
        numerical_instability_error = 1e-7
        if eps_squared < 0:
            if np.abs(eps_squared) < numerical_instability_error:
                eps_squared = 0
            else:
                raise ValueError(
                    "Propagation failed - eps_squared negative and larger in absolute value than "
                    "a potential numerical instability."
                )
        return eps_squared
