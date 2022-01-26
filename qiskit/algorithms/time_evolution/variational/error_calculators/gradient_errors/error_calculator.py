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
        allowed_imaginary_part: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
    ):
        """
        Args:
            h_squared: Squared Hamiltonian.
            operator: Operator composed of a Hamiltonian and a quantum state.
            h_squared_sampler: CircuitSampler for a squared Hamiltonian.
            operator_sampler: CircuitSampler for an operator.
            backend: Optional backend tht enables the use of circuit samplers.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                                    imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                                           rounded up to 0 for quantities that are expected to be
                                           non-negative.
        """
        self._h_squared = h_squared
        self._operator = operator
        self._h_squared_sampler = h_squared_sampler
        self._operator_sampler = operator_sampler
        self._backend = backend
        self._allowed_imaginary_part = allowed_imaginary_part
        self._allowed_num_instability_error = allowed_num_instability_error

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
        an operator. A bound operator is then evaluated.
        Args:
            operator: Operator to be bound with parameter values.
            operator_circuit_sampler: CircuitSampler for the operator.
            param_dict: Dictionary which relates parameter values to the parameters in the operator.
        Returns:
            An operator after binding parameters and executing associated quantum
            circuits. If an operator contains unexpected imaginary parts arising from numerical
            instabilities, i.e. parts that are reasonably small, they are removed.
        """
        if operator_circuit_sampler is not None:
            operator = operator_circuit_sampler.convert(operator, params=param_dict)
        else:
            operator = operator.assign_parameters(param_dict)

        return cls._remove_op_imag_part_from_instability(operator)

    @classmethod
    def _remove_op_imag_part_from_instability(
        cls, operator: Union[np.ndarray, int, float, complex]
    ) -> Union[np.ndarray, int, float, complex]:
        return np.real(operator.eval())

    def _remove_float_imag_part_from_instability(
        self, value: Union[np.ndarray, float]
    ) -> Union[np.ndarray, float]:
        """
        Fixes a value or array of values which is/are expected to be non-negative and real in case
        it/they is/are not by a small margin due to numerical instabilities.
        Args:
            value: Value or array of values to be fixed.
        Returns:
            Modified (or not) value provided.
        Raises:
            ValueError: If the value provided cannot be fixed.
        """
        if np.linalg.norm(np.imag(value)) > self._allowed_imaginary_part:
            raise ValueError(
                "Value provided has an unexpected imaginary part that is not to be neglected."
            )
        value = np.real(value)

        value = np.where((value < 0.0) & (value > -allowed_num_instability_error), 0.0, value)
        if value.any() < 0:
            raise ValueError(
                "Propagation failed - value provided is negative and larger in absolute value "
                "than a potential numerical instability."
            )
        return value

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
            L2 norm error with a potential imaginary part arising from numerical instabilities
            removed, norm of the time derivative of a state, time derivative of the expectation
            value ⟨ψ(ω)| H | ψ(ω)〉.
        """
        pass

    @abstractmethod
    def _calc_single_step_error_gradient(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> np.ndarray:
        """
        Evaluate the gradient of the l2 norm error for a single time step of VarQTE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉.
            metric: Fubini-Study Metric.
        Returns:
            Gradient of the l2 norm error with a potential imaginary part arising from numerical
            instabilities removed.
        """
        pass
