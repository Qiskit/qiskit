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
from abc import abstractmethod
from typing import Union, List, Tuple, Any, Dict, Optional, Iterable

import numpy as np

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class ErrorCalculator:
    def __init__(
        self,
        h_squared: OperatorBase,
        exp_operator: OperatorBase,
        h_squared_sampler: CircuitSampler,
        exp_operator_sampler: CircuitSampler,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        self._h_squared = h_squared
        self._exp_operator = exp_operator
        self._h_squared_sampler = h_squared_sampler
        self._exp_operator_sampler = exp_operator_sampler
        self._backend = backend

    def _bind_or_sample_operator(
        self,
        operator: OperatorBase,
        operator_circuit_sampler: CircuitSampler,
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> OperatorBase:
        """
        Args:
            operator: Operator to be bound with parameter values.
            operator_circuit_sampler: CircuitSampler for the operator.
            param_dict: Dictionary which relates parameter values to the parameters in the operator.
        Returns:
            Operator with all parameters bound.
        """
        # ⟨ψ(ω)|H^2|ψ(ω)〉
        if operator_circuit_sampler:
            operator = operator_circuit_sampler.convert(operator, params=param_dict)
        else:
            operator = operator.assign_parameters(param_dict)
        operator = np.real(operator.eval())
        return operator

    @abstractmethod
    def _calc_single_step_error(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any],]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.
        Args:
            ng_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉.
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        """
        pass

    @abstractmethod
    def _calc_single_step_error_gradient(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> float:
        """
        Evaluate the gradient of the l2 norm for a single time step of VarQRTE.
        Args:
            ng_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉.
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        """
        pass

    def _inner_prod(self, x: Iterable, y: Iterable) -> Union[np.ndarray, np.complex, np.float]:
        """
        Compute the inner product of two vectors.
        Args:
            x: vector.
            y: vector.
        Returns:
            Inner product of x and y.
        """
        return np.matmul(np.conj(np.transpose(x)), y)

    def _validate_epsilon_squared(self, eps_squared: float) -> float:
        """
        Sets an epsilon provided to 0 if it is small enough and negative (smaller than 1e-3 in
        absolute value).
        Args:
            eps_squared: Value to be validated.
        Returns:
            Modified (or not) value provided.
        Raises:
            Warning if the value provided is too large.
        """
        if eps_squared < 0:
            if np.abs(eps_squared) < 1e-3:
                eps_squared = 0
            else:
                # TODO should this be an excpetion?
                raise Warning("Propagation failed")
        return eps_squared
