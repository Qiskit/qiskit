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
from typing import Union, List, Tuple, Any, Dict, Optional

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
        param_dict: Dict[Parameter, Union[float, complex]],
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        self._h_squared = self._bind_or_sample_operator(
            h_squared, h_squared_sampler, param_dict
        )
        self._exp_operator = self._bind_or_sample_operator(
            exp_operator, exp_operator_sampler, param_dict
        )
        self._param_dict = param_dict
        self._backend = backend

    def _bind_or_sample_operator(
        self,
        operator: OperatorBase,
        operator_circuit_sampler: CircuitSampler,
        param_dict: Dict[Parameter, float],
    ) -> OperatorBase:
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
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any],]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.
        Args:
            ng_res: dω/dt
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric
        Returns:
            square root of the l2 norm of the error
        """
        raise NotImplementedError

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
            ng_res: dω/dt
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric
        Returns:
            square root of the l2 norm of the error
        """
        raise NotImplementedError

    def _validate_epsilon_squared(self, eps_squared: float) -> float:
        if eps_squared < 0:
            if np.abs(eps_squared) < 1e-3:
                eps_squared = 0
            else:
                raise Warning("Propagation failed")
        return eps_squared
