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

"""Class for calculating gradient errors for Variational Quantum Imaginary Time Evolution."""

from typing import Union, List, Tuple, Any, Dict

import numpy as np

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.circuit import Parameter


class ImaginaryErrorCalculator(ErrorCalculator):
    """Class for calculating gradient errors for Variational Quantum Imaginary Time Evolution."""

    def _calc_single_step_error(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any]]:
        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
            param_dict: Dictionary of parameters to be bound.
        Returns:
            Real part of a squared gradient error, norm of the time derivative of a state,
            time derivative of the expectation value ⟨ψ(ω)| H | ψ(ω)〉.
        """
        gradient_error_squared = 0
        h_squared_bound = self._bind_or_sample_operator(
            self._h_squared, param_dict, self._h_squared_sampler
        )
        exp_operator_bound = self._bind_or_sample_operator(
            self._operator, param_dict, self._operator_sampler
        )
        gradient_error_squared += np.real(h_squared_bound)
        gradient_error_squared -= np.real(exp_operator_bound ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        state_time_derivative_norm = np.conj(nat_grad_res).T.dot(np.dot(metric, nat_grad_res))
        gradient_error_squared += state_time_derivative_norm

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        expected_val_time_derivative = np.conj(grad_res).T.dot(nat_grad_res)
        gradient_error_squared += expected_val_time_derivative

        gradient_error_squared = self._validate_epsilon_squared(gradient_error_squared)

        return (
            np.real(gradient_error_squared),
            state_time_derivative_norm,
            expected_val_time_derivative * 0.5,
        )

    def _calc_single_step_error_gradient(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> float:
        """
        Evaluate the gradient of the l2 norm for a single time step of VarQITE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
        Returns:
            Real part of a squared gradient error.
        """
        gradient_error_squared = 0
        # dω_jF_ij^Q
        gradient_error_squared += np.dot(metric, nat_grad_res) + np.dot(
            np.diag(np.diag(metric)), np.power(nat_grad_res, 2)
        )
        # 2Re⟨dωψ(ω)|H | ψ(ω)〉
        gradient_error_squared += grad_res
        return np.real(gradient_error_squared)
