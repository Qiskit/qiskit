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

"""Class for calculating gradient errors for Variational Quantum Real Time Evolution."""

from typing import Union, List, Dict, Tuple, Any

import numpy as np

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.circuit import Parameter


class RealErrorCalculator(ErrorCalculator):
    """Class for calculating gradient errors for Variational Quantum Real Time Evolution."""

    def _calc_single_step_error(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[float, Union[np.ndarray, complex, float], Union[Union[complex, float], Any],]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQRTE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
            param_dict: Dictionary of parameters to be bound.
        Returns:
            L2 norm error with a potential imaginary part arising from numerical instabilities
            removed, norm of the time derivative of a state, time derivative of the expectation
            value ⟨ψ(ω)| H | ψ(ω)〉.
        """
        error_squared = 0.0
        h_squared_bound = self._bind_or_sample_operator(
            self._h_squared, param_dict, self._h_squared_sampler
        )
        exp_operator_bound = self._bind_or_sample_operator(
            self._operator, param_dict, self._operator_sampler
        )
        error_squared += h_squared_bound
        error_squared -= np.real(exp_operator_bound ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        state_time_derivative_norm = np.conj(nat_grad_res).T.dot(np.dot(metric, nat_grad_res))
        error_squared += state_time_derivative_norm

        # 2Im⟨dtψ(ω)| H | ψ(ω)〉= 2Im dtω⟨dωψ(ω)|H | ψ(ω)
        # 2 missing b.c. of Im
        expected_val_time_derivative = np.conj(grad_res).T.dot(nat_grad_res)
        error_squared -= expected_val_time_derivative

        return (
            self._remove_float_imag_part_from_instability(error_squared),
            state_time_derivative_norm,
            expected_val_time_derivative * 0.5,
        )

    # TODO some duplication compared to the imaginary counterpart
    def _calc_single_step_error_gradient(
        self,
        nat_grad_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> np.ndarray:
        """
        Evaluate the gradient of the l2 norm error for a single time step of VarQRTE.
        Args:
            nat_grad_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
        Returns:
            Gradient of the l2 norm error with a potential imaginary part arising from numerical
            instabilities removed.
        """
        gradient_error_squared = 0.0
        # dω_jF_ij^Q
        gradient_error_squared += np.dot(metric, nat_grad_res) + np.dot(
            np.diag(np.diag(metric)), np.power(nat_grad_res, 2)
        )

        # 2Im⟨dωψ(ω)|H | ψ(ω)〉
        gradient_error_squared -= grad_res
        return self._remove_float_imag_part_from_instability(gradient_error_squared)
