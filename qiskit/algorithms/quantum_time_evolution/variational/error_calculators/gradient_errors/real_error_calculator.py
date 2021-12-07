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

from typing import Union, List, Dict, Tuple

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.circuit import Parameter


class RealErrorCalculator(ErrorCalculator):
    """Class for calculating gradient errors for Variational Quantum Real Time Evolution."""

    def _calc_single_step_error(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[float, float, float]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQRTE.
        Args:
            ng_res: dω/dt.
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
            param_dict: Dictionary of parameters to be bound.
        Returns:
            L2 norm of the error.
        """
        eps_squared = 0
        h_squared_bound = self._bind_or_sample_operator(
            self._h_squared, self._h_squared_sampler, param_dict
        )
        exp_operator_bound = self._bind_or_sample_operator(
            self._operator, self._operator_sampler, param_dict
        )
        eps_squared += h_squared_bound
        eps_squared -= np.real(exp_operator_bound ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))
        eps_squared += dtdt_state

        # 2Im⟨dtψ(ω)| H | ψ(ω)〉= 2Im dtω⟨dωψ(ω)|H | ψ(ω)
        # 2 missing b.c. of Im
        imgrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared -= imgrad2

        eps_squared = self._validate_epsilon_squared(eps_squared)

        return np.real(eps_squared), dtdt_state, imgrad2 * 0.5

    # TODO some duplication compared to the imaginary counterpart
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
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        Raises:
            Warning if the value of an error has too large imaginary part (larger than 1e-6).
        """
        grad_eps_squared = 0
        # dω_jF_ij^Q
        grad_eps_squared += np.dot(metric, ng_res) + np.dot(
            np.diag(np.diag(metric)), np.power(ng_res, 2)
        )

        # 2Im⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared -= grad_res
        # TODO should this be an exception?
        if np.linalg.norm(np.imag(grad_eps_squared)) > 1e-6:
            raise Warning("Error gradient complex part are not to be neglected.")
        return np.real(grad_eps_squared)
