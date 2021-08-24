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
from typing import Union, List, Tuple, Any

import numpy as np
from qiskit.algorithms.quantum_time_evolution.variational.calculators.distance_energy_calculator import (
    _inner_prod,
)
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.residual_errors.error_calculator import (
    ErrorCalculator,
)


# TODO used by variational principle
class ImaginaryErrorCalculator(ErrorCalculator):
    def __init__(
        self,
        h_squared,
        exp_operator,
        h_squared_sampler,
        exp_operator_sampler,
        param_dict,
        backend=None,
    ):
        super().__init__(
            h_squared, exp_operator, h_squared_sampler, exp_operator_sampler, param_dict, backend
        )

    def _calc_single_step_error(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any]]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.
        Args:
            ng_res: dω/dt
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric
        Returns:
            square root of the l2 norm of the error
        """
        eps_squared = 0

        eps_squared += np.real(self._h_squared)
        eps_squared -= np.real(self._exp_operator ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = _inner_prod(ng_res, np.dot(metric, ng_res))
        eps_squared += dtdt_state

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        regrad2 = _inner_prod(grad_res, ng_res)
        eps_squared += regrad2

        eps_squared = self._validate_epsilon_squared(eps_squared)

        return np.real(eps_squared), dtdt_state, regrad2 * 0.5

    def _calc_single_step_error_gradient(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> float:

        """
        Evaluate the gradient of the l2 norm for a single time step of VarQITE.
        Args:
            ng_res: dω/dt
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric
        Returns:
            square root of the l2 norm of the error
        """
        grad_eps_squared = 0
        # dω_jF_ij^Q
        grad_eps_squared += np.dot(metric, ng_res) + np.dot(
            np.diag(np.diag(metric)), np.power(ng_res, 2)
        )
        # 2Re⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared += grad_res
        return np.real(grad_eps_squared)
