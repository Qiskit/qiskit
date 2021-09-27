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
from typing import Union, Dict
import math

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.time_step_errors.time_step_error_calculator import (
    _calculate_max_bures,
    _calculate_energy_factor,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    OperatorBase,
)


class ImaginaryVariationalPrinciple(VariationalPrinciple):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
    ):
        super().__init__(
            qfi_method,
            grad_method,
        )

    @abstractmethod
    def _get_raw_metric_tensor(
        self,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        pass

    @abstractmethod
    def _get_raw_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        pass

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(
        raw_metric_tensor: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(
        raw_evolution_grad: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        pass

    def _calc_nat_grad(
        self,
        raw_operator: OperatorBase,
        param_dict: Dict[Parameter, Union[float, complex]],
        regularization: str,
    ) -> OperatorBase:
        return super()._calc_nat_grad(-raw_operator, param_dict, regularization)

    def _calc_error_bound(
        self,
        error: float,
        et: float,
        h_squared_expectation: float,
        h_norm: float,
        trained_energy: float,
    ) -> float:
        # TODO stack dt params with the gradient for the error update
        error_store = max(error, 0)
        error_store = min(error_store, np.sqrt(2))
        error_bound_grad = self._get_error_grad(
            delta_t=1e-4,
            eps_t=error_store,
            grad_err=et,
            energy=trained_energy,
            h_squared_expectation=h_squared_expectation,
            h_norm=h_norm,
            stddev=np.sqrt(h_squared_expectation - trained_energy ** 2),
        )

        print("returned grad", error_bound_grad)
        return error_bound_grad

    def _get_error_grad(
        self,
        delta_t: float,
        eps_t: float,
        grad_err: float,
        energy: float,
        h_squared_expectation: float,
        h_norm: float,
        stddev: float,
    ) -> float:
        eps_t = max(eps_t, 0)
        eps_t = min(eps_t, np.sqrt(2))
        eps_t_next = self._get_error_term(
            delta_t, eps_t, grad_err, energy, h_squared_expectation, h_norm, stddev
        )
        eps_t_next = max(eps_t_next, 0)
        eps_t_next = min(eps_t_next, np.sqrt(2))
        grad = (eps_t_next - eps_t) / delta_t
        if grad > 10:
            print("huge grad", grad)
        if grad < 0:
            print("negative grad", grad)
        return grad

    def _get_error_term(
        self,
        d_t,
        eps_t,
        grad_err,
        energy: float,
        h_squared_expectation: float,
        h_norm: float,
        stddev: float,
    ) -> float:
        """
        Compute the error term for a given time step and a point in the simulation time
        Args:
            d_t: time step
            j: jth step in VarQITE
        Returns: eps_j(delta_t)
        """
        if eps_t < 0:
            eps_t = 0
            print("Warn eps_t neg. clipped to 0")
        if eps_t == 0 and grad_err == 0:
            eps_t_next = 0
            energy_factor = 0
            y = 0
        else:
            energy_factor = _calculate_energy_factor(eps_t, energy, stddev, h_norm)
            # max B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
            y = _calculate_max_bures(eps_t, energy, energy_factor, h_squared_expectation, d_t)
            # eps_t*sqrt(var) + eps_t^2/2 * |E_t - ||H||_infty |
            # energy_factor = (2 * eps_t * stddev +
            #                  eps_t ** 2 / 2 * self._h_norm)
            eps_t_next = y + d_t * grad_err + d_t * energy_factor

        if math.isnan(energy_factor):
            print("nan")
        # TODO save to file
        return eps_t_next
