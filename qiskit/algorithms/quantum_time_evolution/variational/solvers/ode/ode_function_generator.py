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
from typing import Iterable

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.calculators.distance_energy_calculator import (
    _calculate_distance_energy,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.error_based_ode_function_generator import (
    ErrorBaseOdeFunctionGenerator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)


class OdeFunctionGenerator:
    def __init__(
        self,
        error_calculator,
        param_dict,
        variational_principle: VariationalPrinciple,
        state,
        exact_state,
        h_matrix,
        h_norm,
        grad_circ_sampler,
        metric_circ_sampler,
        nat_grad_circ_sampler,
        regularization=None,
        state_circ_sampler=None,
        backend=None,
        error_based_ode=False,
    ):
        self._error_calculator = error_calculator
        self._param_dict = param_dict
        self._variational_principle = variational_principle
        self._state_circ_sampler = state_circ_sampler
        self._state = state
        self._exact_state = exact_state
        self._h_matrix = h_matrix
        self._h_norm = h_norm
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._nat_grad_circ_sampler = nat_grad_circ_sampler
        self._regularization = regularization
        self._backend = backend
        self._error_based_ode = error_based_ode
        self._linear_solver = VarQteLinearSolver(
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._nat_grad_circ_sampler,
            self._variational_principle._grad_method,
            self._variational_principle._qfi_method,
            self._regularization,
            self._backend,
        )

    def var_qte_ode_function(self, t: float, x: Iterable) -> Iterable:
        error = max(x[-1], 0)
        error = min(error, np.sqrt(2))
        print("previous error", error)
        if self._error_based_ode:
            error_based_ode_fun_gen = ErrorBaseOdeFunctionGenerator(
                self._error_calculator,
                self._param_dict,
                self._variational_principle,
                self._grad_circ_sampler,
                self._metric_circ_sampler,
                self._nat_grad_circ_sampler,
                self._regularization,
                self._backend,
            )
            nat_grad_res, grad_res, metric_res = error_based_ode_fun_gen.error_based_ode_fun()
        else:
            nat_grad_res, grad_res, metric_res = self._linear_solver._solve_sle(
                self._variational_principle, self._param_dict
            )
        # TODO log
        print("Gradient ", grad_res)
        print("Gradient norm", np.linalg.norm(grad_res))

        # Get the residual for McLachlan's Variational Principle
        # self._storage_params_tbd = (t, params, et, resid, f, true_error, true_energy,
        #                             trained_energy, h_squared, dtdt_state, reimgrad)

        (
            et,
            dtdt_state,
            reimgrad,
        ) = self._error_calculator._calc_single_step_error(
            nat_grad_res, grad_res, metric_res
        )

        et = self._inspect_fix_et_negative_part(et)
        (
            f,
            true_error,
            phase_agnostic_true_error,
            true_energy,
            trained_energy,
        ) = _calculate_distance_energy(
            self._state,
            self._exact_state,
            self._h_matrix,
            self._param_dict,
            self._state_circ_sampler,
        )
        h_squared = self._error_calculator._h_squared

        error_bound_grad = self._variational_principle._calc_error_bound(
            error, et, h_squared, self._h_norm, trained_energy, self._variational_principle
        )

        return np.append(nat_grad_res, error_bound_grad)

    def _inspect_fix_et_negative_part(self, et):
        print("returned et", et)
        try:
            if et < 0:
                if np.abs(et) > 1e-4:
                    raise Warning("Non-neglectible negative et observed")
                else:
                    et = 0
            else:
                et = np.sqrt(np.real(et))
        except Exception:
            et = 1000
        print("after try except", et)
        return et
