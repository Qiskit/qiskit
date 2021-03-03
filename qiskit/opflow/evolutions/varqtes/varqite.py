# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Imaginary Time Evolution"""

from typing import List, Union, Dict, Iterable, Tuple, Any, Optional
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.integrate import ode, OdeSolver, solve_ivp

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, ComposedOp, OperatorBase

from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve


class VarQITE(VarQTE):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191

    Algorithms that use McLachlans variational principle to approximate the imaginary time
    evolution for a given Hermitian operator (Hamiltonian) and quantum state.
    """

    def convert(self,
                operator: ListOp) -> StateFn:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given operator

        Args:
            operator:
                ⟨ψ(ω)|H|ψ(ω)〉
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQITE)
                The coefficient of the operator (operator.coeff) determines the evolution time.

                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.

        Returns:
            StateFn (parameters are bound) which represents an approximation to the respective
            time evolution.

        """
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError('Please provide the respective Ansatz as a CircuitStateFn.')
        elif not isinstance(operator, ComposedOp) and not all(isinstance(op, CircuitStateFn) for \
                op in operator.oplist):
            raise TypeError('Please provide the operator either as ComposedOp or as ListOp of a '
                            'CircuitStateFn potentially with a combo function.')

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        self._operator = operator / operator.coeff # Remove the time from the operator
        self._operator_eval = operator / operator.coeff

        # if self._snapshot_dir is not None:
        #     # Compute the norm of the Hamiltonian
        #     h_norm = np.linalg.norm(self._h_matrix, ord=2)

        # Step size
        dt = np.abs(operator.coeff)*np.sign(operator.coeff) / self._num_time_steps

        self._init_grad_objects()
        # Run ODE Solver
        parameter_values = self._run_ode_solver(dt * self._num_time_steps)
        # return evolved
        return self._state.assign_parameters(dict(zip(self._parameters,
                                                      parameter_values)))

        # Initialize error bound
        # error_bound_en_diff = 0
        # error_bound_l2 = 0

        # # Zip the parameter values to the parameter objects
        # param_dict = dict(zip(self._parameters, self._parameter_values))
        #
        # et_list = []
        # h_squared_factor_list = []
        # trained_energy_factor_list = []
        # stddev_factor_list = []
        #
        # f, true_error, true_energy, trained_energy = self._distance_energy(0, param_dict)
        # for j in range(self._num_time_steps):
        #     # Get the natural gradient - time derivative of the variational parameters - and
        #     # the gradient w.r.t. H and the QFI/4.
        #     nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
        #
        #     if self._snapshot_dir is not None:
        #         # Get the residual for McLachlan's Variational Principle
        #         resid = np.linalg.norm(np.matmul(metric_res, nat_grad_result) + 0.5 * grad_res)
        #         print('Residual norm', resid)
        #
        #         # Get the error for the current step
        #         et, h_squared, dtdt_state, regrad, exp, = self._error_t(
        #             nat_grad_result, grad_res, metric_res)
        #         if et < 0 and np.abs(et) > 1e-4:
        #             raise Warning('Non-neglectible negative et observed')
        #         else:
        #             et = np.sqrt(np.real(et))
        #         # h_norm_factor: (1 + 2 \delta_t | | H | |) ^ {T - t}
        #         # h_norm_factor = (1 + 2 * dt * h_norm) ** (operator.coeff - j - 1)
        #         # h_squared_factor: (1 + 2\delta_t \sqrt{| < trained | H ^ 2 | trained > |})
        #         # h_squared_factor = (1 + 2 * dt * np.sqrt(h_squared))
        #         # h_squared_factor_list.append(h_squared_factor)
        #         # trained_energy_factor: (1 + 2 \delta_t | < trained | H | trained > |)
        #         # trained_energy_factor = (1 + 2 * dt * np.abs(exp))
        #         # trained_energy_factor_list.append(trained_energy_factor)
        #         et_list.append(et)
        #         stddev_factor_list.append(1 + 2 * dt * np.sqrt(h_squared - exp**2))
        #
        #         if np.imag(et) > 1e-5:
        #             raise Warning(
        #                 'The error of this step is imaginary. Thus, the SLE underlying '
        #                 'McLachlan was not solved well. The residuum of the SLE is ', resid,
        #                 '. Please use a regularized least square method to resolve this issue.')
        #
        #         print('et', et)
        #
        #         # Store the current status
        #         self._store_params(j * dt, self._parameter_values, None, et,
        #                            resid, f, true_error, None,
        #                            None, true_energy,
        #                            trained_energy, h_norm, h_squared, dtdt_state, regrad)
        #
        #         # error_bound_l2 += dt / 2 * np.exp(2 * np.abs(operator.coeff) * h_norm) * \
        #         #                        (et + et_prev)
        #         # et_prev = et
        #
        #         # Propagate the Ansatz parameters step by step using explicit Euler
        #         # if self._backend is not None:
        #         #     state_for_grad = self._state_circ_sampler.convert(self._state,
        #         #                                                       params=param_dict)[0]
        #         # else:
        #         #     state_for_grad = self._state.assign_parameters(param_dict)
        #         # self._exact_euler_state += dt * \
        #         #                            self._exact_grad_state(
        #         #                                state_for_grad.eval().primitive.data)
        #
        #     # Propagate the Ansatz parameters step by step using explicit Euler
        #     # Subtract is correct either
        #     # omega_new = omega - A^(-1)Cdt or
        #     # omega_new = omega + A^(-1)((-1)*C)dt
        #
        #     self._parameter_values = list(np.add(self._parameter_values, dt * np.real(
        #                                   nat_grad_result)))
        #     # Zip the parameter values to the parameter objects
        #     param_dict = dict(zip(self._parameters, self._parameter_values))
        #     if self._snapshot_dir:
        #         # If initial parameter values were set compute the fidelity, the error between the
        #         # prepared and the target state, the energy w.r.t. the target state and the energy
        #         # w.r.t. the prepared state
        #         f, true_error, true_energy, trained_energy = self._distance_energy((j + 1) * dt,
        #                                                                            param_dict)
        #         #
        #         # error_bound_en_diff += dt * (et + np.sqrt(np.linalg.norm(trained_energy -
        #         #                                                          true_energy)))
        #         #
        #         # print('Error bound based on exact energy difference', np.round(error_bound_en_diff,
        #         #                                                                6), 'after',
        #         #       (j + 1) * dt)
        #         # print('Error bound based on ||H||_2 and integration', np.round(error_bound_l2, 6),
        #         #       'after', (j + 1) * dt)
        #
        # # Store the current status
        # if self._snapshot_dir:
        #     # error_bound_h_squared = 0
        #     # error_bound_trained_energy = 0
        #     error_bound_stddev = 0
        #     for l, grad_error in enumerate(et_list):
        #         # error_bound_h_squared += grad_error * dt * np.prod(h_squared_factor_list[l:])
        #         # error_bound_trained_energy += grad_error * dt * \
        #         #                               np.prod(trained_energy_factor_list[l:])
        #         error_bound_stddev += grad_error * dt * np.prod(stddev_factor_list[l:])
        #         # print('Error bound based on sqrt(<H^2>)', np.round(error_bound_h_squared, 6),
        #         #       'after', (l + 1) * dt)
        #         # print('Error bound based on <H>', np.round(error_bound_trained_energy, 6),
        #         #       'after', (l + 1) * dt)
        #         print('Error bound based on stdev', np.round(error_bound_stddev, 6),
        #               'after', (l + 1) * dt)
        #
        #     self._store_params((j+1) * dt, self._parameter_values, error_bound_stddev, None,
        #                        None, f, true_error, None,
        #                        None, true_energy,
        #                        trained_energy, h_norm, None, None, None)

        # Return variationally evolved operator
        # return self._state.assign_parameters(param_dict)

    def _error_t(self,
                 param_values: Union[List, np.ndarray],
         ng_res: Union[List, np.ndarray],
         grad_res: Union[List, np.ndarray],
         metric: Union[List, np.ndarray]) -> Tuple[
         int, Union[np.ndarray, int, float, complex], Union[np.ndarray, complex, float], Union[
            Union[complex, float], Any], Union[np.ndarray, int, float, complex]]:

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
        param_dict = dict(zip(self._parameters, param_values))

        # ⟨ψ(ω)|H^2|ψ(ω)〉Hermitian
        if self._backend is not None:
            h_squared = self._h_squared_circ_sampler.convert(self._h_squared,
                                                             params=param_dict)[0]
        else:
            h_squared = self._h_squared.assign_parameters(param_dict)
        h_squared = np.real(h_squared.eval())

        # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        if self._backend is not None:
            exp = self._operator_circ_sampler.convert(self._operator_eval,
                                                      params=param_dict)[0]
        else:
            exp = self._operator_eval.assign_parameters(param_dict)
        exp = np.real(exp.eval())
        eps_squared += np.real(h_squared)
        eps_squared -= np.real(exp ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))

        eps_squared += dtdt_state

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        regrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared += regrad2
        if eps_squared < 0:
            if np.abs(eps_squared) < 1e-3:
                eps_squared = 0
            else:
                raise Warning('Propagation failed')

        return np.real(eps_squared), h_squared, dtdt_state, regrad2 * 0.5, exp

    def _grad_error_t(self,
                      ng_res: Union[List, np.ndarray],
                      grad_res: Union[List, np.ndarray],
                      metric: Union[List, np.ndarray]) -> float:

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
        grad_eps_squared += np.dot(metric, ng_res) + np.dot(np.diag(np.diag(metric)),
                                                            np.power(ng_res, 2))
        # 2Re⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared += grad_res
        return np.real(grad_eps_squared)

    def _get_error_bound(self,
                         gradient_errors: List,
                         time_steps: List,
                         stddevs: List,
                         use_integral_approx: bool = True,
                         imag_reverse_bound: bool = True,
                         H: Optional[Union[List, np.ndarray]] = None) -> List:

        if not len(gradient_errors) == len(time_steps) + 1:
            print(gradient_errors)
            print()
            raise Warning('The number of the gradient errors is incompatible with the number of '
                          'the time steps.')
        gradient_error_factors = []
        for j, dt in enumerate(time_steps):
            if use_integral_approx:
                stddev_factor = 0
            else:
                stddev_factor = 1
            for k in range(j, len(time_steps)):
                if use_integral_approx:
                    stddev_factor += (stddevs[k]+stddevs[k+1]) * 0.5 * time_steps[k]
                else:
                    stddev_factor *= (1 + 2 * time_steps[k] * stddevs[k])
            gradient_error_factors.append(stddev_factor)
        gradient_error_factors.append(0)

        print('Error factors ', gradient_error_factors)
        print('Gradient Errors', gradient_errors)

        e_bounds = [0]
        for j, dt in enumerate(time_steps):
            if use_integral_approx:
                e_bounds.append(e_bounds[j]+(gradient_errors[j] * np.exp(2*gradient_error_factors[j])
                                           + gradient_errors[j + 1] *
                                           np.exp(2*gradient_error_factors[j + 1])
                                           ) * 0.5 * dt)
            else:
                e_bounds.append(e_bounds[j] + (gradient_errors[j] * gradient_error_factors[j]) * dt)
        print('Error bounds ', e_bounds)

        if imag_reverse_bound:
            if H is None:
                raise Warning('Please support the respective Hamiltonian.')
            eigvals = sorted(list(set(np.linalg.eigvals(H))))
            e0 = eigvals[0]
            e1 = eigvals[1]
            reverse_bounds = np.zeros(len(e_bounds))
            reverse_bounds[-1] = stddevs[-1] / (e1 - e0)
            for j, dt in enumerate(time_steps):
                if use_integral_approx:
                    reverse_bounds[-j] = reverse_bounds[-(j+1)] - (gradient_errors[-j] * np.exp(2 *
                                                                       gradient_error_factors[-j])
                                      + gradient_errors[-(j + 1)] *
                                      np.exp(2 * gradient_error_factors[-(j + 1)])) * 0.5 * \
                                         time_steps[-j]
                else:
                    reverse_bounds[-j] = reverse_bounds[-(j + 1)] - \
                                        (gradient_errors[-j] * gradient_error_factors[-j]) * \
                                         time_steps[-j]
            return e_bounds, reverse_bounds
        return e_bounds

    def _exact_state(self,
                     time: Union[float, complex]) -> Iterable:
        """

        Args:
            time: current time

        Returns:
            Exactly evolved state for the respective time

        """

        # Evolve with exponential operator
        target_state = np.dot(expm(-1 * self._h_matrix * time), self._init_state)
        # Normalization
        target_state /= np.sqrt(self._inner_prod(target_state, target_state))
        return target_state

    def _exact_grad_state(self,
                          state: Iterable) -> Iterable:
        """
        Return the gradient of the given state
        (E_t - H ) |state>

        Args:
            state: State for which the exact gradient shall be evaluated

        Returns:
            Exact gradient of the given state

        """

        energy_t = self._inner_prod(state, np.matmul(self._h_matrix, state))
        return np.matmul(np.subtract(energy_t*np.eye(len(self._h_matrix)), self._h_matrix), state)

