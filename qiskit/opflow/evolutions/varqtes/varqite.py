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

from typing import List, Union, Dict
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.integrate import ode, OdeSolver, solve_ivp

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, ComposedOp, OperatorBase, \
    CircuitSampler

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
        # print('oplist', self._operator.oplist)

        if self._get_error:
            # Compute the norm of the Hamiltonian
            h_norm = np.linalg.norm(self._operator.oplist[0].primitive.to_matrix(massive=True),
                                    ord=2)

        # Step size
        dt = np.abs(operator.coeff)*np.sign(operator.coeff) / self._num_time_steps

        # ODE Solver

        if self._ode_solver is not None:
            self._parameter_values = self._run_ode_solver(dt * self._num_time_steps)
            return self._operator[-1].assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values)))

        # Initialize error bound
        error_bound_en_diff = 0
        error_bound_l2 = 0

        # Zip the parameter values to the parameter objects
        param_dict = dict(zip(self._parameters, self._parameter_values))

        et_list = []
        h_squared_factor_list = []
        trained_energy_factor_list = []
        stddev_factor_list = []

        f, true_error, true_energy, trained_energy = self._distance_energy(0, param_dict)
        for j in range(self._num_time_steps):
            # Get the natural gradient - time derivative of the variational parameters - and
            # the gradient w.r.t. H and the QFI/4.
            nat_grad_result, grad_res, metric_res = self._propagate(param_dict)

            if self._get_error:
                # Get the residual for McLachlan's Variational Principle
                resid = np.linalg.norm(np.matmul(metric_res, nat_grad_result) + 0.5 * grad_res)
                print('Residual norm', resid)

                # Get the error for the current step
                et, h_squared, exp, dtdt_state, regrad = self._error_t(
                    self._operator, nat_grad_result, grad_res, metric_res)
                if et < 0 and np.abs(et) > 1e-4:
                    raise Warning('Non-neglectible negative et observed')
                else:
                    et = np.sqrt(np.real(et))
                # h_norm_factor: (1 + 2 \delta_t | | H | |) ^ {T - t}
                h_norm_factor = (1 + 2 * dt * h_norm) ** (operator.coeff - j - 1)
                # h_squared_factor: (1 + 2\delta_t \sqrt{| < trained | H ^ 2 | trained > |})
                h_squared_factor = (1 + 2 * dt * np.sqrt(h_squared))
                h_squared_factor_list.append(h_squared_factor)
                # trained_energy_factor: (1 + 2 \delta_t | < trained | H | trained > |)
                trained_energy_factor = (1 + 2 * dt * np.abs(exp))
                trained_energy_factor_list.append(trained_energy_factor)
                et_list.append(et)
                stddev_factor_list.append(1 + 2 * dt * np.sqrt(h_squared - exp**2))

                if np.imag(et) > 1e-5:
                    raise Warning(
                        'The error of this step is imaginary. Thus, the SLE underlying '
                        'McLachlan was not solved well. The residuum of the SLE is ', resid,
                        '. Please use a regularized least square method to resolve this issue.')

                print('et', et)

                # Store the current status
                if self._snapshot_dir:
                    if self._get_error:
                        if self._init_parameter_values is None:
                            self._store_params(j * dt, self._parameter_values,
                                               error_bound_en_diff, et,
                                               resid)
                        else:
                            # h_norm_factor: (1 + 2 \delta_t | | H | |) ^ {T - t}
                            # h_squared_factor: (1 + 2\delta_t \sqrt{| < trained | H ^ 2 |
                            # trained > |})
                            # trained_energy_factor: (1 + 2 \delta_t | < trained | H |
                            # trained > |)
                            if self._get_h_terms:
                                self._store_params(j * dt, self._parameter_values,
                                                   error_bound_l2, et,
                                                   resid, f, true_error, true_energy,
                                                   trained_energy,
                                                   h_norm, h_squared, dtdt_state, regrad,
                                                   h_norm_factor, h_squared_factor,
                                                   trained_energy_factor)
                            else:
                                self._store_params(j * dt, self._parameter_values,
                                                   error_bound_en_diff, et,
                                                   resid, f, true_error, true_energy,
                                                   trained_energy)
                    else:

                        if self._init_parameter_values is None:
                            self._store_params(j * dt, self._parameter_values, f, true_error,
                                               true_energy, trained_energy)
                        else:
                            self._store_params(j * dt, self._parameter_values)

                if j == 0:
                    et_prev = 0
                    sqrt_h_prev_square = 0
                error_bound_l2 += dt / 2 * np.exp(2 * np.abs(operator.coeff) * h_norm) * \
                                       (et + et_prev)
                et_prev = et

            # Propagate the Ansatz parameters step by step using explicit Euler
            # Subtract is correct either
            # omega_new = omega - A^(-1)Cdt or
            # omega_new = omega + A^(-1)((-1)*C)dt

            self._parameter_values = list(np.add(self._parameter_values, dt * np.real(
                                          nat_grad_result)))
            print('Params', self._parameter_values)



            # Zip the parameter values to the parameter objects
            param_dict = dict(zip(self._parameters, self._parameter_values))
            if self._init_parameter_values is not None:
                # If initial parameter values were set compute the fidelity, the error between the
                # prepared and the target state, the energy w.r.t. the target state and the energy
                # w.r.t. the prepared state
                f, true_error, true_energy, trained_energy = \
                    self._distance_energy((j + 1) * dt, param_dict)
                if j == 0:
                    trained_energy_0 = trained_energy

                error_bound_en_diff += dt * (et + np.sqrt(np.linalg.norm(trained_energy -
                                                                         true_energy)))

            print('Error bound based on exact energy difference', np.round(error_bound_en_diff,
                                                                           6), 'after',
                  (j + 1) * dt)
            print('Error bound based on ||H||_2 and integration', np.round(error_bound_l2, 6),
                  'after', (j + 1) * dt)


        # Store the current status
        if self._snapshot_dir:
            if self._get_error:
                if self._init_parameter_values is None:
                    self._store_params((j + 1) * dt, self._parameter_values,
                                       error_bound_l2)
                else:
                    # h_norm_factor: (1 + 2 \delta_t | | H | |) ^ {T - t}
                    # h_squared_factor: (1 + 2\delta_t \sqrt{| < trained | H ^ 2 | trained > |})
                    # trained_energy_factor: (1 + 2 \delta_t | < trained | H | trained > |)
                    if self._get_h_terms:
                        self._store_params((j + 1) * dt, self._parameter_values,
                                           error_bound_l2, fidelity_to_target=f,
                                           true_error=true_error,
                                           target_energy=true_energy,
                                           trained_energy= trained_energy,
                                           h_norm=h_norm)
                    else:
                        self._store_params((j + 1) * dt, self._parameter_values,
                                           error_bound_l2, fidelity_to_target=f,
                                           true_error=true_error,
                                           target_energy=true_energy,
                                           trained_energy= trained_energy)
            else:

                if self._init_parameter_values is None:
                    self._store_params((j + 1) * dt, self._parameter_values,
                                       error_bound_l2, fidelity_to_target=f,
                                       true_error=true_error,
                                       target_energy=true_energy,
                                       trained_energy=trained_energy)
                else:
                    self._store_params((j + 1) * dt, self._parameter_values)
                    self._store_params((j + 1) * dt, self._parameter_values)

        error_bound_h_squared = 0
        error_bound_trained_energy = 0
        error_bound_stddev = 0
        for l, grad_error in enumerate(et_list):
            error_bound_h_squared += grad_error * dt * np.prod(h_squared_factor_list[l:])
            error_bound_trained_energy += grad_error * dt * \
                                          np.prod(trained_energy_factor_list[l:])
            error_bound_stddev += grad_error * dt * np.prod(stddev_factor_list[l:])
            print('Error bound based on sqrt(<H^2>)', np.round(error_bound_h_squared, 6),
                  'after', (l + 1) * dt)
            print('Error bound based on <H>', np.round(error_bound_trained_energy, 6),
                  'after', (l + 1) * dt)
            print('Error bound based on stdev', np.round(error_bound_stddev, 6),
                  'after', (l + 1) * dt)

        # Return variationally evolved operator
        return self._operator[-1].assign_parameters(dict(zip(self._parameters, self._ode_solver.y)))

    def _error_t(self,
         operator: OperatorBase,
         ng_res: Union[List, np.ndarray],
         grad_res: Union[List, np.ndarray],
         metric: Union[List, np.ndarray]) -> float:

        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.

        Args:
            operator: ⟨ψ(ω)|H|ψ(ω)〉
            ng_res: dω/dt
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric

        Returns:
            square root of the l2 norm of the error
        """
        if not isinstance(operator, ComposedOp):
            raise TypeError('Currently this error can only be computed for operators given as '
                            'ComposedOps')
        eps_squared = 0

        # ⟨ψ(ω)|H^2|ψ(ω)〉Hermitian
        h_squared = ComposedOp([~StateFn(operator.oplist[0].primitive ** 2) *
                                operator.oplist[0].coeff ** 2,
                                operator.oplist[-1]]).assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values)))

        # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        exp = operator.assign_parameters(dict(zip(self._parameters,
                                                   self._parameter_values)))
        if self._backend is not None:
            h_squared = CircuitSampler(self._backend).convert(h_squared)
            exp = CircuitSampler(self._backend).convert(exp)

        h_squared = np.real(h_squared.eval())
        eps_squared += np.real(h_squared)

        exp = np.real(exp.eval())
        eps_squared -= exp ** 2

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))

        eps_squared += dtdt_state

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        regrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared += regrad2
        if eps_squared < 0:
            if np.abs(eps_squared) < 1e-10:
                eps_squared = 0
        return eps_squared, h_squared,  exp, dtdt_state, regrad2 * 0.5

    def _grad_error_t(self,
                 operator: OperatorBase,
                 ng_res: Union[List, np.ndarray],
                 grad_res: Union[List, np.ndarray],
                 metric: Union[List, np.ndarray]) -> float:

        """
        Evaluate the gradient of the l2 norm for a single time step of VarQITE.

        Args:
            operator: ⟨ψ(ω)|H|ψ(ω)〉
            ng_res: dω/dt
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric

        Returns:
            square root of the l2 norm of the error
        """
        if not isinstance(operator, ComposedOp):
            raise TypeError('Currently this error can only be computed for operators given as '
                            'ComposedOps')
        grad_eps_squared = 0
        # dω_jF_ij^Q
        grad_eps_squared += np.dot(metric, ng_res)
        # 2Re⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared += grad_res

        return grad_eps_squared

    def _distance_energy(self,
                  time: Union[float, complex],
                  trained_param_dict: Dict)-> (float, float, float):
        """
        Evaluate the fidelity to the target state, the energy w.r.t. the target state and
        the energy w.r.t. the trained state for a given time and the current parameter set
        Args:
            time: current evolution time
            trained_param_dict: dictionary which matches the operator parameters to the current
            values of parameters for the given time
        Returns: fidelity to the target state, the energy w.r.t. the target state and
        the energy w.r.t. the trained state
        """
        state = self._operator[-1]

        target_state = state.assign_parameters(dict(zip(trained_param_dict.keys(),
                                                      self._init_parameter_values)))
        trained_state = state.assign_parameters(trained_param_dict)

        if self._backend is not None:
            target_state = CircuitSampler(self._backend).convert(target_state)
            trained_state = CircuitSampler(self._backend).convert(trained_state)

        target_state = target_state.eval().primitive.data

        hermitian_op = self._operator[0].primitive.to_matrix_op().primitive.data * \
                       self._operator[0].primitive.to_matrix_op().coeff

        # Evolve with exponential operator
        target_state = np.dot(expm(-1 * hermitian_op * time), target_state)
        # Normalization
        target_state /= np.sqrt(self._inner_prod(target_state, target_state))
        trained_state = trained_state.eval().primitive.data
        # Fidelity
        f = state_fidelity(target_state, trained_state)
        # Actual error
        act_err = np.linalg.norm(target_state - trained_state, ord=2)
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(hermitian_op, target_state))
        print('actual energy', act_en)
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(hermitian_op, trained_state))
        print('trained_en', trained_en)

        print('Fidelity', f)
        print('True error', act_err)
        return f, act_err, np.real(act_en), np.real(trained_en)
