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
from scipy.integrate import ode, OdeSolver

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, ComposedOp, OperatorBase, \
    CircuitSampler


class VarQITE(VarQTE):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191

    Algorithms that use McLachlans variational principle to approximate the imaginary time
    evolution for a given Hermitian operator (Hamiltonian) and quantum state.
    """

    def convert(self,
                operator: ListOp) -> StateFn:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarIQTE) w.r.t. the given operator
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
            print('h_norm', h_norm)

        # Step size
        dt = np.abs(operator.coeff)*np.sign(operator.coeff) / self._num_time_steps

        # ODE Solver

        if self._ode_solver is not None:
            # ||e_t||^2
            # def ode_fun(time, params):
            #     param_dict = dict(zip(self._parameters, params))
            #     nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
            #     e_t = self._error_t(self._operator, nat_grad_result, grad_res,
            #                                    metric_res)[0]
            #     warnings.warn('Be careful that the following output is for the fidelity before '
            #                   'the parameter update.')
            #     fid_and_errors = self._distance_energy(time, trained_param_dict=dict(zip(
            #         self._parameters, self._parameter_values)))
            #     return [e_t ** 2]
            def ode_fun(time, params):
                param_dict = dict(zip(self._parameters, params))
                nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
                et = self._error_t(self._operator, nat_grad_result, grad_res,
                                               metric_res)[0]
                dw_et = self._grad_error_t(self._operator, nat_grad_result, grad_res,
                                               metric_res)
                print('grad error', et)
                warnings.warn('Be careful that the following output is for the fidelity before '
                              'the parameter update.')
                fid_and_errors = self._distance_energy(time, trained_param_dict=dict(zip(
                    self._parameters, self._parameter_values)))
                # return nat_grad_result
                return dw_et

            # def jac_ode_fun(params):
            #     param_dict = dict(zip(self._parameters, params))
            #     nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
            #     return np.dot(metric_res, nat_grad_result) + grad_res

            # self._ode_solver.fun = ode_fun
            # self._ode_solver.jac = jac_ode_fun
            if isinstance(self._ode_solver, OdeSolver):
                self._ode_solver=self._ode_solver(fun=ode_fun, t0=0, y0=self._parameter_values,
                                 t_bound=dt*self._num_time_steps)
            # elif isinstance(self._ode_solver, ode):
            else:
                self._ode_solver = ode(ode_fun).set_integrator('vode', method='bdf')

                self._ode_solver.set_initial_value(self._parameter_values, 0)

                t1 = dt*self._num_time_steps

                dt = 1

                while self._ode_solver.successful() and self._ode_solver.t < t1:
                    print('ode step', self._ode_solver.t + dt, self._ode_solver.integrate(
                        self._ode_solver.t +
                                                                              dt))
                    self._parameter_values = self._ode_solver.y
                    print('time', self._ode_solver.t)
                _ = self._distance_energy(dt * self._num_time_steps,
                                                       trained_param_dict=dict(zip(
                                                       self._parameters,
                                                       self._parameter_values)))

                # Return variationally evolved operator
                return self._operator[-1].assign_parameters(dict(zip(self._parameters,
                                                                     self._parameter_values)))
            # else:
            #     raise TypeError('Please provide a scipy ODESolver or ode type object.')

        # Initialize error bound
        error_bound_en_diff = 0
        error_bound_l2 = 0

        # Zip the parameter values to the parameter objects
        param_dict = dict(zip(self._parameters, self._parameter_values))

        et_list = []
        h_squared_factor_list = []
        trained_energy_factor_list = []
        stddev_factor_list = []

        f, true_error, true_energy, trained_energy = self._distance_energy(dt, param_dict)
        for j in range(self._num_time_steps):
            # Get the natural gradient - time derivative of the variational parameters - and
            # the gradient w.r.t. H and the QFI/4.
            if self._ode_solver is None:
                nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
                # print('nat_grad_result', np.round(nat_grad_result, 3))
                # print('C', grad_res)
                # print('metric', metric_res)

                if self._get_error:
                    # Get the residual for McLachlan's Variational Principle
                    resid = np.linalg.norm(np.matmul(metric_res, nat_grad_result) + 0.5 * grad_res)
                    print('Residual norm', resid)

                    # Get the error for the current step
                    et, h_squared, exp, dtdt_state, regrad = self._error_t(
                        self._operator, nat_grad_result, grad_res, metric_res)
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
                    if j == 0:
                        et_prev = 0
                        sqrt_h_prev_square = 0
                    # print('factor', (1 + 2 * dt * h_norm) ** (self._num_time_steps - j))
                    # error += dt * et * (1 + 2 * dt * h_norm) ** (self._num_time_steps - j - 1)

                    # error += dt * et * (1 + 2 * dt * np.sqrt(h_squared)
                    #                      ) ** (self._num_time_steps - j - 1)
                    # error += dt/2 * (et * np.exp(2*np.abs(operator.coeff)*np.sqrt(h_squared)) +
                    #                 et_prev * np.exp(2*np.abs(operator.coeff)*sqrt_h_prev_square))
                    error_bound_l2 += dt / 2 * np.exp(2 * np.abs(operator.coeff) * h_norm) * \
                                           (et + et_prev)
                    # error += dt / 2 * (et + et_prev)
                    et_prev = et
                    # sqrt_h_prev_square = np.sqrt(h_squared)

                    # error += dt * (et + 2 * h_norm)
                    # error += dt * et * (1 + 2 * dt * h_norm) ** (np.abs(operator.coeff) - (j * dt))

                # Propagate the Ansatz parameters step by step using explicit Euler
                # TODO enable the use of arbitrary ODE solvers
                # Subtract is correct either
                # omega_new = omega - A^(-1)Cdt or
                # omega_new = omega + A^(-1)((-1)*C)dt

                # Store the current status
                if self._snapshot_dir:
                    if self._get_error:
                        if self._init_parameter_values is None:
                            self._store_params(j * dt, self._parameter_values,
                                               error_bound_en_diff, et,
                                               resid)
                        else:
                            # h_norm_factor: (1 + 2 \delta_t | | H | |) ^ {T - t}
                            # h_squared_factor: (1 + 2\delta_t \sqrt{| < trained | H ^ 2 | trained > |})
                            # trained_energy_factor: (1 + 2 \delta_t | < trained | H | trained > |)
                            if self._get_h_terms:
                                self._store_params(j * dt, self._parameter_values,
                                                   error_bound_l2, et,
                                                   resid,  f,  true_error, true_energy,
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
                    # error += dt * (et + 2*h_norm)
                    # error += dt * (et + np.sqrt(np.linalg.norm(trained_energy - trained_energy_0)))

                print('Error bound based on exact energy difference', np.round(error_bound_en_diff,
                                                                               6), 'after',
                      (j + 1) * dt)
                print('Error bound based on ||H||_2 and integration', np.round(error_bound_l2, 6),
                      'after', (j + 1) * dt)

            else:
                self._ode_solver.step()
                self._parameter_values = self._ode_solver.y
                if self._ode_solver.status == 'finished' or self._ode_solver.status == 'failed':
                    break
                pass

        if self._ode_solver is None:

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
        else:
            _ = self._distance_energy(dt*self._num_time_steps,
                                                   trained_param_dict=dict(zip(
                self._parameters, self._parameter_values)))

        # Return variationally evolved operator
        return self._operator[-1].assign_parameters(param_dict)

    def _error_t(self,
         operator: OperatorBase,
         ng_res: Union[List, np.ndarray],
         grad_res: Union[List, np.ndarray],
         metric: Union[List, np.ndarray]) -> float:

        """
        Evaluate the square root of the l2 norm for a single time step of VarQRTE.

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
        # print('State', operator.oplist[-1].assign_parameters(dict(zip(self._parameters,
        #                                                          self._parameter_values))).eval())

        # print('H^2', (operator.oplist[0].primitive ** 2).eval() * operator.oplist[0].coeff ** 2)

        #TODO CircuitSampler
        # ⟨ψ(ω)|H^2|ψ(ω)〉Hermitian
        h_squared = np.real(ComposedOp([~StateFn(operator.oplist[0].primitive ** 2) *
                                operator.oplist[0].coeff ** 2,
                                operator.oplist[-1]]).assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values))).eval())
        print('h^2', np.round(h_squared, 6))
        eps_squared += np.real(h_squared)

        # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        exp = np.real(operator.assign_parameters(dict(zip(self._parameters,
                                                   self._parameter_values))).eval())
        print('exp squared', np.round(exp ** 2, 6))
        eps_squared -= exp ** 2

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))
        print('dtdt state', np.round(dtdt_state, 6))

        eps_squared += dtdt_state

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        regrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared += regrad2
        print('2dt state', np.round(regrad2, 6))

        # if exp != 0:
        #     overlap_op = ComposedOp([~StateFn(exp * I^operator.num_qubits), operator[-1]])
        #     overlap = Gradient(self._grad_method).convert(overlap_op, self._parameters
        #                                                   ).assign_parameters(
        #         dict(zip(self._parameters, self._parameter_values))).eval()
        #     overlap = self._inner_prod(ng_res, overlap)
        #     print('energy overlap', overlap)
        #     eps_squared -= overlap
        print('Energy Variance', h_squared - exp **2)
        if eps_squared < 0:
            if np.abs(eps_squared) < 1e-10:
                eps_squared = 0
        print('Grad error', np.sqrt(eps_squared))
        return np.sqrt(eps_squared), h_squared,  exp, dtdt_state, regrad2 * 0.5

    def _grad_error_t(self,
                 operator: OperatorBase,
                 ng_res: Union[List, np.ndarray],
                 grad_res: Union[List, np.ndarray],
                 metric: Union[List, np.ndarray]) -> float:

        """
        Evaluate the gradient of the l2 norm for a single time step of VarQRTE.

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
