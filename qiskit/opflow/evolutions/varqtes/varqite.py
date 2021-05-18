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

from scipy.optimize import fmin_cobyla

import math
import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.integrate import ode, OdeSolver, solve_ivp

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, ComposedOp, PauliExpectation

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
        self._operator_eval = PauliExpectation().convert(operator / operator.coeff)

        # Step size
        dt = np.abs(operator.coeff)*np.sign(operator.coeff) / self._num_time_steps

        self._init_grad_objects()
        # Run ODE Solver
        parameter_values = self._run_ode_solver(dt * self._num_time_steps,
                                                self._init_parameter_values)
        # return evolved
        return self._state.assign_parameters(dict(zip(self._parameters,
                                                      parameter_values)))

    def _error_t(self,
                 param_values: Union[List, np.ndarray],
         ng_res: Union[List, np.ndarray],
         grad_res: Union[List, np.ndarray],
         metric: Union[List, np.ndarray]) -> Tuple[
         int, Union[np.ndarray, int, float, complex], Union[np.ndarray, complex, float], Union[
            Union[complex, float], Any], float]:

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
                                                             params=param_dict)
            h_trip = self._h_trip_circ_sampler.convert(self._h_trip, params=param_dict)
        else:
            h_squared = self._h_squared.assign_parameters(param_dict)
            h_trip = self._h_trip.assign_parameters(param_dict)
        h_squared = np.real(h_squared.eval())
        h_trip = np.real(h_trip.eval())

        # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        if self._backend is not None:
            exp = self._operator_circ_sampler.convert(self._operator_eval,
                                                      params=param_dict)
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

        return np.real(eps_squared), h_squared, dtdt_state, regrad2 * 0.5, h_trip

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
                         times: List,
                         stddevs: List,
                         h_squareds: List,
                         h_trips: List,
                         H: Union[List, np.ndarray],
                         energies: List,
                         imag_reverse_bound: bool = True) -> Union[List, Tuple[List, List]]:
        """
        Get the upper bound to a global phase agnostic l2-norm error for VarQITE simulation
        Args:
            gradient_errors: Error of the state propagation gradient for each t in times
            times: List of all points in time considered throughout the simulation
            stddevs: Standard deviations for times sqrt(⟨ψ(ω)|H^2| ψ(ω)〉- ⟨ψ(ω)|H| ψ(ω)〉^2)
            h_squareds: ⟨ψ(ω)|H| ψ(ω)〉^2 for all times
            imag_reverse_bound: If True compute the reverse error bound
            H: If imag_reverse_bound find the first and second Eigenvalue of H to compute the
               reverse bound

        Returns:
            List of the error upper bound for all times

        Raises: NotImplementedError

        """

        if not len(gradient_errors) == len(times):
            raise Warning('The number of the gradient errors is incompatible with the number of '
                          'the time steps.')


        def optimization(eps: float,
                         e: float,
                         h_squared: float,
                         h_trip: float,
                         delta_t: float) -> float:
            print('hsquared ', h_squared)
            print('e ', e)

            c_alpha = lambda a: np.sqrt((1-a)**2 + 2*a *(1-a)*e + a**2*h_squared)

            def optimization_fun(alpha: Iterable[float]) -> float:
                print('e ', e)
                alpha = alpha[0]

                e_star = ((1 - alpha)**2 * e + 2* (alpha - alpha**2) * h_squared + alpha **2 *
                          h_trip) / c_alpha(alpha) ** 2

                abs_value = (1 - alpha) * (1 + delta_t * (e_star - e)) + \
                            alpha * (e + delta_t * (e * (e + e_star) - 2 * h_squared))

                abs_value = np.abs(abs_value / c_alpha(alpha))

                print('abs value ', abs_value)
                return_val = np.sqrt(2)*np.sqrt(1 - abs_value)
                return return_val

            def constraint1(alpha: Iterable[float]) -> float:
                alpha = alpha[0]
                return np.abs((1 - alpha + alpha * e) / c_alpha(alpha)) - 1 + eps**2 /2

            def constraint2(alpha: Iterable[float]) -> float:
                # Constraint alpha >= 0
                return alpha[0]

            def constraint3(alpha: Iterable[float]) -> float:
                # Constraint alpha <= 1
                return 1 - alpha[0]
            # alpha_opt = fmin_cobyla(func=optimization_fun, x0=[0.5], rhobeg=0.01,
            #                         rhoend=1e-8, cons=[constraint2, constraint3, constraint1],
            #                         catol=1e-16)[0]
            alpha_opt_list = []
            objective_list = []
            # constraint0_list = []
            constraint1_list = []
            for a in np.linspace(0, 1, 1000):
                opt_fun = optimization_fun([a])
                # print('optimization function ', opt_fun)
                # print('alpha ', a)
                if math.isnan(opt_fun):
                    pass
                elif constraint1([a]) < 0:
                    pass
                else:
                    objective_list.append(opt_fun)
                    # constraint0_list.append(constraint0([a]))
                    # constraint1_list.append()
                    alpha_opt_list.append([a])
            if len(objective_list) == 0:
                print('No suitable alpha found')

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(alpha_opt_list, objective_list)
            plt.ylabel('objective value')
            plt.xlabel(r'$\alpha$')
            plt.savefig(self._snapshot_dir + '/objective_values.png')
            plt.close()

            # plt.figure(2)
            # plt.scatter(alpha_opt_list, constraint1_list)
            # plt.ylabel('constraint value')
            # plt.xlabel(r'$\alpha$')
            # plt.savefig(self._snapshot_dir + '/constraint_values.png')
            # plt.close()

            print('be careful maximization happening')
            # index = objective_list.index(max(objective_list))
            index = objective_list.index(max(objective_list))
            alpha_opt = alpha_opt_list[index]
            print('alpha_opt ', alpha_opt)
            print('Y(alpha_opt) ', optimization_fun(alpha_opt))
            return optimization_fun(alpha_opt)

        error_bounds = [0]

        for j in range(len(times)):
            if j == 0:
                continue
            delta_t = times[j]-times[j-1]
            y = optimization(error_bounds[j-1], energies[j-1], h_squareds[j-1],
                             h_trips[j-1], delta_t)

            # \epsilon_{t+1}
            error_bounds.append(y + delta_t * gradient_errors[j-1])

#--------------------------------
        """
       
        norms = []
        for e in energies:
            norms.append(np.linalg.norm(e * np.eye(np.shape(H)[0]) - H, np.inf))
        
        # integral_items = np.add(2 * stddevs, norms)
        # or
        
        integral_items = np.add(stddevs, norms)
        # integral_items = stddevs
        gradient_error_factors = []
        for j in range(len(times)):
            stddev_factor = np.exp(np.trapz(integral_items[j:], x=times[j:]))
            gradient_error_factors.append(stddev_factor)

        e_bounds = []
        for j in range(len(times)):
            e_bounds.append(np.trapz(np.multiply(gradient_errors[:j+1], gradient_error_factors[
                                                                        :j+1]), x=times[:j+1]))
        
        # print('Error bounds ', e_bounds)

        # e_bounds = [np.sqrt(2) if e_bound > np.sqrt(2) else e_bound for e_bound in e_bounds]

        if imag_reverse_bound:
            if H is None:
                raise Warning('Please support the respective Hamiltonian.')
            eigvals = []
            evs = np.linalg.eigh(H)[0]
            for eigv in evs:
                add_ev = True
                for ev in eigvals:
                    if np.isclose(ev, eigv):
                        add_ev = False
                if add_ev:
                    eigvals.append(eigv)
            eigvals = sorted(eigvals)
            e0 = eigvals[0]
            e1 = eigvals[1]
            # Reverse error bound final time
            reverse_bounds = [stddevs[-1] / (e1 - e0)]
            reverse_bounds_temp = np.flip(np.multiply(gradient_errors, gradient_error_factors))
            # reverse_bounds_temp[-1] = reverse_bounds[0]
            reverse_times = np.flip(times)
            for j, dt in enumerate(reverse_times):
                if j == 0:
                    continue
                # if use_integral_approx:
                    # TODO check here if correct
                reverse_bounds.append(reverse_bounds[0] - np.trapz(reverse_bounds_temp[:j],
                                                                   x=reverse_times[:j]))

                # else:
                #
                #     reverse_bounds.append(reverse_bounds[j] + reverse_bounds_temp[j+1] *
                #                           reverse_times[j])

            reverse_bounds.reverse()

            # reverse_bounds = [np.sqrt(2) if e_bound > np.sqrt(2) else e_bound for e_bound in
            #                   reverse_bounds]
            return e_bounds, reverse_bounds
             """
        print('error bounds', np.around(error_bounds, 4))
        print('gradient errors', np.around(gradient_errors, 4))
        return error_bounds

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

