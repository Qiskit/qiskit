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
                                                             params=param_dict)
        else:
            h_squared = self._h_squared.assign_parameters(param_dict)
        h_squared = np.real(h_squared.eval())

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
                         times: List,
                         stddevs: List,
                         h_squareds: List,
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
                         var: float,
                         h_squared: float) -> float:

            c_alpha = lambda a: np.sqrt((1-a)**2 + 2*a *(1-a)*e + a**2*h_squared)

            def optimization_fun(alpha: float) -> float:
                return (-1) * eps**2 / c_alpha(alpha) * (e + alpha * (var - e - e**2))

            def constraint1(alpha: float):
                return (1 - alpha + alpha * e) / c_alpha(alpha)**2 - 1 + eps**2 /2

            def constraint2(alpha: float) -> float:
                # Constraint alpha >= 0
                return alpha

            def constraint3(alpha: float) -> float:
                # Constraint alpha <= 1
                return 1 - alpha

            return fmin_cobyla(func=optimization_fun, x0=0.5, cons=[constraint1, constraint2,
                                                                   constraint3]).tolist()

        error_bounds = [0]

        for j in range(len(times)):
            if j == 0:
                continue
            delta_t = times[j]-times[j-1]
            y = optimization(error_bounds[j-1], energies[j-1], stddevs[j-1]**2, h_squareds[j-1])
            #  bound for |<\psi_{t+1}|\psi^*_{t+1}>|**2
            overlap_item = (1 - error_bounds[j-1]**2 / 2)**2 + \
                           2 * delta_t * (1 - error_bounds[j-1]**2 / 2) * \
                           ((1 - error_bounds[j-1]**2 / 2)*energies[j-1] - y)
            # bound for |<\psi_{t+1}|\psi^*_{t+1}>|
            overlap_item = np.sqrt(overlap_item)
            # \epsilon_{t+1}
            error_bounds.append(np.sqrt(2-2*overlap_item))

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
         """
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

