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

import os
import csv

from typing import List, Union, Iterable, Tuple, Any

from scipy.optimize import fmin_cobyla

import math
import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.integrate import ode, OdeSolver, solve_ivp

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, ComposedOp, PauliExpectation


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

    @staticmethod
    def get_max_bures(eps: float,
                      e: float,
                      e_factor: float,
                      h_squared: float,
                      h_trip: float,
                      delta_t: float) -> float:
        """
        Compute  max_alpha B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
        Args:
            eps: Error from the previous time step
            e: Energy <psi_t|H|psi_t>
            e_factor: Upper bound to |E - E*|
            h_squared: <psi_t|H^2|psi_t>
            h_trip: <psi_t|H^3|psi_t>
            delta_t: time step

        Returns: max_alpha B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))

        """

        c_alpha = lambda a: np.sqrt(
            (1 - np.abs(a)) ** 2 + 2 * a * (1 - np.abs(a)) * e + a ** 2 * h_squared)

        # e_star = lambda a: ((1 - np.abs(a)) ** 2 * e + 2 * (a - a * np.abs(a)) * h_squared +
        #                     a ** 2 * h_trip) / c_alpha(a) ** 2

        def overlap(alpha: Iterable[float]) -> float:
            """
            Compute generalized Bures metric B(I + delta_t(E_t-H)|psi_t>, I + delta_t(
            E_t-H)|psi*_t>(alpha))
            Args:
                alpha: optimization parameter alpha

            Returns: B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))

            """

            alpha = alpha[0]

            x = np.abs(((1 + 2 * delta_t * e) * (1 - np.abs(alpha) + alpha * e) -
                        2 * delta_t * ((1-np.abs(alpha)) * e + alpha * h_squared))
                       / c_alpha(alpha))

            return x

        def constraint1(alpha: Iterable[float]) -> float:
            """
            This constraint ensures that the optimization chooses a |psi*_t> which is in
            accordance with the prior state error
            Args:
                alpha: optimization value

            Returns: |<|psi_t|psi*_t>| - (1 + eps^2/2)

            """
            alpha = alpha[0]
            return np.abs((1 - np.abs(alpha) + alpha * e) / c_alpha(alpha)) - 1 + eps ** 2 / 2

        def constraint2(alpha: Iterable[float]) -> float:
            # Constraint |alpha| <= 1
            return 1 - np.abs(alpha[0])

        alpha_opt = None
        x = None
        # TODO Use again finer grid of 10**6
        # Grid search over alphas for the optimization
        a_grid = np.append(np.linspace(-1, 1, 10**7), 0)
        for a in a_grid:
            returned_bures = overlap([a])
            if math.isnan(returned_bures):
                print('optimization fun is nan')
                pass
            # Ensure that constraint is sufficed
            elif constraint1([a]) < 0:
                pass
            else:
                # Check if the current bures metric is bigger than the max.
                if x is None or returned_bures < x:
                    x = returned_bures
                    alpha_opt = a

        print('after grid x overlap', x)
        # After the grid use the resulting optimal alpha and do another optimization search
        # while True:
        #     alpha_opt = fmin_cobyla(func=lambda x: overlap(x), x0=[alpha_opt],
        #                             rhobeg=1e-5, catol=1e-12, maxfun=1000000,
        #                             rhoend=1e-10, cons=[constraint1, constraint2])[0]
        #     if np.abs(alpha_opt) <= 1:
        #         break
        #     print('Warning illegal alpha ', alpha_opt)
        #
        # x = overlap([alpha_opt])
        # if x > 1:
        #     x = 1.
        #     print('warning x clipped to 1 ', x)
        #
        # print('after cobyla x overlap', x)
        print('dt * e ', delta_t * e_factor)
        print('x ', x)
        max_bures = np.sqrt(2 + 2 * delta_t * e_factor - 2 * x)
        print('alpha_opt ', alpha_opt)
        print('Maximum bures metric ', max_bures)
        return max_bures

    def _get_error_grad(self,
                        delta_t: float,
                        eps_t: float,
                        grad_err: float,
                        energy: float,
                        h_squared: float,
                        h_trip: float,
                        stddev: float,
                        store: bool = False):
        eps_t = max(eps_t, 0)
        eps_t = min(eps_t, np.sqrt(2))
        eps_t_next = self.get_error_term(delta_t, eps_t, grad_err, energy, h_squared, h_trip,
                                         stddev, store)
        eps_t_next = max(eps_t_next, 0)
        eps_t_next = min(eps_t_next, np.sqrt(2))
        grad = (eps_t_next - eps_t) / delta_t
        if grad > 10:
            print('huge grad', grad)
        if grad < 0:
            print('negative grad', grad)
        return grad

    def get_energy_factor(self,
                          eps_t: float,
                          energy: float,
                          stddev: float):
        """
        Get upper bound to |E-E*|
        Args:
            eps_t:
            energy:
            stddev:

        Returns: upper bound to |E-E*|

        """
        def optimize_energy_factor(alpha):
            return np.abs(alpha[0] * energy - np.sqrt(1 - (1-alpha[0])**2)*stddev)

        def constraint1(alpha):
            """
            Ensure that alpha >= 0
            Args:
                alpha:

            Returns: alpha

            """
            return alpha[0]

        def constraint2(alpha):
            """
            Ensure that alpha <= eps_t^2/2
            Args:
                alpha:

            """
            return eps_t**2/2 - alpha[0]

        def constraint3(alpha):
            """
            Ensure that alpha <= eps_t^2/2
            Args:
                alpha:

            """
            return 1 - alpha[0]

        x = None
        # Grid search over alphas for the optimization
        print('eps_t', eps_t)
        if eps_t**2/2 > 1:
            grid = np.linspace(0, 1, int(1e7))
        else:
            grid = np.linspace(0, eps_t**2/2, int(1e7))
        for a in grid:
            returned_x = optimize_energy_factor([a])
            if math.isnan(returned_x):
                print('eps_t', eps_t)
                raise Warning('optimization fun is nan')
                pass
            else:
                # Check if the current bures metric is bigger than the max.
                if x is None or returned_x > x:
                    x = returned_x
                    alpha_opt = a
        print('after grid x energy', x)

        # alpha_opt = fmin_cobyla(func=lambda z: (-1) * optimize_energy_factor(z), x0=[alpha_opt],
        #                         rhobeg=1e-3, catol=1e-12, maxfun=1000000,
        #                         rhoend=1e-10, cons=[constraint1, constraint2, constraint3])
        # x = optimize_energy_factor([alpha_opt])[0]
        # print('after cobyla x energy', x)
        print('alpha opt energy', alpha_opt)
        print('energy factor part 1', eps_t**2 * self._h_norm)
        print('energy factor part 2', 2 * x)
        return eps_t**2 * self._h_norm + 2 * x

    def get_error_term(self,
                       d_t,
                       eps_t,
                       grad_err,
                       energy: float,
                       h_squared: float,
                       h_trip: float,
                       stddev: float,
                       store: bool = False):
        """
        Compute the error term for a given time step and a point in the simulation time
        Args:
            d_t: time step
            j: jth step in VarQITE

        Returns: eps_j(delta_t)

        """
        if eps_t < 0:
            eps_t = 0
            print('Warn eps_t neg. clipped to 0')
        if eps_t == 0 and np.abs(grad_err) < 1e-6:
            eps_t_next = 0
            energy_factor = 0
            y = 0
        else:
            energy_factor = self.get_energy_factor(eps_t, energy, stddev)
            # max B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
            y = self.get_max_bures(eps_t, energy, energy_factor, h_squared, h_trip, d_t)
            # eps_t*sqrt(var) + eps_t^2/2 * |E_t - ||H||_infty |
            # energy_factor = (2 * eps_t * stddev +
            #                  eps_t ** 2 / 2 * self._h_norm)
            eps_t_next = y + d_t * grad_err + d_t * energy_factor

        print('Max Bures ', y)
        print('grad factor ', grad_err)
        print('Energy error factor', energy_factor)
        if math.isnan(energy_factor):
            print('nan')
        if store:
            if not os.path.exists(os.path.join(self._snapshot_dir, 'energy_error_bound.npy')):
                energy_error_bounds = [energy_factor]
                max_bures_metrics = [y]

            else:
                energy_error_bounds = np.load(os.path.join(self._snapshot_dir,
                                                           'energy_error_bound.npy'))
                energy_error_bounds = np.append(energy_error_bounds, energy_factor)
                print('Energy error bounbds', energy_error_bounds)
                max_bures_metrics = np.load(os.path.join(self._snapshot_dir, 'max_bures.npy'))
                max_bures_metrics = np.append(max_bures_metrics, y)
            np.save(os.path.join(self._snapshot_dir, 'energy_error_bound.npy'),
                    energy_error_bounds)
            np.save(os.path.join(self._snapshot_dir, 'max_bures.npy'), max_bures_metrics)

            if not os.path.exists(os.path.join(self._snapshot_dir, 'varqite_bound_output.csv')):
                with open(os.path.join(self._snapshot_dir, 'varqite_bound_output.csv'), mode='w')\
                        as \
                        csv_file:
                    fieldnames = ['eps_t', 'dt', 'opt_factor', 'grad_factor', 'energy_factor',
                                  'stddev',
                                  '|norm(H)|']

                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()


            # Write terms to csv file
            with open(os.path.join(self._snapshot_dir, 'varqite_bound_output.csv'), mode='a') as \
                    csv_file:
                fieldnames = ['eps_t', 'dt', 'opt_factor', 'grad_factor', 'energy_factor', 'stddev',
                              '|norm(H)|']

                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writerow({'eps_t': np.round(eps_t, 8),
                                 'dt': d_t,
                                 'opt_factor': np.round(y, 8),
                                 'grad_factor': np.round(grad_err, 8),
                                 'energy_factor': np.round(energy_factor, 8),
                                 'stddev': np.round(stddev, 8),
                                 '|norm(H)|': np.round(self._h_norm, 8)
                                 })
        # \epsilon_{t+1}
        return eps_t_next


    def _get_error_bound(self,
                         gradient_errors: List,
                         times: List,
                         stddevs: List,
                         h_squareds: List,
                         h_trips: List,
                         energies: List,
                         trapezoidal: bool=True,
                         store: bool=False) -> Union[List, Tuple[List, List]]:
        """
        Get the upper bound to the Bures metric between prepared and target state for VarQITE
        simulation
        Args:
            gradient_errors: Error of the state propagation gradient for each t in times
            times: List of all points in time considered throughout the simulation
            stddevs: Standard deviations for times sqrt(⟨ψ(ω)|H^2| ψ(ω)〉- ⟨ψ(ω)|H| ψ(ω)〉^2)
            h_squareds: ⟨ψ(ω)|H^2| ψ(ω) for all times
            h_trips: ⟨ψ(ω)|H^3| ψ(ω)〉for all times
            H: If imag_reverse_bound find the first and second Eigenvalue of H to compute the
               reverse bound
            energies: ⟨ψ(ω)|H| ψ(ω) for all times
            trapezoidal: If True use trapezoidal rule to compute error bounds.

        Returns:
            List of the error upper bound for all times

        Raises: NotImplementedError

        """

        if not len(gradient_errors) == len(times):
            raise Warning('The number of the gradient errors is incompatible with the number of '
                          'the time steps.')

        if store:
            with open(os.path.join(self._snapshot_dir, 'varqite_bound_output.csv'), mode='w') as \
                    csv_file:
                fieldnames = ['eps_t', 'dt', 'opt_factor', 'grad_factor', 'energy_factor', 'stddev',
                              '|norm(H)|']

                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            energy_error_bounds = [0]
            max_bures_metrics = [0]

            np.save(os.path.join(self._snapshot_dir, 'energy_error_bound.npy'),
                    energy_error_bounds)
            np.save(os.path.join(self._snapshot_dir, 'max_bures.npy'), max_bures_metrics)

        error_bounds = [0]
        if trapezoidal:
            trap_grad = [0]

        # Compute error bound for all the time steps of the ODE Solver
        for j in range(len(times)):
            if j == 0:
                continue
            if not trapezoidal:
                delta_t = times[j] - times[j-1]
                eb_term = self.get_error_term(delta_t, error_bounds[j - 1],
                                                        gradient_errors[j - 1], energies[j - 1],
                                                        h_squareds[j - 1], h_trips[j - 1],
                                                        stddevs[j - 1], store=store)
                eb_append = max(eb_term, 0)
                eb_append = min(eb_append, np.sqrt(2))
                error_bounds.append(eb_append)
            else:
                # Use a finite difference approx. of the gradient underlying the error at time t
                # to enable the use of an integral formulation of the error
                #TODO avoid hard-coding of delta_t
                delta_t_trap = 1e-4
                trap_grad_term = self._get_error_grad(delta_t_trap, error_bounds[j - 1],
                                                        gradient_errors[j - 1], energies[j - 1],
                                                        h_squareds[j - 1], h_trips[j - 1],
                                                        stddevs[j - 1], store=store)
                trap_grad.append(trap_grad_term)
                # Compute an approx. to the integral formulation of eps_t using the trapezoidal rule
                error_trap_term = np.trapz(trap_grad, x=times[:j + 1])
                if error_trap_term < 0:
                    if np.abs(error_trap_term) < 1e-4:
                        error_trap_term = 0
                    else:
                        print('error neg.', error_trap_term)
                        # raise Warning('Negative Error')

                error_bounds.append(np.max(0, error_trap_term))
        print('error bounds', np.around(error_bounds, 4))
        print('gradient errors', np.around(gradient_errors, 4))

        return error_bounds

    def _get_reverse_error_bound(self,
                                 times: List,
                                 error_bound_grads: List,
                                 stddevs: List) -> Union[List, Tuple[List, List]]:
        """
        Get the reverse upper bound to the Bures metric between prepared and target state for
        VarQITE simulation
        Args:
            times: List of all points in time considered throughout the simulation
            error_bound_grads: error_bound = integral(error_bound_grads)
            stddevs: Standard deviations for times sqrt(⟨ψ(ω)|H^2| ψ(ω)〉- ⟨ψ(ω)|H| ψ(ω)〉^2)

        Returns:
            List of the reverse error upper bound for all times

        Raises: NotImplementedError

        """
        eigvals = []
        evs = np.linalg.eigh(self._h_matrix)[0]
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
        reverse_bounds_temp = np.flip(error_bound_grads)
        reverse_times = np.flip(times)
        for j, dt in enumerate(reverse_times):
            if j == 0:
                continue
            reverse_bounds.append(reverse_bounds[0] - np.trapz(reverse_bounds_temp[:j],
                                                               x=reverse_times[:j]))
        return np.flip(reverse_bounds)

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

