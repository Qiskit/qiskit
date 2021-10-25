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

"""The Variational Quantum Time Evolution Interface"""

from typing import Union, Dict, List, Iterable

import warnings

import numpy as np
from scipy.linalg import expm
from scipy.integrate import OdeSolver, ode

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ComposedOp, PauliExpectation

# from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve


class VarQRTE(VarQTE):
    """Variational Quantum Real Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191

    Algorithms that use McLachlans variational principle to compute a real time evolution for a
    given Hermitian operator and quantum state.
    """

    def convert(self,
                operator: ComposedOp) -> StateFn:
        """
        Apply Variational Quantum Time Evolution (VarQTE) w.r.t. the given operator
        Args:
            operator:
                Operator used vor Variational Quantum Real Time Evolution (VarQRTE)
                The coefficient of the operator determines the evolution time.
                If the coefficient is real this method implements VarQRTE.

                The operator may for now ONLY be given as a composed op consisting of a
                Hermitian observable and a CircuitStateFn.

        Returns:
            StateFn (parameters are bound) which represents an approximation to the respective
            time evolution.

        """
        if not isinstance(operator, ComposedOp) or len(operator.oplist) != 2:
            raise TypeError('Please provide the operator as a ComposedOp consisting of the '
                            'observable and the state (as CircuitStateFn).')
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError('Please provide the state as a CircuitStateFn.')

        # For VarQRTE we need to add a -i factor to the operator coefficient.
        self._operator = 1j * operator / operator.coeff
        self._operator_eval = PauliExpectation().convert(operator / operator.coeff)

        self._init_grad_objects()
        # Step size
        dt = np.abs(operator.coeff) / self._num_time_steps
        # Run ODE Solver
        parameter_values = self._run_ode_solver(dt * self._num_time_steps,
                                                self._init_parameter_values)
        # return evolved
        return self._state.assign_parameters(dict(zip(self._parameters,
                                                      parameter_values)))

        # if self._regularization == 'energy':
        #     if self._backend is not None:
        #         exp_0 = self._operator_circ_sampler.convert(self._operator_eval,
        #                                             params=dict(zip(self._parameters,
        #                                                             self._parameter_values)))[0]
        #     else:
        #         exp_0 = self._operator_eval.assign_parameters(dict(zip(self._parameters,
        #                                                 self._parameter_values)))
        #     exp_0 = np.real(exp_0.eval())
        #
        #     def energy_reg(dt_params):
        #         # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        #         if self._backend is not None:
        #             exp_evolved = self._operator_circ_sampler.convert(self._operator_eval,
        #                                                 params=dict(zip(self._parameters,
        #                                                                 self._parameter_values +
        #                                                             dt_params)))[0]
        #         else:
        #             exp_evolved = self._operator_eval.assign_parameters(dict(zip(self._parameters,
        #                                                                    self._parameter_values
        #                                                                    + dt_params)))
        #         exp_evolved = np.real(exp_evolved.eval())
        #         return 0.1 * np.linalg.norm(exp_evolved - exp_0)
        #
        #     self._regularization = energy_reg
        #

    def _error_t(self,
                 param_values: Union[List, np.ndarray],
                 ng_res: Union[List, np.ndarray],
                 grad_res: Union[List, np.ndarray],
                 metric: Union[List, np.ndarray]) -> [float]:

        """
        Evaluate the l2 norm of the error for a single time step of VarQRTE.

        Args:
            operator: ⟨ψ(ω)|H|ψ(ω)〉
            ng_res: dω/dt
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric

        Returns:
            The l2 norm of the error
        """
        eps_squared = 0

        param_dict = dict(zip(self._parameters, param_values))

        # ⟨ψ(ω)|H^2|ψ(ω)〉
        if self._backend is not None:
            h_squared = self._h_squared_circ_sampler.convert(self._h_squared,
                                                             params=param_dict)
        else:
            h_squared = self._h_squared.assign_parameters(param_dict)
        h_squared = np.real(h_squared.eval())

        eps_squared += h_squared

        # ⟨ψ(ω) | H | ψ(ω)〉^2 Hermitian
        if self._backend is not None:
            exp = self._operator_circ_sampler.convert(self._operator_eval,
                                                      params=param_dict)
        else:
            exp = self._operator_eval.assign_parameters(param_dict)
        exp = np.real(exp.eval())
        eps_squared -= np.real(exp ** 2)


        # print('hsquared', h_squared)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))
        # print('dtdt', dtdt_state)
        eps_squared += dtdt_state
        # print('metric', np.round(metric, 3))

        # 2Im⟨dtψ(ω)| H | ψ(ω)〉= 2Im dtω⟨dωψ(ω)|H | ψ(ω)
        # 2 missing b.c. of Im
        # print('nat grad result', np.round(ng_res, 3))
        # print('grad res', np.round(grad_res, 3))
        imgrad2 = self._inner_prod(grad_res, ng_res)
        # print('imgrad 2', imgrad2)
        eps_squared -= imgrad2
        # print('eps squared', eps_squared)
        if eps_squared < 0:
            print(eps_squared)
        return np.real(eps_squared), h_squared, dtdt_state, imgrad2 * 0.5

    def _grad_error_t(self,
                 ng_res: Union[List, np.ndarray],
                 grad_res: Union[List, np.ndarray],
                 metric: Union[List, np.ndarray]) -> float:

        """
        Evaluate the gradient of the l2 norm for a single time step of VarQRTE.

        Args:
            operator: ⟨ψ(ω)|H|ψ(ω)〉
            ng_res: dω/dt
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉
            metric: Fubini-Study Metric

        Returns:
            square root of the l2 norm of the error
        """
        grad_eps_squared = 0
        # dω_jF_ij^Q
        grad_eps_squared += np.dot(metric, ng_res) + np.dot(np.diag(np.diag(metric)),
                                                            np.power(ng_res, 2))

        # 2Im⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared -= grad_res

        # print('E_t squared', np.round(eps_squared, 4))
        if np.linalg.norm(np.imag(grad_eps_squared)) > 1e-6:
            raise Warning('Error gradient complex part are not to be neglected.')
        return np.real(grad_eps_squared)

    def _get_error_bound(self,
                         gradient_errors: List,
                         times: List,
                         energies: List) -> float:
        """
        Get the upper bound to a global phase agnostic l2-norm error for VarQRTE simulation
        Args:
            gradient_errors: Error of the state propagation gradient for each t in times
            times: List of all points in time considered throughout the simulation
            stddevs: Standard deviations for times sqrt(⟨ψ(ω)|H^2| ψ(ω)〉- ⟨ψ(ω)|H| ψ(ω)〉^2)

        Returns:
            List of the error upper bound for all times
        """

        if not len(gradient_errors) == len(times):
            raise Warning('The number of the gradient errors is incompatible with the number of '
                          'the time steps.')
        # errors = [gradient_errors[j] + energies[j]-energies[0] for j in range(len(energies))]
        e_bounds = []
        for j, dt in enumerate(times):
            # e_bound.append(np.trapz(errors[:j+1], x=times[:j+1]))
            e_bounds.append(np.trapz(gradient_errors[:j + 1], x=times[:j + 1]))
            # e_bound.append(e_bound[j] + gradient_errors[j] * dt)
        # TODO add clipping
        # e_bounds = [np.sqrt(2) if e_bound > np.sqrt(2) else e_bound for e_bound in e_bounds]
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
        target_state = np.dot(expm(-1j * self._h_matrix * time), self._init_state)
        return target_state

    def _exact_grad_state(self,
                          state: Iterable) -> Iterable:
        """
        Return the gradient of the given state
        -i H |state>

        Args:
            state: State for which the exact gradient shall be evaluated

        Returns:
            Exact gradient of the given state

        """
        return np.matmul(-1j * self._h_matrix, state)
