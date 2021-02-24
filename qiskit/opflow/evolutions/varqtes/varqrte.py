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

from typing import Union, Dict, List

import warnings

import numpy as np
from scipy.linalg import expm
from scipy.integrate import OdeSolver, ode

from qiskit.quantum_info import state_fidelity

from qiskit.opflow.evolutions.varqte import VarQTE
from qiskit.opflow import StateFn, CircuitStateFn, ComposedOp, CircuitSampler, OperatorBase

from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve


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
        if not isinstance(operator, ComposedOp):
            raise TypeError('Please provide the operator as a ComposedOp consisting of the '
                            'observable and the state (as CircuitStateFn).')
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError('Please provide the state as a CircuitStateFn.')

        # For VarQRTE we need to add a -i factor to the operator coefficient.
        self._operator = 1j * operator / operator.coeff


        # Step size
        dt = np.abs(operator.coeff) / self._num_time_steps

        # Initialize error
        error = 0
        # Assign parameter values to parameter items
        param_dict = dict(zip(self._parameters, self._parameter_values))

        # ODE Solver

        if self._ode_solver is not None:
            self._parameter_values = self._run_ode_solver(dt * self._num_time_steps)
            # Return variationally evolved operator
            return self._operator[-1].assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values)))

        for j in range(self._num_time_steps):

            # Get the natural gradient - time derivative of the variational parameters - and
            # the gradient w.r.t. H and the QFI/4.
            nat_grad_result, grad_res, metric_res = self._propagate(param_dict)

            # Evaluate the error bound
            if self._get_error:
                # Get the residual for McLachlan's Variational Principle
                resid = np.linalg.norm(np.matmul(metric_res, nat_grad_result) - 0.5 * grad_res)

                # Get the error for the current step
                e_t = self._error_t(self._operator, nat_grad_result, grad_res, metric_res)
                if et < 0 and np.abs(et) > 1e-4:
                    raise Warning('Non-neglectible negative et observed')
                else:
                    et = np.sqrt(np.real(et))
                error += dt * e_t
                print('dt', dt)
                print('et', e_t)
                print('Error', np.round(error, 3),  'after', j, ' time steps.')

            # TODO enable the use of arbitrary ODE solvers
            # Propagate the Ansatz parameters step by step using explicit Euler
            self._parameter_values = list(np.add(self._parameter_values, dt *
                                                 np.real(nat_grad_result)))


            # Assign parameter values to parameter items
            param_dict = dict(zip(self._parameters, self._parameter_values))
            # If initial parameter values were set compute the fidelity, the error between the
            # prepared and the target state, the energy w.r.t. the target state and the energy
            # w.r.t. the prepared state
            if self._init_parameter_values is not None:
                f, true_error, true_energy, trained_energy = self._distance_energy((j + 1) * dt,
                                                                            param_dict)

            # Store the current status
            if self._snapshot_dir:
                if self._get_error:
                    if self._init_parameter_values:
                        self._store_params((j + 1) * dt, self._parameter_values)
                    else:
                        self._store_params((j + 1) * dt, self._parameter_values, error, e_t, resid)
                else:
                    if self._init_parameter_values:
                        self._store_params((j + 1) * dt, self._parameter_values, f,
                                       true_error, true_energy, trained_energy)
                    else:
                        self._store_params((j + 1) * dt, self._parameter_values)

        # Return variationally evolved operator
        return self._operator.oplist[-1].assign_parameters(param_dict)

    def _error_t(self,
                 operator: OperatorBase,
                 ng_res: Union[List, np.ndarray],
                 grad_res: Union[List, np.ndarray],
                 metric: Union[List, np.ndarray]) -> float:

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
        if not isinstance(operator, ComposedOp):
            raise TypeError('Currently this error can only be computed for operators given as '
                            'ComposedOps')
        eps_squared = 0

        # ⟨ψ(ω)|H^2|ψ(ω)〉
        h_squared = ComposedOp([~StateFn(operator.oplist[0].primitive ** 2
                                        * operator.oplist[0].coeff ** 2), operator.oplist[-1]]
                              ).assign_parameters(dict(zip(self._parameters,
                                                           self._parameter_values)))

        if self._backend is not None:
            h_squared = CircuitSampler(self._backend).convert(h_squared)

        h_squared = np.real(h_squared.eval())
        eps_squared += h_squared

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dotdot = self._inner_prod(ng_res, np.dot(metric, ng_res))
        eps_squared += dotdot

        # 2Im⟨dtψ(ω)| H | ψ(ω)〉= 2Im dtω⟨dωψ(ω)|H | ψ(ω)
        # 2 missing b.c. of Im
        imgrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared -= imgrad2

        return eps_squared

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
            grad_res: -2Im⟨dψ(ω)/dω|H|ψ(ω)〉
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

        # 2Im⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared = grad_res

        # print('E_t squared', np.round(eps_squared, 4))
        return grad_eps_squared

    def _distance_energy(self,
                         time: Union[float, complex],
                         trained_param_dict: Dict) -> (float, float, float):
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
        # H
        hermitian_op = self._operator[0].primitive.to_matrix_op().primitive.data * \
                       self._operator[0].primitive.to_matrix_op().coeff
        # exp(-iHt)
        evolution_op = expm(-1j * hermitian_op * time)
        #|state>
        state = self._operator[-1]
        # |state_0>
        init_state = state.assign_parameters(dict(zip(trained_param_dict.keys(),
                                                      self._init_parameter_values)))
        # |state_t>
        trained_state = state.assign_parameters(trained_param_dict)

        # Convert the operator with the CircuitSampler
        if self._backend is not None:
            init_state = CircuitSampler(self._backend).convert(init_state)
            trained_state = CircuitSampler(self._backend).convert(trained_state)

        target_state = np.dot(evolution_op, init_state.eval().primitive.data)
        trained_state = trained_state.eval().primitive.data

        # Fidelity
        f = state_fidelity(target_state, trained_state)
        # Actual error
        act_err = np.linalg.norm(target_state - trained_state, ord=2)
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(hermitian_op, target_state))
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(hermitian_op, trained_state))
        return f, act_err, act_en, trained_en
