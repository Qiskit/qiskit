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

import numpy as np
import scipy as sp
from scipy.linalg import expm

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
                                    ord=np.infty)
            print('h_norm', h_norm)

        # Step size
        dt = np.abs(operator.coeff)*np.sign(operator.coeff) / self._num_time_steps

        # Initialize error bound
        error = 0

        # Zip the parameter values to the parameter objects
        param_dict = dict(zip(self._parameters, self._parameter_values))

        for j in range(self._num_time_steps):
            # Get the natural gradient - time derivative of the variational parameters - and
            # the gradient w.r.t. H and the QFI/4.
            if self._ode_solver is None:
                nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
                # print('nat_grad_result', np.round(nat_grad_result, 3))
                print('C', np.round(grad_res, 3))
                print('metric', np.round(metric_res, 3))


                if self._get_error:
                    # Get the residual for McLachlan's Variational Principle
                    resid = np.linalg.norm(np.matmul(metric_res, nat_grad_result) + 0.5 * grad_res)
                    print('Residual norm', resid)

                    # Get the error for the current step
                    e_t = self._error_t(self._operator, nat_grad_result, grad_res, metric_res)

                    # TODO discuss (25) index of time step or time? I think indices
                    print('error before', error)
                    print('dt', dt)
                    print('et', e_t)
                    # print('factor', (1 + 2 * dt * h_norm) ** (self._num_time_steps - j))
                    # error += dt * e_t * (1 + 2 * dt * h_norm) ** (self._num_time_steps
                    #                                                         - j)

                    # error += dt * (e_t + 2 * h_norm)
                    # error += dt * e_t * (1 + 2 * dt * h_norm) ** (np.abs(operator.coeff) - (j * dt))



                # Propagate the Ansatz parameters step by step using explicit Euler
                # TODO enable the use of arbitrary ODE solvers
                # Subtract is correct either
                # omega_new = omega - A^(-1)Cdt or
                # omega_new = omega + A^(-1)((-1)*C)dt

                self._parameter_values = list(np.add(self._parameter_values, dt * np.real(
                                              nat_grad_result)))
                print('Params', self._parameter_values)
            else:
                self._ode_solver.step()
                self._parameter_values = self._ode_solver.y
                if self._ode_solver.status == 'finished' or self._ode_solver.status == 'failed':
                    break
                pass


            # Zip the parameter values to the parameter objects
            param_dict = dict(zip(self._parameters, self._parameter_values))
            if self._init_parameter_values is not None:
                # If initial parameter values were set compute the fidelity, the error between the
                # prepared and the target state, the energy w.r.t. the target state and the energy
                # w.r.t. the prepared state
                f, true_error, true_energy, trained_energy = self._distance_energy((j + 1) * dt,
                                                                            param_dict)
                print('Fidelity', f)
                print('True error', true_error)

                error += dt * (e_t + np.sqrt(np.linalg.norm(trained_energy - true_energy)))

            print('Error', np.round(error, 3), 'after', (j + 1) * dt)

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
        print('State', operator.oplist[-1].assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values))).eval())

        # print('H^2', (operator.oplist[0].primitive ** 2).eval() * operator.oplist[0].coeff ** 2)

        #TODO CircuitSampler
        # ⟨ψ(ω)|H^2|ψ(ω)〉
        h_squared = ComposedOp([~StateFn(operator.oplist[0].primitive ** 2) *
                                operator.oplist[0].coeff ** 2,
                                operator.oplist[-1]]).assign_parameters(dict(zip(self._parameters,
                                                                 self._parameter_values))).eval()
        print('h^2', np.round(h_squared, 6))
        eps_squared += h_squared

        # ⟨ψ(ω) | H | ψ(ω)〉^2
        exp = operator.assign_parameters(dict(zip(self._parameters, self._parameter_values))).eval()
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



        print('Grad error - real returned', np.round(np.sqrt(eps_squared), 6))
        return np.real(np.sqrt(eps_squared))

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
        act_err = np.sqrt(np.linalg.norm(target_state - trained_state, ord=2))
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(hermitian_op, target_state))
        print('actual energy', act_en)
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(hermitian_op, trained_state))
        print('trained_en', trained_en)
        return f, act_err, act_en, trained_en
