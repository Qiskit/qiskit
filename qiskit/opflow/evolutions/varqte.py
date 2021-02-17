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

from abc import abstractmethod
from typing import List, Optional, Union, Dict, Iterable

import numpy as np
import os
import csv

from scipy.integrate import OdeSolver

from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance

from qiskit.circuit import ParameterExpression, ParameterVector

from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow import StateFn, ListOp, CircuitSampler
from qiskit.opflow.gradients import CircuitQFI, CircuitGradient, Gradient, QFI, \
    NaturalGradient

class VarQTE(EvolutionBase):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191

    Algorithms that use McLachlans variational principle to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and quantum state.
    """
    def __init__(self,
                 grad_method: Union[str, CircuitGradient] = 'lin_comb',
                 qfi_method: Union[str, CircuitQFI] = 'lin_comb_full',
                 regularization: Optional[str] = None,
                 num_time_steps: int = 10,
                 parameters: Optional[Union[ParameterExpression, List[ParameterExpression],
                                            ParameterVector]] = None,
                 init_parameter_values: Optional[Union[List, np.ndarray]] = None,
                 ode_solver: Optional[OdeSolver] = None,
                 backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
                 get_error: bool = False,
                 get_h_terms: bool = False,
                 fidelity_to_target: bool = False,
                 snapshot_dir: Optional[str] = None,
                 **kwargs):
        r"""
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                If regularization is None but the metric is ill-conditioned or singular then
                a least square solver is used without regularization
            ode_solver: ODE Solver for y'=f(t,y) with parameters - f(callable), jac(callable): df/dy
                        f to be given as dummy
            snapshot_dir: Directory in to which to store cvs file with parameters,
                if None (default) then no cvs file is created.
            kwargs (dict): Optional parameters for a CircuitGradient
        """
        super().__init__()
        self._grad_method = grad_method
        self._qfi_method = qfi_method
        # TODO enable the use of custom regularization methods
        self._regularization = regularization
        self._epsilon = kwargs.get('epsilon', 1e-6)
        self._num_time_steps = num_time_steps
        if len(parameters) == 0:
            raise TypeError('Please provide parameters for the variational quantum time evolution.')
        self._parameters = parameters
        if init_parameter_values is not None:
            self._parameter_values = init_parameter_values
        else:
            self._parameter_values = np.random.random(len(parameters))
        self._backend = backend
        self._get_error = get_error
        self._get_h_terms = get_h_terms
        self._ode_solver = ode_solver
        self._snapshot_dir = snapshot_dir
        if snapshot_dir:
            if not os.path.exists(snapshot_dir):
                os.mkdir(snapshot_dir)
            if not os.path.exists(os.path.join(self._snapshot_dir, 'varqte_output.csv')):
                with open(os.path.join(self._snapshot_dir, 'varqte_output.csv'), mode='w') as \
                        csv_file:
                    fieldnames = ['t', 'params', 'num_params', 'num_time_steps']
                    if self._get_error:
                        fieldnames.extend(['error_bound', 'error_grad', 'resid'])
                    if fidelity_to_target:
                        fieldnames.extend(['fidelity', 'true_error', 'target_energy',
                                           'trained_energy'])
                    if get_h_terms:
                        fieldnames.extend(['h_norm', 'h_squared', 'dtdt_trained',
                                           're_im_grad', 'h_norm_factor', 'h_squared_factor',
                                           'trained_energy_factor'])
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        if fidelity_to_target:
            self._init_parameter_values = init_parameter_values
        else:
            self._init_parameter_values = None
        self._operator = None
        self._nat_grad = None
        self._metric = None
        self._grad = None

    @abstractmethod
    def convert(self,
                operator: ListOp) -> StateFn:
        """
        Apply Variational Quantum Time Evolution (VarQTE) w.r.t. the given operator
        Args:
            operator:
                Operator used vor Variational Quantum Imaginary or Real Time Evolution
                (VarQITE/VarQRTE)
                The coefficient of the operator determines the evolution time.
                If the coefficient is real/imaginary this method implements VarQITE/VarQRTE.

                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.

        Returns:
            StateFn (parameters are bound) which represents an approximation to the respective
            time evolution.

        """
        raise NotImplementedError

    def _init_grad_objects(self):
        # Adapt coefficients for the real part for McLachlan with 0.5
        # True and needed!! Double checked

        # VarQRTE
        if np.iscomplex(self._operator.coeff):
            self._nat_grad = NaturalGradient(grad_method=self._grad_method,
                                             qfi_method=self._qfi_method,
                                             regularization=self._regularization
                                            ).convert(self._operator * 0.5, self._parameters)
        # VarQITE
        else:
            self._nat_grad = NaturalGradient(grad_method=self._grad_method,
                                             qfi_method=self._qfi_method,
                                             regularization=self._regularization
                                             ).convert(self._operator * -0.5, self._parameters)

        # self._grad = Gradient(self._grad_method, epsilon=self._epsilon).convert(self._operator,
        #                                                                         self._parameters)
        self._grad = Gradient(self._grad_method).convert(self._operator, self._parameters)

        self._metric = QFI(self._qfi_method).convert(self._operator.oplist[-1], self._parameters)

        if self._backend is not None:
            # self._nat_grad = CircuitSampler(self._backend).convert(self._nat_grad)
            self._metric = CircuitSampler(self._backend).convert(self._metric)
            self._grad = CircuitSampler(self._backend).convert(self._grad)

    def _inner_prod(self,
                    x: Iterable,
                    y: Iterable) -> Union[np.ndarray, np.complex, np.float]:
        """
        Compute the inner product of two vectors
        Args:
            x: vector
            y: vector

        Returns: Inner product of x,y
        """
        return np.matmul(np.conj(np.transpose(x)), y)

    def _store_params(self,
                      t: int,
                      params: List,
                      error_bound: Optional[float] = None,
                      error_grad: Optional[float] = None,
                      resid: Optional[float] = None,
                      fidelity_to_target: Optional[float] = None,
                      true_error: Optional[float] = None,
                      target_energy: Optional[float] = None,
                      trained_energy: Optional[float] = None,
                      h_norm: Optional[float] = None,
                      h_squared: Optional[float] = None,
                      dtdt_trained: Optional[float] = None,
                      re_im_grad: Optional[float] = None,
                      h_norm_factor: Optional[float] = None,
                      h_squared_factor: Optional[float] = None,
                      trained_energy_factor: Optional[float] = None):
        """
        Args:
            t: current point in time evolution
            error_bound: ||\epsilon_t|| >= |||target> - |trained>||_2
                         (using exact |trained_energy - target_energy|)
            error_grad: ||e_t||
            resid: residual of McLachlan's SLE||Adtω-C||
            params: Current parameter values
            fidelity_to_target: fidelity between trained and target trained |<target|trained>|^2
            true_error: |||target> - |trained>||_2
            target_energy: <target|H|target>
            trained_energy: <trained|H|trained>
            h_norm: ||H||_2
            h_squared: <trained|H^2|trained>
            dtdt_trained: <dt_trained|dt_trained>
            re_im_grad: Re(<dt_trained|H|trained>) for VarQITE resp.
                        Im(<dt_trained|H|trained>) for VarQRTE
            h_norm_factor: (1 + 2 \delta_t ||H||)^{T - t}
            h_squared_factor: (1 + 2 \delta_t \sqrt{|<trained|H^2|trained>|})
            trained_energy_factor: (1 + 2 \delta_t |<trained|H|trained>|)

        Returns:

        """
        with open(os.path.join(self._snapshot_dir, 'varqte_output.csv'), mode='a') as csv_file:
            fieldnames = ['t', 'params', 'num_params', 'num_time_steps']
            if self._get_error:
                fieldnames.extend(['error_bound', 'error_grad', 'resid'])
            if fidelity_to_target is not None:
                fieldnames.extend(['fidelity', 'true_error', 'target_energy', 'trained_energy'])
            if self._get_h_terms:
                fieldnames.extend(['h_norm', 'h_squared', 'dtdt_trained',
                                   're_im_grad', 'h_norm_factor', 'h_squared_factor',
                                   'trained_energy_factor'])
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if fidelity_to_target is None:
                if not self._get_error:
                    writer.writerow({'t': t, 'params': np.round(params, 8),
                                     'num_params': len(params),
                                     'num_time_steps': self._num_time_steps})
                else:
                    writer.writerow({'t': t, 'params': np.round(params, 8),
                                     'num_params': len(params),
                                     'num_time_steps':self._num_time_steps,
                                     'error_bound': np.round(error_bound, 8),
                                     'error_grad': np.round(error_grad, 8),
                                     'resid': np.round(resid, 8)})
            else:
                if not self._get_error:
                    writer.writerow({'t': t,
                                     'params': np.round(params, 8),
                                     'num_params': len(params),
                                     'num_time_steps': self._num_time_steps,
                                     'fidelity': np.round(fidelity_to_target, 8),
                                     'true_error': np.round(true_error, 8),
                                     'target_energy': np.round(target_energy, 8),
                                     'trained_energy': np.round(trained_energy, 8)})
                else:
                    if not self._get_h_terms:
                        writer.writerow({'t': t,
                                         'params': np.round(params, 8),
                                         'num_params': len(params),
                                         'num_time_steps': self._num_time_steps,
                                         'error_bound': np.round(error_bound, 8),
                                         'error_grad': np.round(error_grad, 8),
                                         'resid': np.round(resid, 8),
                                         'fidelity': np.round(fidelity_to_target, 8),
                                         'true_error': np.round(true_error, 8),
                                         'target_energy': np.round(target_energy, 8),
                                         'trained_energy': np.round(trained_energy, 8)})
                    else:
                        writer.writerow({'t': t,
                                         'params': np.round(params, 8),
                                         'num_params': len(params),
                                         'num_time_steps': self._num_time_steps,
                                         'error_bound': np.round(error_bound, 8),
                                         'error_grad': np.round(error_grad, 8),
                                         'resid': np.round(resid, 8),
                                         'fidelity': np.round(fidelity_to_target, 8),
                                         'true_error': np.round(true_error, 8),
                                         'target_energy': np.round(target_energy, 8),
                                         'trained_energy': np.round(trained_energy, 8),
                                         'h_norm': np.round(h_norm, 8),
                                         'h_squared': np.round(h_squared, 8),
                                         'dtdt_trained': np.round(dtdt_trained, 8),
                                         're_im_grad': np.round(re_im_grad, 8),
                                         'h_norm_factor': np.round(h_norm_factor, 8),
                                         'h_squared_factor': np.round(h_squared_factor, 8),
                                         'trained_energy_factor': np.round(trained_energy_factor, 8)
                                         })

    def _propagate(self,
                   param_dict: Dict
                   ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """

        Args:
            param_dict:

        Returns:

        """

        if self._grad is None:
            self._init_grad_objects()

        nat_grad_result = self._nat_grad.assign_parameters(param_dict).eval()
        print('nat grad result', nat_grad_result)
        # Get the gradient of <H> w.r.t. the variational parameters
        grad_res = self._grad.assign_parameters(param_dict).eval()
        # Get the QFI/4
        metric_res = self._metric.assign_parameters(param_dict).eval() * 0.25

        w, v = np.linalg.eig(metric_res)
        # if not all(ew >= 0 for ew in w):
        #     nat_grad, resids, _, _ = np.linalg.lstsq(a, c, rcond=1e-5)
        #     return nat_grad

        if not all(ew >= -1e-8 for ew in w):
            raise Warning('The underlying metric has ein Eigenvalue < ', -1e-8, '. '
                                                                                'Please use a '
                                                                                'regularized '
                                                                                'least-square '
                                                                                'solver for this '
                                                                                'problem.')
        if not all(ew >= 0 for ew in w):
            w = [max(0, ew) for ew in w]
            metric_res = v @ np.diag(w) @ np.linalg.inv(v)
        # Get the time derivative of the variational parameters
        # VarQRTE
        # if np.iscomplex(self._operator.coeff):
        #     nat_grad_result = np.real(NaturalGradient(regularization=self._regularization
        #                                               ).compute_with_res(metric_res * 4,
        #                                                                  grad_res * 0.5))
        # # VarQITE
        # else:
        #     nat_grad_result = np.real(NaturalGradient(regularization=self._regularization
        #                                               ).compute_with_res(metric_res * 4,
        #                                                                  grad_res * (-0.5)))
        if any(np.abs(np.imag(grad_res_item)) > 1e-8 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-8 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the gradient are non-negligible.')

        return np.real(nat_grad_result), np.real(grad_res), np.real(metric_res)

