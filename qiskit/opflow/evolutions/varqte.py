# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
import warnings

from pathlib import Path


from scipy.integrate import OdeSolver, ode, solve_ivp
from scipy.optimize import least_squares, minimize

from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance

from qiskit.circuit import ParameterExpression, ParameterVector

from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow import StateFn, ListOp, CircuitSampler, ComposedOp
from qiskit.opflow.gradients import CircuitQFI, CircuitGradient, Gradient, QFI, \
    NaturalGradient

from qiskit.quantum_info import state_fidelity

from qiskit.working_files.varQTE.implicit_euler import BDF, backward_euler_fsolve


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
                 ode_solver: Optional[Union[OdeSolver, ode]] = None,
                 backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
                 snapshot_dir: Optional[str] = None,
                 faster: bool = True,
                 error_based_ode: bool = False,
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
            self._init_parameter_values = init_parameter_values
        else:
            self._init_parameter_values = np.random.random(len(parameters))
        self._backend = backend
        if self._backend is not None:
            self._operator_circ_sampler = CircuitSampler(self._backend)
            self._state_circ_sampler = CircuitSampler(self._backend)
            self._h_squared_circ_sampler = CircuitSampler(self._backend)
            self._grad_circ_sampler = CircuitSampler(self._backend)
            self._metric_circ_sampler = CircuitSampler(self._backend)
            if not faster:
                self._nat_grad_circ_sampler = CircuitSampler(self._backend)
        self._faster = faster
        self._ode_solver = ode_solver
        if self._ode_solver is None:
            self._ode_solver = ForwardEuler
        self._snapshot_dir = snapshot_dir
        if snapshot_dir:
            # def ensure_dir(file_path):
            #     directory = os.path.dirname(file_path)
            #     if not os.path.exists(directory):
            #         # os.makedirs(directory)
            #         Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            # ensure_dir(snapshot_dir)
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            # if not os.path.exists(snapshot_dir):
            #     os.makedirs(snapshot_dir)
            if not os.path.exists(os.path.join(self._snapshot_dir, 'varqte_output.csv')):
                with open(os.path.join(self._snapshot_dir, 'varqte_output.csv'), mode='w') as \
                        csv_file:
                    fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                                  'error_bound_factor',
                                  'error_grad', 'resid', 'fidelity', 'true_error',
                                  'phase_agnostic_true_error',
                                  'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                                  'trained_energy', 'energy_error', 'h_norm', 'h_squared',
                                  'variance', 'dtdt_trained', 're_im_grad']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        self._operator = None
        self._nat_grad = None
        self._metric = None
        self._grad = None
        # Current gradient error
        # self._et = 0
        self._error_based_ode = error_based_ode

        self._storage_params_tbd = None

        self._exact_euler_state = None
        self._store_now = False

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

        Raises: NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def _exact_state(self,
                     time: Union[float, complex]) -> Iterable:
        """

        Args:
            time: current time

        Raises: NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def _get_error_bound(self,
                     gradient_errors: List,
                     time_steps: List,
                     stddevs: List) -> List:
        """

        Raises: NotImplementedError

        """
        raise NotImplementedError

    def error_bound(self,
                    data_dir: str,
                    use_integral_approx: bool = True,
                    imag_reverse_bound: bool = True,
                    H: Optional[Union[List, np.ndarray]] = None) -> List:
        # Read data
        # if time already in data skip
        with open(os.path.join(data_dir, 'varqte_output.csv'), mode='r') as csv_file:
            fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                          'error_bound_factor',
                          'error_grad', 'resid', 'fidelity', 'true_error',
                          'phase_agnostic_true_error',
                          'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                          'trained_energy', 'energy_error', 'h_norm', 'h_squared',
                          'variance', 'dtdt_trained', 're_im_grad']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            first = True
            grad_errors = []
            time = []
            time_steps = []
            stddevs = []
            for line in reader:
                if first:
                    first = False
                    continue
                t_line = float(line['t'])
                if t_line in time:
                    continue
                time.append(t_line)
                grad_errors.append(float(line['error_grad']))
                # Method is not used.
                stddevs.append(np.sqrt(float(line['variance'])))

        zipped_lists = zip(time, grad_errors, stddevs)

        sorted_pairs = sorted(zipped_lists)

        triples = zip(*sorted_pairs)

        time, grad_errors, stddevs = [list(triple) for triple in triples]

        for j in range(len(time)-1):
            time_steps.append(time[j+1]-time[j])

        if not np.iscomplex(self._operator.coeff):
            e_bound = self._get_error_bound(grad_errors, time, stddevs,
                                            use_integral_approx, imag_reverse_bound, H)

        else:
            e_bound = self._get_error_bound(grad_errors, time, stddevs, use_integral_approx)
        return e_bound

    def _init_grad_objects(self):
        # Adapt coefficients for the real part for McLachlan with 0.5
        # True and needed!! Double checked

        self._state = self._operator[-1]
        if self._backend is not None:
            self._init_state = self._state_circ_sampler.convert(self._state,
                                                                params=dict(zip
                                                                            (self._parameters,
                                                                             self._init_parameter_values
                                                                             )
                                                                            )
                                                                )
        else:
            self._init_state = self._state.assign_parameters(dict(zip(self._parameters,
                                                                      self._init_parameter_values)))
        self._init_state = self._init_state.eval().primitive.data
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_matrix = self._h.to_matrix(massive=True)

        self._h_squared = ComposedOp([~StateFn(self._h ** 2), self._state])

        self._exact_euler_state = self._exact_state(0)

        if not self._faster:
            # # VarQRTE
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
            # VarQRTE
            # if np.iscomplex(self._operator.coeff):
            #     self._nat_grad = NaturalGradient(grad_method=self._grad_method,
            #                                      qfi_method=self._qfi_method,
            #                                      regularization=self._regularization
            #                                     ).convert(self._operator, self._parameters)
            # # VarQITE
            # else:
            #     self._nat_grad = NaturalGradient(grad_method=self._grad_method,
            #                                      qfi_method=self._qfi_method,
            #                                      regularization=self._regularization
            #                                      ).convert(self._operator * -1, self._parameters)

        self._grad = Gradient(self._grad_method).convert(self._operator, self._parameters)

        self._metric = QFI(self._qfi_method).convert(self._operator.oplist[-1], self._parameters)

    def _init_ode_solver(self, t: float,
                         init_params: Union[List, np.ndarray]):
        """
        Initialize ODE Solver
        Args:
            t: Evolution time
            init_params: Set of initial parameters for time 0
        """
        def error_based_ode_fun(time, params):
            param_dict = dict(zip(self._parameters, params))
            nat_grad_result, grad_res, metric_res = self._propagate(param_dict)
            # w, v = np.linalg.eig(metric_res)
            #
            # if not all(ew >= -1e-8 for ew in w):
            #     raise Warning('The underlying metric has ein Eigenvalue < ', -1e-8,
            #                   '. Please use a regularized least-square solver for this problem.')
            # if not all(ew >= 0 for ew in w):
            #     w = [max(0, ew) for ew in w]
            #     metric_res = v @ np.diag(w) @ np.linalg.inv(v)

            def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
                """
                Search for the dω/dt which minimizes ||e_t||^2

                Args:
                    dt_param_values: values for dω/dt
                Returns:
                    ||e_t||^2 for given for dω/dt

                """
                et_squared = self._error_t(params, dt_param_values, grad_res,
                                           metric_res)[0]
                print('grad error', et_squared)
                return et_squared

            def jac_argmin_fun(dt_param_values: Union[List, np.ndarray]
                               ) -> Union[List, np.ndarray]:
                """
                Get tge gradient of ||e_t||^2 w.r.t. dω/dt for given values

                Args:
                    dt_param_values: values for dω/dt
                Returns:
                    Gradient of ||e_t||^2 w.r.t. dω/dt

                """
                dw_et_squared = self._grad_error_t(dt_param_values, grad_res,
                                                   metric_res)
                return dw_et_squared

            # return nat_grad_result
            # Use the natural gradient result as initial point for least squares solver
            # argmin = least_squares(fun=argmin_fun, x0=nat_grad_result,
            #                        ftol=1e-8)
            print('initial natural gradient result', nat_grad_result)
            argmin = minimize(fun=argmin_fun, x0=nat_grad_result,
                              method='COBYLA', tol=1e-6)
            # argmin = minimize(fun=argmin_fun, x0=nat_grad_result, jac=jac_argmin_fun,
            #                   method='CG', tol=1e-8)
            #

            print('final dt_omega', np.real(argmin.x))
            # self._et = argmin_fun(argmin.x)
            return argmin.x, grad_res, metric_res

        # Use either the argument that minimizes the gradient error or the result from McLachlan's
        # variational principle to run the ODE solver
        def ode_fun(t: float, params: Iterable) -> Iterable:
            param_dict = dict(zip(self._parameters, params))
            if self._error_based_ode:
                dt_params, grad_res, metric_res = error_based_ode_fun(t, params)
            else:
                dt_params, grad_res, metric_res = self._propagate(param_dict)
            if (self._snapshot_dir is not None) and (self._store_now):
                # Get the residual for McLachlan's Variational Principle
                # self._storage_params_tbd = (t, params, et, resid, f, true_error, true_energy,
                #                             trained_energy, h_squared, dtdt_state, reimgrad)
                    if np.iscomplex(self._operator.coeff):
                        # VarQRTE
                        resid = np.linalg.norm(np.matmul(metric_res, dt_params) - grad_res * 0.5)
                    else:
                        # VarQITE
                        resid = np.linalg.norm(np.matmul(metric_res, dt_params) + grad_res * 0.5)
                    # Get the error for the current step
                    et, h_squared, dtdt_state, reimgrad = self._error_t(params, dt_params, grad_res,
                                                                        metric_res)[:4]
                    print('returned et', et)
                    try:
                        if et < 0 and np.abs(et) > 1e-4:
                            raise Warning('Non-neglectible negative et observed')
                        else:
                            et = np.sqrt(np.real(et))
                    except Exception:
                        et = 1000
                    print('after try except', et)
                    f, true_error, phase_agnostic_true_error, true_energy, trained_energy = \
                        self._distance_energy(t, param_dict)
                    self._store_params(t, params, None, None, et,
                                       resid, f, true_error, phase_agnostic_true_error, None,
                                       None, true_energy,
                                       trained_energy, None, h_squared, dtdt_state, reimgrad)
            return dt_params

        if issubclass(self._ode_solver, OdeSolver):
            self._ode_solver=self._ode_solver(ode_fun, t_bound=t, t0=0, y0=init_params,
                                              atol=1e-6)

        elif self._ode_solver == ForwardEuler:
            self._ode_solver = self._ode_solver(ode_fun, t_bound=t, t0=0,
                                                y0=init_params,
                                                num_t_steps=self._num_time_steps)

        # elif self._ode_solver == backward_euler_fsolve:
        #     self._ode_solver = self._ode_solver(ode_fun,  tspan=(0, t), y0=self._parameter_values,
        #                                         n=self._num_time_steps)
        else:
            raise TypeError('Please define a valid ODESolver')
        return

    def _run_ode_solver(self, t, init_params):
        """
        Find numerical solution with ODE Solver
        Args:
            t: Evolution time
            init_params: Set of initial parameters for time 0
        """
        self._init_ode_solver(t, init_params)
        # param_values = self._parameter_values
        if isinstance(self._ode_solver, OdeSolver) or isinstance(self._ode_solver, ForwardEuler):
            self._store_now = True
            _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
            while self._ode_solver.t < t:
                self._store_now = False
                self._ode_solver.step()
                if self._snapshot_dir is not None and self._ode_solver.t <= t:
                    self._store_now = True
                    _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
                # if self._snapshot_dir is not None:
                #     self._ode_solver_store(param_values, t_old)
                # if self._et is None:
                #     warnings.warn('Time evolution failed.')
                #     break
                # if self._snapshot_dir is not None:
                #     time, params, et, resid, f, true_error, true_energy, trained_energy, \
                #     h_squared, dtdt_state, reimgrad = self._storage_params_tbd
                #     self._store_params(time, params, None, None, et,
                #                        resid, f, true_error,  phase_agnostic_true_error, None,
                #                        None, true_energy,
                #                        trained_energy, None, h_squared, dtdt_state, reimgrad)
                print('ode time', self._ode_solver.t)
                param_values = self._ode_solver.y
                print('ode parameters', self._ode_solver.y)
                print('ode step size', self._ode_solver.step_size)
                if self._ode_solver.status == 'finished':
                    break
                elif self._ode_solver.status == 'failed':
                    raise Warning('ODESolver failed')
            # self._et = 0
            # if self._snapshot_dir is not None:
            #     self._ode_solver_store(param_values, self._ode_solver.t)

        # elif isinstance(self._ode_solver, backward_euler_fsolve):
        #     while self._ode_solver._n_count < self._ode_solver._n:
        #         param_values = self._ode_solver.step()
                # if self._snapshot_dir is not None:
                #     self._ode_solver_store(param_values, self._ode_solver._t[-2])
                # param_values = param_values_new
                # if self._et is None:
                #     warnings.warn('Time evolution failed.')
                #     break

            # if self._snapshot_dir is not None:
            #     self._ode_solver_store(param_values, self._ode_solver._t[-1])

        else:
            raise TypeError('Please provide a scipy ODESolver or ode type object.')
        print('Parameter Values ', param_values)

        return param_values

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
                      error_bound_factor: Optional[float] = None,
                      error_grad: Optional[float] = None,
                      resid: Optional[float] = None,
                      fidelity_to_target: Optional[float] = None,
                      true_error: Optional[float] = None,
                      phase_agnostic_true_error: Optional[float] = None,
                      true_to_euler_error: Optional[float] = None,
                      trained_to_euler_error: Optional[float] = None,
                      target_energy: Optional[float] = None,
                      trained_energy: Optional[float] = None,
                      h_norm: Optional[float] = None,
                      h_squared: Optional[float] = None,
                      dtdt_trained: Optional[float] = None,
                      re_im_grad: Optional[float] = None):
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
            phase_agnostic_true_error: min_phi|||target> - exp(i*phi)|trained>||_2
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
            fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                          'error_bound_factor',
                          'error_grad', 'resid', 'fidelity', 'true_error',
                          'phase_agnostic_true_error', 'true_to_euler_error',
                          'trained_to_euler_error', 'target_energy', 'trained_energy',
                          'energy_error', 'h_norm', 'h_squared', 'variance',
                          'dtdt_trained', 're_im_grad']

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            variance = None
            if h_squared is not None:
                variance = h_squared - trained_energy**2
            writer.writerow({'t': t,
                             'params': np.round(params, 8),
                             'num_params': len(params),
                             'num_time_steps': self._num_time_steps,
                             'error_bound': error_bound,
                             'error_bound_factor': error_bound_factor,
                             'error_grad': error_grad,
                             'resid': resid,
                             'fidelity': np.round(fidelity_to_target, 8),
                             'true_error': np.round(true_error, 8),
                             'phase_agnostic_true_error': np.round(phase_agnostic_true_error, 8),
                             'true_to_euler_error': true_to_euler_error,
                             'trained_to_euler_error': trained_to_euler_error,
                             'target_energy': np.round(target_energy, 8),
                             'trained_energy': np.round(trained_energy, 8),
                             'energy_error': np.round(trained_energy - target_energy, 8),
                             'h_norm': h_norm,
                             'h_squared': h_squared,
                             'variance': variance,
                             'dtdt_trained': dtdt_trained,
                             're_im_grad': re_im_grad
                             })

    def _propagate(self,
                   param_dict: Dict
                   ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """

        Args:
            param_dict:

        Returns:

        """
        if self._backend is not None:
            grad_res = np.array(self._grad_circ_sampler.convert(self._grad,
                                                                params=param_dict).eval())
            # Get the QFI/4
            metric_res = np.array(self._metric_circ_sampler.convert(self._metric,
                                                           params=param_dict).eval()) * 0.25
        else:
            grad_res = np.array(self._grad.assign_parameters(param_dict).eval())
            # Get the QFI/4
            metric_res = np.array(self._metric.assign_parameters(param_dict).eval()) * 0.25

        w, v = np.linalg.eig(metric_res)

        if not all(ew >= -1e-8 for ew in w):
            raise Warning('The underlying metric has ein Eigenvalue < ', -1e-8,
                          '. Please use a regularized least-square solver for this problem.')
        if not all(ew >= 0 for ew in w):
            w = [max(0, ew) for ew in w]
            metric_res = v @ np.diag(w) @ np.linalg.inv(v)

        if self._faster:
            if np.iscomplex(self._operator.coeff):
                # VarQRTE
                nat_grad_result = NaturalGradient.nat_grad_combo_fn(x=[grad_res * 0.5,
                                                                       metric_res],
                                                                    regularization=
                                                                    self._regularization)
            else:
                # VarQITE
                nat_grad_result = NaturalGradient.nat_grad_combo_fn(x=[grad_res * -0.5,
                                                                       metric_res],
                                                                    regularization=
                                                                    self._regularization)
        else:
            if self._backend is not None:
                nat_grad_result = \
                    self._nat_grad_circ_sampler.convert(self._nat_grad, params=param_dict).eval()
            else:
                nat_grad_result = self._nat_grad.assign_parameters(param_dict).eval()
        print('nat grad result', nat_grad_result)

        if any(np.abs(np.imag(grad_res_item)) > 1e-8 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-8 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the gradient are non-negligible.')

        return np.real(nat_grad_result), np.real(grad_res), np.real(metric_res)

    def _l2_norm_phase_agnostic(self,
                                target_state: Union[List, np.ndarray],
                                trained_state: Union[List, np.ndarray]) -> float:
        """
        Find a global phase agnostic l2 norm between the target and trained state
        Args:
            target_state: Target state
            trained_state: Trained state with potential phase mismatch

        Returns:
            global phase agnostic l2 norm value

        """
        def phase_agnostic(phi):
            return np.linalg.norm(np.subtract(target_state, np.exp(1j*phi) * trained_state), ord=2)
        l2_norm_pa = minimize(fun=phase_agnostic, x0=0, method='COBYLA', tol=1e-6)
        return l2_norm_pa.fun

    def _distance_energy(self,
                         time: Union[float, complex],
                         param_dict: Dict) -> (float, float, float):
        """
        Evaluate the fidelity to the target state, the energy w.r.t. the target state and
        the energy w.r.t. the trained state for a given time and the current parameter set
        Args:
            time: current evolution time
            param_dict: dictionary which matches the operator parameters to the current
            values of parameters for the given time
        Returns: fidelity to the target state, the energy w.r.t. the target state and
        the energy w.r.t. the trained state
        """

        # |state_t>
        if self._backend is not None:
            trained_state = self._state_circ_sampler.convert(self._state,
                                                             params=param_dict)
        else:
            trained_state = self._state.assign_parameters(param_dict)
        trained_state = trained_state.eval().primitive.data
        target_state = self._exact_state(time)

        # Fidelity
        f = state_fidelity(target_state, trained_state)
        # Actual error
        act_err = np.linalg.norm(np.subtract(target_state, trained_state), ord=2)
        phase_agnostic_act_err = self._l2_norm_phase_agnostic(target_state, trained_state)
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(self._h_matrix, target_state))
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(self._h_matrix, trained_state))
        print('Fidelity', f)
        print('True error', act_err)
        print('Global phase agnostic true error', phase_agnostic_act_err)
        print('actual energy', act_en)
        print('trained_en', trained_en)
        # Error between exact evolution and Euler evolution using exact gradients
        # exact_exact_euler_err = np.linalg.norm(target_state - self._exact_euler_state, ord=2)
        # Error between Euler evolution using exact gradients and the trained state
        # exact_euler_trained_err = np.linalg.norm(trained_state - self._exact_euler_state, ord=2)
        # return f, act_err, act_en, trained_en, exact_exact_euler_err, exact_euler_trained_err
        return f, act_err, phase_agnostic_act_err, np.real(act_en), np.real(trained_en)

    @staticmethod
    def plot_results(data_directories: List[str],
                     error_bound_directories: List[str],
                     reverse_bound_directories: Optional[List[str]] = None):
        """

        Args:
            data_directories:
            error_bound_directories:

        Returns:

        """
        if len(data_directories) != len(error_bound_directories):
            raise Warning('Please provide data and error bound directories of the same length '
                          'corresponding to the same data.')
        import matplotlib.pyplot as plt
        for j, data_dir in enumerate(data_directories):
            with open(os.path.join(data_dir, 'varqte_output.csv'), mode='r') as csv_file:
                fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                              'error_bound_factor',
                              'error_grad', 'resid', 'fidelity', 'true_error',
                              'phase_agnostic_true_error',
                              'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                              'trained_energy', 'energy_error', 'h_norm', 'h_squared',
                              'variance', 'dtdt_trained', 're_im_grad']
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)
                first = True
                grad_errors = []
                true_error = []
                phase_agnostic_true_error = []
                time = []
                time_steps = []
                fid = []
                true_energy = []
                trained_energy = []
                stddevs = []

                for line in reader:
                    if first:
                        first = False
                        continue
                    t_line = float(line['t'])
                    if t_line in time:
                        continue
                    time.append(t_line)
                    fid.append(float(line['fidelity']))
                    grad_errors.append(float(line['error_grad']))
                    true_error.append(float(line['true_error']))
                    phase_agnostic_true_error.append(float(line['phase_agnostic_true_error']))
                    true_energy.append(float(line['target_energy']))
                    trained_energy.append(float(line['trained_energy']))
                    stddevs.append(np.sqrt(float(line['variance'])))

                zipped_lists = zip(time, grad_errors, true_error, fid, true_energy, trained_energy)

                sorted_pairs = sorted(zipped_lists)

                zipped_sorted = zip(*sorted_pairs)

                time, grad_errors, true_error, fid, true_energy, \
                trained_energy = [list(zipped_items) for zipped_items in zipped_sorted]

            error_bounds = np.load(error_bound_directories[j])

            if reverse_bound_directories is not None:
                reverse_error_bounds = np.load(reverse_bound_directories[j])

            plt.figure(1)
            plt.title('Actual Error and Error Bound ')
            plt.scatter(time, error_bounds, color='blue', marker='o', s=40,
                        label='error bound')

            plt.scatter(time, true_error, color='purple', marker='x', s=40, label='true error')
            plt.scatter(time, phase_agnostic_true_error, color='orange', marker='x', s=40,
                        label='phase agnostic true error')
            plt.legend(loc='best')
            plt.xlabel('time')
            plt.ylabel('error')
            # plt.xticks(range(counter-1))
            plt.savefig(os.path.join(data_dir, 'error_bound_actual.png'))
            if reverse_bound_directories is not None:
                plt.scatter(time, reverse_error_bounds, color='green', marker='o', s=40,
                            label='reverse error bound')
                plt.legend(loc='best')
                plt.savefig(os.path.join(data_dir, 'error_bound_actual_reverse.png'))
            plt.close()

            plt.figure(2)
            plt.title('Fidelity')
            plt.scatter(time, fid, color='grey', marker='o', s=40,
                        label='fidelity')
            plt.xlabel('time')
            plt.ylabel('fidelity')
            # plt.xticks(range(counter-1))
            plt.savefig(os.path.join(data_dir, 'fidelity.png'))
            plt.close()

            plt.figure(3)
            plt.title('Energy')
            plt.scatter(time, true_energy, color='black', marker='o', s=40,
                        label='true energy')
            plt.scatter(time, trained_energy, color='grey', marker='x', s=40,
                        label='trained energy')
            plt.xlabel('time')
            plt.ylabel('energy')
            plt.legend(loc='best')
            # plt.xticks(range(counter-1))
            plt.savefig(os.path.join(data_dir, 'energy.png'))
            plt.close()



        return


class ForwardEuler:

    def __init__(self,
                 fun: callable,
                 t0: float,
                 y0: Iterable,
                 t_bound: float,
                 num_t_steps: int):
            """Forward Euler ODE solver

            Parameters
            ----------
            fun : callable
                Right-hand side of the system. The calling signature is ``fun(t, y)``.
                Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
                It can either have shape (n,); then ``fun`` must return array_like with
                shape (n,). Alternatively it can have shape (n, k); then ``fun``
                must return an array_like with shape (n, k), i.e., each column
                corresponds to a single column in ``y``. The choice between the two
                options is determined by `vectorized` argument (see below). The
                vectorized implementation allows a faster approximation of the Jacobian
                by finite differences (required for this solver).
            t0 : float
                Initial time.
            y0 : array_like, shape (n,)
                Initial state.
            t_bound : float
                Boundary time - the integration won't continue beyond it. It also
                determines the direction of the integration.
            num_t_steps: int
                Number of time steps for forward Euler


            Attributes
            ----------

            status : string
                Current status of the solver: 'running' or 'finished'.
            t_bound : float
                Boundary time.
            t : float
                Current time.
            y : ndarray
                Current state.
            step_size : float
                Size of time steps
            """
            self.fun = fun
            self.t = t0
            self.y = y0
            self.t_bound = t_bound
            self.step_size = (t_bound - t0)/num_t_steps
            self.status = 'running'

    def step(self):
        self.y = list(np.add(self.y, self.step_size * self.fun(self.t, self.y)))
        self.t += self.step_size
        if self.t == self.t_bound:
            self.status = 'finished'



