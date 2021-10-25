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
from typing import List, Optional, Union, Dict, Iterable, Tuple

import numpy as np
import scipy as sp
import os
import csv

import matplotlib.pyplot as plt

from pathlib import Path


from scipy.integrate import OdeSolver, ode, solve_ivp, RK45
from scipy.optimize import least_squares, minimize

from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance

from qiskit.circuit import ParameterExpression, ParameterVector

from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow import StateFn, ListOp, CircuitSampler, ComposedOp, PauliExpectation
from qiskit.opflow.gradients import CircuitQFI, CircuitGradient, Gradient, QFI, \
    NaturalGradient

from qiskit.quantum_info import state_fidelity


class ForwardEuler:

    def __init__(self,
                 fun: callable,
                 t0: float,
                 y0: Iterable,
                 t_bound: float,
                 num_t_steps: int):
        """
        Forward Euler ODE solver
        Args:
            fun :
                Right-hand side of the system. The calling signature is ``fun(t, y)``.
                Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
                It can either have shape (n,); then ``fun`` must return array_like with
                shape (n,). Alternatively it can have shape (n, k); then ``fun``
                must return an array_like with shape (n, k), i.e., each column
                corresponds to a single column in ``y``. The choice between the two
                options is determined by `vectorized` argument (see below). The
                vectorized implementation allows a faster approximation of the Jacobian
                by finite differences (required for this solver).
            t0 :
                Initial time.
            y0 :shape (n,)
                Initial state.
            t_bound :
                Boundary time - the integration won't continue beyond it. It also
                determines the direction of the integration.
            num_t_steps:
                Number of time steps for forward Euler

        Attributes:
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
        """
        Take an Euler step

        """
        self.y = list(np.add(self.y, self.step_size * self.fun(self.t, self.y)))
        self.t += self.step_size
        if self.t == self.t_bound:
            self.status = 'finished'


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
                 ode_solver: Optional[Union[OdeSolver, ode, ForwardEuler]] = None,
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
            num_time_steps: Number of time steps (deprecated if ode_solver is not ForwardEuler)
            parameters: Parameter objects for the parameters to be used for the time propagation
            init_parameter_values: Initial values for the parameters used for the time propagation
            ode_solver: ODE Solver for y'=f(t,y) with parameters - f(callable), jac(callable): df/dy
                        f to be given as dummy
            backend: Backend used to evaluate the quantum circuit outputs
            snapshot_dir: Directory in to which to store cvs file with parameters,
                if None (default) then no cvs file is created.
            faster: Use additional CircuitSampler objects to increase the processing speed
                    (deprecated if backend is None)
            error_based_ode: If False use McLachlan to get the parameter updates
                             If True use the argument that minimizes the error bounds
            kwargs (dict): Optional parameters for a CircuitGradient
        """
        super().__init__()
        self._grad_method = grad_method
        self._qfi_method = qfi_method
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
            #
            self._operator_circ_sampler = CircuitSampler(self._backend)
            self._state_circ_sampler = CircuitSampler(self._backend)
            self._h_squared_circ_sampler = CircuitSampler(self._backend)
            self._h_trip_circ_sampler = CircuitSampler(self._backend)
            self._grad_circ_sampler = CircuitSampler(self._backend)
            self._metric_circ_sampler = CircuitSampler(self._backend)
            if not faster:
                self._nat_grad_circ_sampler = CircuitSampler(self._backend, caching='all')
        self._faster = faster
        self._ode_solver = ode_solver
        if self._ode_solver is None:
            self._ode_solver = ForwardEuler
        self._snapshot_dir = snapshot_dir
        if snapshot_dir:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(os.path.join(self._snapshot_dir, 'varqte_output.csv')):
                with open(os.path.join(self._snapshot_dir, 'varqte_output.csv'), mode='w') as \
                        csv_file:
                    fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                                  'error_bound_grad',
                                  'error_grad', 'resid', 'fidelity', 'true_error',
                                  'phase_agnostic_true_error',
                                  'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                                  'trained_energy', 'energy_error', 'h_norm', 'h_squared', 'h_trip',
                                  'variance', 'dtdt_trained', 're_im_grad']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        self._operator = None
        self._nat_grad = None
        self._metric = None
        self._grad = None
        self._error_based_ode = error_based_ode

        self._storage_params_tbd = None
        self._store_now = False

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, op):
        self._operator = op
        #Initialize H Norm
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_matrix = self._h.to_matrix(massive=True)
        self._h_norm = np.linalg.norm(self._h_matrix, np.infty)
        return

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
                         time: List,
                         *args) -> Union[List, Tuple[List, List]]:
        """
        Get the upper bound to a global phase agnostic l2-norm error for VarQTE simulation
        Args:
            gradient_errors: Error of the state propagation gradient for each t in times
            time: List of all points in time considered throughout the simulation
            args: Optional parameters for the error bound

        Returns:
            List of the error upper bound for all times

        Raises: NotImplementedError

        """
        raise NotImplementedError

    def error_bound(self,
                    data_dir: str,
                    imag_reverse_bound: bool = False) -> Union[List, Tuple[List, List]]:
        """
        Evaluate an upper bound to the error of the VarQTE simulation
        Args:
            data_dir: Directory where the snapshots were stored
            imag_reverse_bound: Compute the additional reverse bound (ignored if notVarQITE)
            trunc_bound: Compute truncated bound - Only useful for ground state evaluation
                        (ignored if not VarQITE)
            H: Hamiltonian used for the reverse bound (ignored if not VarQITE)

        Returns: Error bounds for all accessed time points in the evolution

        """
        # Read data
        with open(os.path.join(data_dir, 'varqte_output.csv'), mode='r') as csv_file:
            fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                          'error_bound_grad',
                          'error_grad', 'resid', 'fidelity', 'true_error',
                          'phase_agnostic_true_error',
                          'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                          'trained_energy', 'energy_error', 'h_norm', 'h_squared', 'h_trip',
                          'variance', 'dtdt_trained', 're_im_grad']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            first = True
            error_bound = []
            error_bound_grad = []
            grad_errors = []
            energies = []
            time = []
            # time_steps = []
            h_squareds = []
            h_trips = []
            stddevs = []

            for line in reader:
                if first:
                    first = False
                    continue
                t_line = float(line['t'])
                if t_line in time:
                    continue
                time.append(t_line)
                error_bound.append(float(line['error_bound']))
                error_bound_grad.append(float(line['error_bound_grad']))
                grad_errors.append(float(line['error_grad']))
                energies.append(float(line['trained_energy']))
                h_squareds.append(float(line['h_squared']))
                try:
                    h_trips.append(float(line['h_trip']))
                except Exception:
                    pass
                stddevs.append(np.sqrt(float(line['variance'])))

        if not np.iscomplex(self._operator.coeff):
            direct_error_bounds = self._get_error_bound(grad_errors, time, stddevs,
                                                             h_squareds, h_trips, energies,
                                                             trapezoidal=False)
            print('direct errors', direct_error_bounds)
            np.save(os.path.join(data_dir, 'direct_error_bounds.npy'), direct_error_bounds)

            if imag_reverse_bound:
                imag_reverse_bound = self._get_reverse_error_bound(time, error_bound_grad, stddevs)
                return error_bound, imag_reverse_bound

        return error_bound

    def _init_grad_objects(self):
        """
        Initialize the gradient objects needed to perform VarQTE

        """

        self._state = self._operator[-1]
        if self._backend is not None:
            self._init_state = \
                self._state_circ_sampler.convert(self._state,
                                                 params=dict(zip(self._parameters,
                                                                 self._init_parameter_values)))
        else:
            self._init_state = self._state.assign_parameters(dict(zip(self._parameters,
                                                                      self._init_parameter_values)))
        self._init_state = self._init_state.eval().primitive.data
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_matrix = self._h.to_matrix(massive=True)
        self._h_norm = np.linalg.norm(self._h_matrix, np.infty)
        h_squared = self._h ** 2
        self._h_squared = ComposedOp([~StateFn(h_squared.reduce()), self._state])
        self._h_squared = PauliExpectation().convert(self._h_squared)
        h_trip = self._h ** 3
        self._h_trip = ComposedOp([~StateFn(h_trip.reduce()), self._state])
        self._h_trip = PauliExpectation().convert(self._h_trip)

        if not self._faster:
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

            self._nat_grad = PauliExpectation().convert(self._nat_grad)

        self._grad = Gradient(self._grad_method).convert(self._operator, self._parameters)
        # self._grad = PauliExpectation().convert(self._grad)

        self._metric = QFI(self._qfi_method).convert(self._operator.oplist[-1], self._parameters)
        # self._metric = PauliExpectation().convert(self._metric)

    def _init_ode_solver(self,
                         t: float,
                         init_params: Union[List, np.ndarray]):
        """
        Initialize ODE Solver
        Args:
            t: Evolution time
            init_params: Set of initial parameters for time 0
        """
        def error_based_ode_fun(time, params):
            param_dict = dict(zip(self._parameters, params))
            nat_grad_result, grad_res, metric_res = self._solve_sle(param_dict)

            def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
                """
                Search for the dω/dt which minimizes ||e_t||^2

                Args:
                    dt_param_values: values for dω/dt
                Returns:
                    ||e_t||^2 for given for dω/dt

                """
                et_squared = self._error_t(params, dt_param_values, grad_res,
                                           metric_res)
                # print('grad error', et_squared)
                return et_squared

            # def jac_argmin_fun(dt_param_values: Union[List, np.ndarray]
            #                    ) -> Union[List, np.ndarray]:
            #     """
            #     Get tge gradient of ||e_t||^2 w.r.t. dω/dt for given values
            #
            #     Args:
            #         dt_param_values: values for dω/dt
            #     Returns:
            #         Gradient of ||e_t||^2 w.r.t. dω/dt
            #
            #     """
            #     dw_et_squared = self._grad_error_t(dt_param_values, grad_res,
            #                                        metric_res)
            #     return dw_et_squared

            # return nat_grad_result
            # Use the natural gradient result as initial point for least squares solver
            # print('initial natural gradient result', nat_grad_result)
            argmin = minimize(fun=argmin_fun, x0=nat_grad_result, method='COBYLA', tol=1e-6)
            # argmin = sp.optimize.least_squares(fun=argmin_fun, x0=nat_grad_result, ftol=1e-6)

            print('final dt_omega', np.real(argmin.x))
            # self._et = argmin_fun(argmin.x)
            return argmin.x, grad_res, metric_res

        # Use either the argument that minimizes the gradient error or the result from McLachlan's
        # variational principle to run the ODE solver
        def ode_fun(t: float, x: Iterable) -> Iterable:
            params = x[:-1]
            error = max(x[-1], 0)
            error = min(error, np.sqrt(2))
            print('previous error', error)
            param_dict = dict(zip(self._parameters, params))
            if self._error_based_ode:
                dt_params, grad_res, metric_res = error_based_ode_fun(t, params)
            else:
                dt_params, grad_res, metric_res = self._solve_sle(param_dict)
            print('Gradient ', grad_res)
            print('Gradient norm', np.linalg.norm(grad_res))

            # Get the residual for McLachlan's Variational Principle
            # self._storage_params_tbd = (t, params, et, resid, f, true_error, true_energy,
            #                             trained_energy, h_squared, dtdt_state, reimgrad)
            if np.iscomplex(self._operator.coeff):
                # VarQRTE
                resid = np.linalg.norm(np.matmul(metric_res, dt_params) - grad_res * 0.5)
                et, h_squared, dtdt_state, reimgrad = self._error_t(params, dt_params,
                                                                    grad_res,
                                                                    metric_res)
                h_trip = None
            else:
                # VarQITE
                resid = np.linalg.norm(np.matmul(metric_res, dt_params) + grad_res * 0.5)
                # Get the error for the current step
                et, h_squared, dtdt_state, reimgrad, h_trip = self._error_t(params,
                                                                            dt_params,
                                                                            grad_res,
                                                                            metric_res)
            print('returned et', et)
            try:
                if et < 0:
                    if np.abs(et) > 1e-4:
                        raise Warning('Non-neglectible negative et observed')
                    else:
                        et = 0
                else:
                    et = np.sqrt(np.real(et))
            except Exception:
                et = 1000
            print('after try except', et)
            f, true_error, phase_agnostic_true_error, true_energy, trained_energy = \
                self._distance_energy(t, param_dict)

            # TODO stack dt params with the gradient for the error update
            if np.iscomplex(self._operator.coeff):
                # VarQRTE
                error_bound_grad = et
            else:
                # VarQITE
                error_store = max(error, 0)
                error_store = min(error_store, np.sqrt(2))
                error_bound_grad = self._get_error_grad(delta_t=1e-4, eps_t=error_store,
                                                        grad_err=et,
                                                        energy=trained_energy,
                                                        h_squared=h_squared,
                                                        h_trip=h_trip,
                                                        stddev=np.sqrt(h_squared -
                                                                       trained_energy**2),
                                                        store=self._store_now)

                print('returned grad', error_bound_grad)

            if (self._snapshot_dir is not None) and (self._store_now):
                self._store_params(t, params, error, error_bound_grad, et,
                                   resid, f, true_error, phase_agnostic_true_error, None,
                                   None, true_energy,
                                   trained_energy, h_squared, h_trip, dtdt_state,
                                   reimgrad)

            return np.append(dt_params, error_bound_grad)
        # if self._ode_solver == RK45:
        #     self._ode_solver = self._ode_solver(ode_fun, t_bound=t, t0=0, y0=init_params,
        #                                         atol=1e-6, max_step=0.01)

        if issubclass(self._ode_solver, OdeSolver):
            self._ode_solver=self._ode_solver(ode_fun, t_bound=t, t0=0, y0=init_params,
                                              atol=1e-6)

        elif self._ode_solver == ForwardEuler:
            self._ode_solver = self._ode_solver(ode_fun, t_bound=t, t0=0,
                                                y0=init_params,
                                                num_t_steps=self._num_time_steps)
        else:
            raise TypeError('Please define a valid ODESolver')
        return

    def _run_ode_solver(self, t: float,
                        init_params: Union[List, np.ndarray]):
        """
        Find numerical solution with ODE Solver
        Args:
            t: Evolution time
            init_params: Set of initial parameters for time 0
        """
        self._init_ode_solver(t, np.append(init_params, 0))
        if isinstance(self._ode_solver, OdeSolver) or isinstance(self._ode_solver, ForwardEuler):
            self._store_now = True
            _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
            while self._ode_solver.t < t:
                self._store_now = False
                self._ode_solver.step()
                if self._snapshot_dir is not None and self._ode_solver.t <= t:
                    self._store_now = True
                    _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
                print('ode time', self._ode_solver.t)
                param_values = self._ode_solver.y[:-1]
                print('ode parameters', self._ode_solver.y[:-1])
                print('ode error', self._ode_solver.y[-1])
                print('ode step size', self._ode_solver.step_size)
                if self._ode_solver.status == 'finished':
                    break
                elif self._ode_solver.status == 'failed':
                    raise Warning('ODESolver failed')

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
                      error_bound_grad: Optional[float] = None,
                      error_grad: Optional[float] = None,
                      resid: Optional[float] = None,
                      fidelity_to_target: Optional[float] = None,
                      true_error: Optional[float] = None,
                      phase_agnostic_true_error: Optional[float] = None,
                      true_to_euler_error: Optional[float] = None,
                      trained_to_euler_error: Optional[float] = None,
                      target_energy: Optional[float] = None,
                      trained_energy: Optional[float] = None,
                      h_squared: Optional[float] = None,
                      h_trip: Optional[float] = None,
                      dtdt_trained: Optional[float] = None,
                      re_im_grad: Optional[float] = None):
        """
        Store values for time t
        Args:
            t: current point in time evolution
            error_bound: ||\epsilon_t|| >= |||target> - |trained>||_2
                         (using exact |trained_energy - target_energy|)
            error_bound_grad: error_bound = integral(error_bound_grad)
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
            h_trip: <trained|H^3|trained>
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
                          'error_bound_grad',
                          'error_grad', 'resid', 'fidelity', 'true_error',
                          'phase_agnostic_true_error', 'true_to_euler_error',
                          'trained_to_euler_error', 'target_energy', 'trained_energy',
                          'energy_error', 'h_norm', 'h_squared', 'h_trip', 'variance',
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
                             'error_bound_grad': error_bound_grad,
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
                             'h_norm': self._h_norm,
                             'h_squared': h_squared,
                             'h_trip': h_trip,
                             'variance': variance,
                             'dtdt_trained': dtdt_trained,
                             're_im_grad': re_im_grad
                             })

    def _solve_sle(self,
                   param_dict: Dict
                   ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle

        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the Ansatz

        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
        Fubini-Study Metric

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

        if any(np.abs(np.imag(grad_res_item)) > 1e-3 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-3 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        print('Metric', metric_res)
        print('grad', grad_res)
        metric_res = np.real(metric_res)
        grad_res = np.real(grad_res)

        # Check if numerical instabilities lead to a metric which is not positive semidefinite
        while True:
            w, v = np.linalg.eigh(metric_res)

            if not all(ew >= -1e-2 for ew in w):
                raise Warning('The underlying metric has ein Eigenvalue < ', -1e-2,
                              '. Please use a regularized least-square solver for this problem.')
            if not all(ew >= 0 for ew in w):
                # If not all eigenvalues are non-negative, set them to a small positive
                # value
                w = [max(1e-10, ew) for ew in w]
                # Recompose the adapted eigenvalues with the eigenvectors to get a new metric
                metric_res = np.real(v @ np.diag(w) @ np.linalg.inv(v))
            else:
                # If all eigenvalues are non-negative use the metric
                break

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

        if any(np.abs(np.imag(nat_grad_item)) > 1e-8 for nat_grad_item in nat_grad_result):
            raise Warning('The imaginary part of the gradient are non-negligible.')

        print('nat grad result', nat_grad_result)

        return np.real(nat_grad_result), grad_res, metric_res

    def _bures_distance(self,
                        state1: Union[List, np.ndarray],
                        state2: Union[List, np.ndarray]) -> float:
        """
        Find the Bures metric between two normalized pure states
        Args:
            state1: Target state
            state2: Trained state with potential phase mismatch

        Returns:
            global phase agnostic l2 norm value

        """
        def bures_dist(phi):
            return np.linalg.norm(np.subtract(state1, np.exp(1j * phi) * state2), ord=2)
        bures_distance = minimize(fun=bures_dist, x0=np.array([0]), method='COBYLA', tol=1e-6)
        return bures_distance.fun

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
        phase_agnostic_act_err = self._bures_distance(target_state, trained_state)
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(self._h_matrix, target_state))
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(self._h_matrix, trained_state))
        # print('Fidelity', f)
        # print('True error', act_err)
        # print('Global phase agnostic actual error', phase_agnostic_act_err)
        # print('actual energy', act_en)
        # print('trained_en', trained_en)
        return f, act_err, phase_agnostic_act_err, np.real(act_en), np.real(trained_en)

    @staticmethod
    def plot_results(data_directories: List[str],
                     error_bound_directories: List[str],
                     reverse_bound_directories: Optional[List[str]] = None):
        """
        Plot results

        Args:
            data_directories: Directories snapshots
            error_bound_directories: Directories to error bound file
            reverse_bound_directories: Directories to reverse error bound file (only to be used
            with VarQITE)

        """
        if len(data_directories) != len(error_bound_directories):
            raise Warning('Please provide data and error bound directories of the same length '
                          'corresponding to the same data.')

        for j, data_dir in enumerate(data_directories):
            with open(os.path.join(data_dir, 'varqte_output.csv'), mode='r') as csv_file:
                fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                              'error_bound_grad',
                              'error_grad', 'resid', 'fidelity', 'true_error',
                              'phase_agnostic_true_error',
                              'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                              'trained_energy', 'energy_error', 'h_norm', 'h_squared', 'h_trip',
                              'variance', 'dtdt_trained', 're_im_grad']
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)
                first = True
                grad_errors = []
                true_error = []
                phase_agnostic_true_error = []
                time = []
                fid = []
                true_energy = []
                trained_energy = []
                stddevs = []
                variance = []

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
                    variance.append(float(line['variance']))

                zipped_lists = zip(time, grad_errors, true_error, fid, true_energy, trained_energy)

                sorted_pairs = sorted(zipped_lists)

                zipped_sorted = zip(*sorted_pairs)

                time, grad_errors, true_error, fid, true_energy, \
                trained_energy = [list(zipped_items) for zipped_items in zipped_sorted]

            error_bounds = np.load(error_bound_directories[j])
            print('error bounds', error_bounds)
            # energy_error_bounds = []
            fidelity_bounds = []
            for s, er_b in enumerate(error_bounds):
                # energy_error_bounds.append(trained_energy[s]/2.*er_b**2 + stddevs[s]*er_b)
                fidelity_bounds.append((1-er_b**2/2)**2)

            # plt.rcParams.update({'font.size': 24})

            plt.figure(0, figsize=(4, 3))
            # plt.title('Error Bound')
            plt.scatter(time, phase_agnostic_true_error, color='indigo', marker='o', s=12,
                         alpha=0.5)
            plt.scatter(time, error_bounds, color='royalblue', marker='o', s=12,
                        alpha=0.5)
            plt.plot(time, phase_agnostic_true_error, color='indigo',
                     label=r'$B(|\psi^{\omega}_t\rangle\langle\psi^{\omega}_t|, '
                           r'|\psi^*_t\rangle\langle\psi^*_t|)$',
                     alpha=0.5, linewidth=2)
            plt.plot(time, error_bounds, color='royalblue', label=r'$\epsilon_t$', alpha=0.5,
                     linewidth=2)

            # plt.scatter(time, true_error, color='purple', marker='o', s=40, label='actual error')
            plt.legend(loc='best')
            plt.xlabel('time')
            plt.ylabel('error', rotation=90)
            # plt.ylim(bottom=-0.01)
            # plt.ylim((-0.01, 0.25))
            plt.autoscale(enable=True)
            plt.ylim((-0.001, 0.11))
            # plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            plt.yticks(np.linspace(0, 0.1, 6))
            # plt.ylim((-0.0001, 1.425))
            # plt.yticks(np.linspace(0, 1.4, 15))
            # plt.autoscale(enable=True)
            # plt.autoscale(enable=True)
                          # plt.xticks(range(counter-1))
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, 'error_bound_actual.pdf'))
            plt.close()



            plt.figure(3, figsize=(4, 3))
            # plt.title('Fidelity')
            plt.scatter(time, fid, color='indigo', marker='o', s=12,
                        alpha=0.5)
            plt.scatter(time, fidelity_bounds, color='royalblue', marker='o', s=12,
                        alpha=0.5)
            plt.plot(time, fid, color='indigo', label=r'$|\langle\psi_t|\psi^*_t\rangle|^2$',
                     alpha=0.5, linewidth=2)
            plt.plot(time, fidelity_bounds, color='royalblue',
                     label=r'$(1 - \frac{\epsilon_t^2}{2})^2$', alpha=0.5, linewidth=2)
            plt.xlabel('time')
            plt.ylabel('fidelity', rotation=90)
            # plt.autoscale(enable=True)
            # plt.xticks(range(counter-1))
            # plt.ylim((0.9, 1.003))
            # plt.yticks(np.linspace(0.9, 1, 6))
            plt.ylim((0.942, 1.003))
            # plt.autoscale(enable=True)
            plt.yticks(np.linspace(0.95, 1, 6))
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, 'fidelity.pdf'))
            plt.close()

            if os.path.exists(os.path.join(data_dir, 'energy_error_bound.npy')):
                plt.figure(4, figsize=(4, 3))
                # plt.title('Energy')

                plt.scatter(time, trained_energy, color='royalblue', marker='o', s=12,
                            label=r'$E_t^{\omega}$', alpha=0.5)
                plt.scatter(time, true_energy, color='indigo', marker='o', s=12,
                            label=r'$E_t^*$', alpha=0.5)
                plt.plot(time, true_energy, color='indigo', alpha=0.5)
                plt.plot(time, trained_energy, color='royalblue', alpha=0.5)
                energy_error_bounds = np.load(os.path.join(data_dir, 'energy_error_bound.npy'))
                plt.plot(time, trained_energy + energy_error_bounds, label=r'$E_t \pm \zeta$',
                         color='turquoise', alpha=0.5, linewidth=2)
                plt.plot(time, trained_energy - energy_error_bounds,
                         color='turquoise', alpha=0.5, linewidth=2)
                # plt.errorbar(time, trained_energy, yerr=energy_error_bounds, fmt='o', ecolor='blue')
                # plt.scatter(time, energy_error_bounds, color='blue', marker='o', s=40,
                #             label='energy error bound')
                plt.ylim((-4.2, 0.3))
                plt.yticks(np.linspace(-4, 0, 5))
                # plt.autoscale(enable=True)
                # plt.yticks([-1., -0.7, -0.4, -0.1, 0.2])
                plt.xlabel('time')
                plt.ylabel('energy', rotation=90)
                plt.tight_layout()
                plt.legend(loc='best')
                # plt.autoscale(enable=True)
                plt.savefig(os.path.join(data_dir, 'energy_w_error_bounds.pdf'))
                plt.close()

            plt.figure(5, figsize=(4, 3))
            # plt.title('Energy')
            plt.scatter(time, true_energy, color='indigo', marker='o', s=12,
                        alpha=0.5)
            plt.scatter(time, trained_energy, color='royalblue', marker='o', s=12,
                        alpha=0.5)
            plt.plot(time, true_energy, color='indigo', label=r'$E_t^*$', alpha=0.5, linewidth=2)
            plt.plot(time, trained_energy, color='royalblue', label=r'$E_t^{\omega}$', alpha=0.5, linewidth=2)
            plt.xlabel('time')
            plt.ylabel('energy', rotation=90)
            plt.ylim((0.265, 0.345))
            plt.yticks(np.linspace(0.27, 0.34, 8))
            plt.tight_layout()
            plt.legend(loc='best')
            # plt.autoscale(enable=True)
            plt.savefig(os.path.join(data_dir, 'energy.pdf'))
            plt.close()

            if os.path.exists(os.path.join(data_dir, 'energy_error_bound.npy')):
                plt.figure(6, figsize=(4, 3))
                # plt.title('Energy Error Bound')
                print('time', time)
                print('energy error bounds', energy_error_bounds)
                plt.scatter(time, energy_error_bounds, color='turquoise', marker='o', s=12,
                            alpha=0.5)
                plt.plot(time, energy_error_bounds, color='turquoise', alpha=0.5, linewidth=2)
                plt.xlabel('time')
                plt.ylabel('energy error bound', rotation=90)
                plt.tight_layout()
                # plt.legend(loc='best')
                # plt.autoscale(enable=True)
                plt.savefig(os.path.join(data_dir, 'energy_error_bound.pdf'))
                plt.close()

                max_bures = np.load(os.path.join(data_dir, 'max_bures.npy'))

                plt.figure(7, figsize=(4, 3))
                plt.title('Max. Bures Metric')
                plt.scatter(time, max_bures, color='pink', marker='o', s=12,
                            alpha=0.5)
                plt.plot(time, max_bures, color='pink', alpha=0.5, linewidth=2)
                plt.xlabel('time')
                plt.ylabel('Max. Bures Metric', rotation=90)
                plt.tight_layout()
                # plt.legend(loc='best')
                # plt.autoscale(enable=True)
                plt.savefig(os.path.join(data_dir, 'max_bures.pdf'))
                plt.close()

                plt.figure(8, figsize=(4, 3))
                # plt.title('Gradient Errors')
                plt.scatter(time, grad_errors, color='purple', marker='o', s=12,
                            alpha=0.5)
                plt.plot(time, grad_errors, color='purple', alpha=0.5, linewidth=2)
                plt.xlabel('time')
                plt.ylabel('error', rotation=90)
                plt.ylim((-0.001, 0.101))
                # plt.autoscale(enable=True)
                plt.yticks(np.linspace(0, 0.1, 6))
                plt.tight_layout()
                # plt.legend(loc='best')
                # plt.autoscale(enable=True)
                plt.savefig(os.path.join(data_dir, 'grad_errors.pdf'))
                plt.close()

                plt.figure(8, figsize=(4, 3))
                # plt.title('Energy Variance')
                plt.scatter(time, variance, color='turquoise', marker='o', s=12,
                            alpha=0.5)
                plt.plot(time, variance, color='turquoise', alpha=0.5, linewidth=2)
                plt.xlabel('time')
                plt.ylabel('energy variance', rotation=90)
                # plt.ylim((-0.0001, 0.01))
                plt.autoscale(enable=True)
                plt.tight_layout()
                # plt.yticks(rotation=90)
                # plt.yticks(np.linspace(0, 0.01, 11))
                # plt.legend(loc='best')
                # plt.autoscale(enable=True)
                plt.savefig(os.path.join(data_dir, 'energy_var.pdf'))
                plt.close()

                # if os.path.exists(os.path.join(data_dir, 'direct_error_bounds.npy')):
                #     direct_error_bounds = np.load(os.path.join(data_dir, 'direct_error_bounds.npy'))
                #     plt.plot(time, direct_error_bounds, color='pink', label='direct error '
                #                                                             'bound',
                #              alpha=0.5, linewidth=2)
                #     plt.legend(loc='best')
                #     plt.savefig(os.path.join(data_dir, 'error_bound_actual_incl_direct.pdf'))
                #     plt.close()
                #
                # else:
                #     plt.close()
                #
                # if reverse_bound_directories is not None:
                #     reverse_error_bounds = np.load(reverse_bound_directories[j])
                #     reverse_fidelity_bounds = []
                #     for s, er_b in enumerate(reverse_error_bounds):
                #         reverse_fidelity_bounds.append((1 - er_b ** 2 / 2) ** 2)
                #     plt.figure(1, figsize=(4, 3))
                #     # plt.title('Reverse Error Bound ')
                #     # plt.scatter(time, true_error, color='purple', marker='o', s=40,
                #     # label='actual error')
                #     plt.scatter(time, phase_agnostic_true_error, color='indigo', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.scatter(time, error_bounds, color='royalblue', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.plot(time, phase_agnostic_true_error, label='actual error', color='indigo',
                #              linewidth=2)
                #     plt.plot(time, error_bounds, color='royalblue', label='error bound', alpha=0.5
                #              , linewidth=2)
                #     plt.scatter(time, reverse_error_bounds, color='magenta', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.plot(time, reverse_error_bounds, color='magenta',
                #              label='reverse error bound',
                #              alpha=0.5, linewidth=2)
                #
                #     plt.legend(loc='best')
                #     plt.xlabel('time')
                #     plt.ylabel('error', rotation=90)
                #     plt.ylim(bottom=-0.01)
                #     plt.autoscale(enable=True)
                #     plt.legend(loc='best')
                #     plt.autoscale(enable=True)
                #     plt.savefig(os.path.join(data_dir, 'error_bound_actual_reverse.pdf'))
                #     plt.close()
                #
                #     plt.figure(2, figsize=(4, 3))
                #     # plt.title('Fidelity')
                #     plt.scatter(time, fid, color='indigo', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.scatter(time, fidelity_bounds, color='royalblue', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.scatter(time, reverse_fidelity_bounds, color='magenta', marker='o', s=12,
                #                 alpha=0.5)
                #     plt.plot(time, fid, color='indigo', label='fidelity', alpha=0.5, linewidth=2)
                #     plt.plot(time, fidelity_bounds, color='royalblue', label='fidelity bound',
                #              alpha=0.5, linewidth=2)
                #     plt.plot(time, reverse_fidelity_bounds, color='magenta',
                #              label='reverse fidelity bound', alpha=0.5, linewidth=2)
                #     plt.xlabel('time')
                #     plt.ylabel('fidelity', rotation=90)
                #     # plt.autoscale(enable=True)
                #     # plt.xticks(range(counter-1))
                #     plt.ylim((-0.03, 1.03))
                #     # plt.autoscale(enable=True)
                #     plt.yticks(np.linspace(0., 1, 11))
                #     plt.legend(loc='best')
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(data_dir, 'reverse_fidelity.pdf'))
                #     plt.close()

        return
