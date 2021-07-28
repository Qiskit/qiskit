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
from abc import abstractmethod, ABC
from typing import List, Optional, Union, Dict, Iterable, Tuple
import os
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import OdeSolver, ode
from scipy.optimize import minimize

from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.opflow import StateFn, ListOp, CircuitSampler, ComposedOp, PauliExpectation
from qiskit.opflow.gradients import CircuitQFI, CircuitGradient, Gradient, QFI, \
    NaturalGradient
from qiskit.quantum_info import state_fidelity


class VarQte(ABC):
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
        # Initialize H Norm
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_matrix = self._h.to_matrix(massive=True)
        self._h_norm = np.linalg.norm(self._h_matrix, np.infty)

    @abstractmethod
    def _exact_state(self,
                     time: Union[float, complex]) -> Iterable:
        """
        Args:
            time: current time
        Raises: NotImplementedError
        """
        raise NotImplementedError

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
                                           metric_res)[0]
                # print('grad error', et_squared)
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
                                                                       trained_energy ** 2),
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
            self._ode_solver = self._ode_solver(ode_fun, t_bound=t, t0=0, y0=init_params,
                                                atol=1e-10)

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
                                                                    params=param_dict).eval()) * \
                         0.25
        else:
            grad_res = np.array(self._grad.assign_parameters(param_dict).eval())
            # Get the QFI/4
            metric_res = np.array(self._metric.assign_parameters(param_dict).eval()) * 0.25

        if any(np.abs(np.imag(grad_res_item)) > 1e-3 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-3 for metric_res_item in metric_res_row]
                   for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the gradient are non-negligible.')

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
