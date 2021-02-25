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


from scipy.integrate import OdeSolver, ode, solve_ivp
from scipy.optimize import least_squares

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
        self._faster = faster
        self._ode_solver = ode_solver
        self._snapshot_dir = snapshot_dir
        if snapshot_dir:
            if not os.path.exists(snapshot_dir):
                os.mkdir(snapshot_dir)
            if not os.path.exists(os.path.join(self._snapshot_dir, 'varqte_output.csv')):
                with open(os.path.join(self._snapshot_dir, 'varqte_output.csv'), mode='w') as \
                        csv_file:
                    fieldnames = ['t', 'params', 'num_params', 'num_time_steps', 'error_bound',
                                  'error_grad', 'resid', 'fidelity', 'true_error',
                                  'true_to_euler_error', 'trained_to_euler_error', 'target_energy',
                                  'trained_energy', 'energy_error', 'h_norm', 'h_squared',
                                  'variance', 'dtdt_trained', 're_im_grad']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        self._operator = None
        self._nat_grad = None
        self._metric = None
        self._grad = None

        self._exact_euler_state = None


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

    def _init_grad_objects(self):
        # Adapt coefficients for the real part for McLachlan with 0.5
        # True and needed!! Double checked

        self._state = self._operator[-1]
        # Convert the operator with the CircuitSampler
        if self._backend is not None:
            self._state = CircuitSampler(self._backend).convert(self._state)
        self._init_state = self._state.assign_parameters(dict(zip(self._parameters,
                                                                  self._parameter_values)))
        self._init_state = self._init_state.eval().primitive.data
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_matrix = self._h.to_matrix(massive=True)

        self._h_squared = ComposedOp([~StateFn(self._h ** 2), self._state])
        if self._backend is not None:
            self._h_squared = CircuitSampler(self._backend).convert(self._h_squared)

        self._exact_euler_state = self._exact_state(0)

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

        self._grad = Gradient(self._grad_method).convert(self._operator, self._parameters)

        self._metric = QFI(self._qfi_method).convert(self._operator.oplist[-1], self._parameters)

        if self._backend is not None:
            if not self._faster:
                self._nat_grad = CircuitSampler(self._backend).convert(self._nat_grad)
            self._metric = CircuitSampler(self._backend).convert(self._metric)
            self._grad = CircuitSampler(self._backend).convert(self._grad)

    def _init_ode_solver(self, t: float):
        """
        Initialize ODE Solver
        Args:
            t: time
        """
        def ode_fun(time, params):
            param_dict = dict(zip(self._parameters, params))
            nat_grad_result, grad_res, metric_res = self._propagate(param_dict)

            def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
                """
                Search for the dω/dt which minimizes ||e_t||^2

                Args:
                    dt_param_values: values for dω/dt
                Returns:
                    ||e_t||^2 for given for dω/dt

                """
                et_squared = self._error_t(self._operator, dt_param_values, grad_res,
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
                dw_et_squared = self._grad_error_t(self._operator, dt_param_values, grad_res,
                                                   metric_res)
                return dw_et_squared

            # TODO remove
            # Used to check the intermediate fidelity
            _ = self._distance_energy(time, trained_param_dict=dict(zip(
                self._parameters, params)))
            # TODO remove alternative
            # TODO We could also easily run this with the natural gradient result. Good results.
            # return nat_grad_result

            # ls = least_squares(fun=argmin_fun, x0=nat_grad_result, jac=jac_argmin_fun,
            #                      bounds=(0, 2 * np.pi), ftol=1e-2)

            # Use the natural gradient result as initial point for least squares solver
            ls = least_squares(fun=argmin_fun, x0=nat_grad_result, jac=jac_argmin_fun,
                               ftol=1e-8)
            print('least squares solved')
            print('initial natural gradient result', nat_grad_result)
            print('final dt_omega', ls.x)
            return ls.x

        if issubclass(self._ode_solver, OdeSolver):
            self._ode_solver=self._ode_solver(fun=ode_fun, t0=0,
                                              t_bound=t, y0=self._parameter_values)
        elif self._ode_solver == ode:
            self._ode_solver = ode(ode_fun)

        elif self._ode_solver == solve_ivp:
            self._ode_solver = self._ode_solver(ode_fun, (0, t), y0=self._parameter_values,
                                                method='BDF')
        elif self._ode_solver == backward_euler_fsolve:
            self._ode_solver=self._ode_solver(ode_fun,  tspan=(0, t), y0=self._parameter_values,
                                              n=self._num_time_steps)
        else:
            raise TypeError('Please define a valid ODESolver')
        return

    def _run_ode_solver(self, t):
        self._init_ode_solver(t=t)
        if isinstance(self._ode_solver, OdeSolver):
            while self._ode_solver.t < t:
                self._ode_solver.step()
                print('ode time', self._ode_solver.t)
                param_values = self._ode_solver.y
                print('ode parameters', self._ode_solver.y)
                print('ode step size', self._ode_solver.step_size)
                if self._ode_solver.status == 'finished':
                    break
                elif self._ode_solver.status == 'failed':
                    raise Warning('ODESolver failed')
        # elif isinstance(self._ode_solver, ode):
        #     self._ode_solver.set_integrator('vode', method='bdf')
        #     self._ode_solver.set_initial_value(self._parameter_values, 0)
        #
        #     t1 = t
        #
        #     while self._ode_solver.successful() and self._ode_solver.t < t1:
        #         print('ode step', self._ode_solver.t + dt, self._ode_solver.integrate(
        #             self._ode_solver.t +
        #             dt))
        #         param_values = self._ode_solver.y
        #         print('ode time', self._ode_solver.t)
        elif isinstance(self._ode_solver, backward_euler_fsolve):
            t, param_values = self._ode_solver.run()

        else:
            raise TypeError('Please provide a scipy ODESolver or ode type object.')
        print('Parameter Values ', param_values)
        _ = self._distance_energy(t, trained_param_dict=dict(zip(self._parameters, param_values)))

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
                      error_grad: Optional[float] = None,
                      resid: Optional[float] = None,
                      fidelity_to_target: Optional[float] = None,
                      true_error: Optional[float] = None,
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
                          'error_grad', 'resid', 'fidelity', 'true_error', 'true_to_euler_error',
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
                             'error_grad': error_grad,
                             'resid': resid,
                             'fidelity': np.round(fidelity_to_target, 8),
                             'true_error': np.round(true_error, 8),
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

        grad_res = self._grad.assign_parameters(param_dict).eval()
        # Get the QFI/4
        metric_res = self._metric.assign_parameters(param_dict).eval() * 0.25

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
            print('nat grad result', nat_grad_result)
        else:
            nat_grad_result = self._nat_grad.assign_parameters(param_dict).eval()
            print('nat grad result', nat_grad_result)

        w, v = np.linalg.eig(metric_res)

        if not all(ew >= -1e-8 for ew in w):
            raise Warning('The underlying metric has ein Eigenvalue < ', -1e-8,
                          '. Please use a regularized least-square solver for this problem.')
        if not all(ew >= 0 for ew in w):
            w = [max(0, ew) for ew in w]
            metric_res = v @ np.diag(w) @ np.linalg.inv(v)

        if any(np.abs(np.imag(grad_res_item)) > 1e-8 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-8 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the gradient are non-negligible.')

        return np.real(nat_grad_result), np.real(grad_res), np.real(metric_res)

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

        # |state_t>
        trained_state = self._state.assign_parameters(trained_param_dict)
        target_state = self._exact_state(time)
        trained_state = trained_state.eval().primitive.data

        # Fidelity
        f = state_fidelity(target_state, trained_state)
        # Actual error
        act_err = np.linalg.norm(target_state - trained_state, ord=2)
        # Target Energy
        act_en = self._inner_prod(target_state, np.dot(self._h_matrix, target_state))
        # Trained Energy
        trained_en = self._inner_prod(trained_state, np.dot(self._h_matrix, trained_state))
        print('Fidelity', f)
        print('True error', act_err)
        print('actual energy', act_en)
        print('trained_en', trained_en)
        # Error between exact evolution and Euler evolution using exact gradients
        exact_exact_euler_err = np.linalg.norm(target_state - self._exact_euler_state, ord=2)
        # Error between Euler evolution using exact gradients and the trained state
        exact_euler_trained_err = np.linalg.norm(trained_state - self._exact_euler_state, ord=2)
        return f, act_err, act_en, trained_en, exact_exact_euler_err, exact_euler_trained_err

