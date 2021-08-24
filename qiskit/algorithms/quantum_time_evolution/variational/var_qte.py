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
from typing import List, Optional, Union, Iterable

import numpy as np
from scipy.integrate import OdeSolver, ode

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, CircuitSampler, ComposedOp, PauliExpectation
from qiskit.opflow.gradients import Gradient, QFI, NaturalGradient


class VarQte(ABC):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191
    Algorithms that use McLachlans variational principle to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and quantum state.
    """

    def __init__(
        self,
        variational_principle: VariationalPrinciple,
        regularization: Optional[str] = None,
        init_parameter_values: Optional[Union[List, np.ndarray]] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        snapshot_dir: Optional[str] = None,
        faster: bool = True,
        error_based_ode: bool = False,
        **kwargs,
    ):
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
                             If True use the argument that minimizes the error error_bounds
            kwargs (dict): Optional parameters for a CircuitGradient
        """
        super().__init__()
        self._variational_principle = variational_principle
        self._grad_method = variational_principle._grad_method
        self._qfi_method = variational_principle._qfi_method
        self._regularization = regularization
        self._epsilon = kwargs.get("epsilon", 1e-6)
        self._parameters = variational_principle._parameters
        if init_parameter_values is not None:
            self._init_parameter_values = init_parameter_values
        else:
            self._init_parameter_values = np.random.random(len(self._parameters))
        self._backend = backend
        if self._backend is not None:
            # we define separate instances of CircuitSamplers as it caches aggresively according
            # to it documentation
            self._operator_circ_sampler = CircuitSampler(self._backend)
            self._state_circ_sampler = CircuitSampler(self._backend)
            self._h_squared_circ_sampler = CircuitSampler(self._backend)
            self._h_trip_circ_sampler = CircuitSampler(self._backend)
            self._grad_circ_sampler = CircuitSampler(self._backend)
            self._metric_circ_sampler = CircuitSampler(self._backend)
            if not faster:
                self._nat_grad_circ_sampler = CircuitSampler(self._backend, caching="all")
        self._faster = faster
        self._ode_function_generator = OdeFunctionGenerator(
            self._dt_params,
            self._nat_grad_res,
            self._grad_res,
            self._metric_res,
            self._param_dict,
            self._variational_principle,
            self._state,
            self._exact_state,
            self._h_matrix,
            self._state_circ_sampler,
        )
        self._ode_solver = VarQteOdeSolver(t, init_parameter_values, self._ode_function_generator)
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
    def _exact_state(self, time: Union[float, complex]) -> Iterable:
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
            self._init_state = self._state_circ_sampler.convert(
                self._state, params=dict(zip(self._parameters, self._init_parameter_values))
            )
        else:
            self._init_state = self._state.assign_parameters(
                dict(zip(self._parameters, self._init_parameter_values))
            )
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
                self._nat_grad = NaturalGradient(
                    grad_method=self._grad_method,
                    qfi_method=self._qfi_method,
                    regularization=self._regularization,
                ).convert(self._operator * 0.5, self._parameters)
            # VarQITE
            else:
                self._nat_grad = NaturalGradient(
                    grad_method=self._grad_method,
                    qfi_method=self._qfi_method,
                    regularization=self._regularization,
                ).convert(self._operator * -0.5, self._parameters)

            self._nat_grad = PauliExpectation().convert(self._nat_grad)

        self._grad = Gradient(self._grad_method).convert(self._operator, self._parameters)
        # self._grad = PauliExpectation().convert(self._grad)

        self._metric = QFI(self._qfi_method).convert(self._operator.oplist[-1], self._parameters)
        # self._metric = PauliExpectation().convert(self._metric)
