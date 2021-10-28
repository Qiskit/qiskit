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
from abc import ABC
from typing import Optional, Union, Dict

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode \
    .error_based_ode_function_generator import \
    ErrorBasedOdeFunctionGenerator
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.ode_function_generator \
    import \
    OdeFunctionGenerator
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, CircuitSampler, ComposedOp, PauliExpectation, OperatorBase


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
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
            error_based_ode: Optional[bool] = False,
            optimizer: str = "COBYLA",
            epsilon: Optional[float] = 10e-6,
    ):
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                If regularization is None but the metric is ill-conditioned or singular then
                a least square solver is used without regularization
            backend: Backend used to evaluate the quantum circuit outputs
            error_based_ode: If False use the provided variational principle to get the parameter
                                updates.
                             If True use the argument that minimizes the error error_bounds.
            optimizer: Optimizer used in case error_based_ode is true.
            epsilon: # TODO, not sure where this will be used.
        """
        super().__init__()
        self._variational_principle = variational_principle
        self._regularization = regularization
        self._epsilon = epsilon

        self._backend = backend
        # we define separate instances of CircuitSamplers as it caches aggresively according
        # to it documentation
        self._init_samplers()

        self._error_based_ode = error_based_ode
        self._optimizer = optimizer

        self._operator = None
        self._initial_state = None

    @property
    def initial_state(self):
        return self._initial_state

    def bind_initial_state(self, state, param_dict: Dict[Parameter, Union[float, complex]]):
        if self._backend is not None:
            self._initial_state = self._state_circ_sampler.convert(state, params=param_dict)
        else:
            self._initial_state = state.assign_parameters(param_dict)
        self._initial_state = self._initial_state.eval().primitive.data

    def _init_samplers(self):
        self._operator_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._state_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._h_squared_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._h_trip_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._grad_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._metric_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._nat_grad_circ_sampler = (
            CircuitSampler(self._backend, caching="all") if self._backend else None
        )

    def _init_grad_objects(self) -> None:
        """
        Initialize the gradient objects needed to perform VarQTE.
        """
        self._h = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._h_squared = self._h_pow(2)
        self._h_trip = self._h_pow(3)

    def _h_pow(self, power: int) -> OperatorBase:
        h_power = self._h ** power
        h_power = ComposedOp([~StateFn(h_power.reduce()), StateFn(self._initial_state)])
        h_power = PauliExpectation().convert(h_power)
        return h_power

    def _create_init_state_param_dict(self, hamiltonian_value_dict, init_state_parameters):
        if hamiltonian_value_dict is None:
            init_state_parameter_values = np.random.random(len(init_state_parameters))
        else:
            init_state_parameter_values = []
            for param in init_state_parameters:
                if param in hamiltonian_value_dict.keys():
                    init_state_parameter_values.append(hamiltonian_value_dict[param])
        init_state_param_dict = dict(zip(init_state_parameters, init_state_parameter_values))
        return init_state_param_dict, init_state_parameter_values

    def _create_ode_function_generator(self, error_calculator, init_state_param_dict, t_param):
        # TODO potentially introduce a factory
        if self._error_based_ode:
            ode_function_generator = ErrorBasedOdeFunctionGenerator(
                error_calculator,
                init_state_param_dict,
                self._variational_principle,
                self._grad_circ_sampler,
                self._metric_circ_sampler,
                self._nat_grad_circ_sampler,
                self._regularization,
                self._backend,
                t_param,
                self._optimizer,
            )
        else:
            ode_function_generator = OdeFunctionGenerator(
                init_state_param_dict,
                self._variational_principle,
                self._grad_circ_sampler,
                self._metric_circ_sampler,
                self._nat_grad_circ_sampler,
                self._regularization,
                self._backend,
                t_param,
            )

        return ode_function_generator
