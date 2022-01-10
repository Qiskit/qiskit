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

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Callable, List

import numpy as np
from scipy.integrate import RK45, OdeSolver

from qiskit import QuantumCircuit

from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.error_based_ode_function_generator import (
    ErrorBasedOdeFunctionGenerator,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    StateFn,
    CircuitSampler,
    ComposedOp,
    PauliExpectation,
    OperatorBase,
    CircuitStateFn,
    Gradient, EvolutionBase,
)


class VarQte(ABC, EvolutionBase):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191
    Algorithms that use variational variational_principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state.
    """

    def __init__(
        self,
        variational_principle: VariationalPrinciple,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        error_based_ode: Optional[bool] = False,
        ode_solver_callable: OdeSolver = RK45,
        optimizer: str = "COBYLA",
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
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
            optimizer: Optimizer used in case error_based_ode is true.
        """
        super().__init__()
        self._variational_principle = variational_principle
        self._regularization = regularization

        self._backend = backend
        # we define separate instances of CircuitSamplers as it caches aggressively according
        # to its documentation
        self._init_samplers()

        self._error_based_ode = error_based_ode
        self._ode_solver_callable = ode_solver_callable
        self._optimizer = optimizer

        self._operator = None
        self._initial_state = None

    @abstractmethod
    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: OperatorBase = None,
        observable: OperatorBase = None,
        t_param: Parameter = None,
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]] = None,
    ) -> StateFn:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given
        operator
        Args:
            hamiltonian:
                ⟨ψ(ω)|H(t, theta)|ψ(ω)〉
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQITE)
                The coefficient of the operator (operator.coeff) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a
                ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved. Not supported by VarQite.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                                    certain values, including the t_param.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective
            time evolution.
        """
        pass

    def _evolve_helper(
        self,
        ode_function_generator_callable: Callable,
        init_state_param_dict: Dict[Parameter, Union[float, complex]],
        hamiltonian: OperatorBase,
        time: float,
        initial_state: OperatorBase = None,
        observable: OperatorBase = None,
        t_param: Parameter = None,
    ) -> OperatorBase:
        """
        Helper method for performing time evolution. Works both for imaginary and real case.
        Args:
            ode_function_generator_callable: Callable of a function that will be used by an ODE
                                            solver.
            init_state_param_dict: Parameter dictionary with initial values for a given
                                parametrized state/ansatz.
            hamiltonian:
                ⟨ψ(ω)|H|ψ(ω)〉
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQITE)
                The coefficient of the operator (operator.coeff) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a
                ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved. Not supported by VarQite.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective
            time evolution.
        Raises:
            TypeError: If observable is provided - not supported by this algorithm.
        """
        if observable is not None:
            raise TypeError(
                "Observable argument provided. Observable evolution not supported by VarQte."
            )
        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameters_values = list(init_state_param_dict.values())

        self._variational_principle._lazy_init(hamiltonian, initial_state, init_state_parameters)
        self.bind_initial_state(StateFn(initial_state), init_state_param_dict)
        self._operator = self._variational_principle._operator
        self._validate_operator(self._operator)

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator

        self._init_ham_objects()

        ode_function_generator = ode_function_generator_callable(init_state_param_dict, t_param)

        ode_solver = VarQteOdeSolver(
            init_state_parameters_values, ode_function_generator, self._ode_solver_callable
        )
        parameter_values = ode_solver._run(time)
        # return evolved
        # initial state here is not with self because we need a parametrized state (input to this
        # method)
        param_dict_from_ode = dict(zip(init_state_parameters, parameter_values))

        # if self._state_circ_sampler:
        #     return self._state_circ_sampler.convert(initial_state, param_dict_from_ode)
        return initial_state.assign_parameters(param_dict_from_ode)

    @abstractmethod
    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: OperatorBase = None,
        t_param: Parameter = None,
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]] = None,
        gradient_params=None,
    ):
        """Performs Variational Quantum Time Evolution of gradient expressions."""
        pass

    def bind_initial_state(
        self,
        state: Union[QuantumCircuit, StateFn],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> None:
        """
        Bind parameters in a given quantum state to values provided. Uses a CircuitSampler if
        available.
        Args:
            state: Parametrized quantum state to be bound.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        """
        if self._state_circ_sampler:
            self._initial_state = self._state_circ_sampler.convert(state, param_dict)
        else:
            self._initial_state = state.assign_parameters(param_dict)
        self._initial_state = self._initial_state.eval().primitive.data

    def _init_samplers(self) -> None:
        """Creates all possible samplers if a backend is present."""
        self._operator_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._state_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._h_squared_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._grad_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._metric_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._energy_sampler = CircuitSampler(self._backend) if self._backend else None

    def _init_ham_objects(self) -> None:
        """Initialize the gradient objects needed to perform VarQTE."""
        self._hamiltonian = self._operator.oplist[0].primitive * self._operator.oplist[0].coeff
        self._hamiltonian_squared = self._hamiltonian_power(2)

    def _hamiltonian_power(self, power: int) -> OperatorBase:
        """
        Calculates a Hamiltonian raised to a given power.
        Args:
            power: Power to which a Hamiltonian operator should be raised.
        Returns:
            Hamiltonian raised to a given power.
        """
        h_power = self._hamiltonian ** power
        h_power = ComposedOp([~StateFn(h_power.reduce()), StateFn(self._initial_state)])
        h_power = PauliExpectation().convert(h_power)
        # TODO Include Sampler here if backend is given
        return h_power

    # TODO handle the case where quantum state params are not present in a dictionary; possibly
    #  rename the dictionary because it not only relates to a Hamiltonian but also to a state
    def _create_init_state_param_dict(
        self,
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]],
        init_state_parameters: List[Parameter],
    ) -> Dict[Parameter, Union[float, complex]]:
        """
        Looks for parameters present in an initial state (an ansatz) in a hamiltonian_value_dict
        provided. Based on that, it creates a new dictionary containing only parameters present
        in an initial state and their respective values. If not hamiltonian_value_dict is
        present, values are chosen uniformly at random.
        Args:
            hamiltonian_value_dict: Dictionary which relates parameter values to the parameters.
            init_state_parameters: Parameters present in a quantum state.
        Returns: Dictionary that maps parameters of an initial state to some values.
        """
        if hamiltonian_value_dict is None:
            init_state_parameter_values = np.random.random(len(init_state_parameters))
        else:
            init_state_parameter_values = []
            for param in init_state_parameters:
                if param in hamiltonian_value_dict.keys():
                    init_state_parameter_values.append(hamiltonian_value_dict[param])
        init_state_param_dict = dict(zip(init_state_parameters, init_state_parameter_values))
        return init_state_param_dict

    def _create_ode_function_generator(
        self,
        error_calculator: ErrorCalculator,
        init_state_param_dict: Dict[Parameter, Union[float, complex]],
        t_param: Optional[Parameter] = None,
    ) -> AbstractOdeFunctionGenerator:
        """
        Creates an ODE function generator for variational time evolution, with an
        ErrorCalculator in case of an error-based evolution.
        Args:
            error_calculator: ErrorCalculator object to calculate gradient errors in case of
                                error-based evolution.
            init_state_param_dict: Dictionary mapping parameters to their initial values for a
                                quantum state.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
        Returns:
            Instantiated ODE function generator.
        """
        # TODO potentially introduce a factory
        if self._error_based_ode:
            ode_function_generator = ErrorBasedOdeFunctionGenerator(
                error_calculator,
                init_state_param_dict,
                self._variational_principle,
                self._grad_circ_sampler,
                self._metric_circ_sampler,
                self._energy_sampler,
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
                self._energy_sampler,
                self._regularization,
                self._backend,
                t_param,
            )

        return ode_function_generator

    def _validate_operator(self, operator) -> None:
        """
        Validates a constructed operator to make sure that it includes an ansatz and that it is
        an ComposedOp or a ListOp.
        Args:
            operator: Operator to be validated.
        Raises:
            TypeError: In case an operator provided is not valid.
        """
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError("Please provide the respective ansatz as a CircuitStateFn.")
        if not isinstance(operator, ComposedOp) and not all(
            isinstance(op, CircuitStateFn) for op in operator.oplist
        ):
            raise TypeError(
                "Please provide the operator either as ComposedOp or as ListOp of a "
                "CircuitStateFn potentially with a combo function."
            )
