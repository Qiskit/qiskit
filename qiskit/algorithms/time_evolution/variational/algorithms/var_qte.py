# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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
from typing import Optional, Union, Dict, List, Callable

import numpy as np
from scipy.integrate import RK45, OdeSolver

from qiskit import QuantumCircuit
from qiskit.algorithms.time_evolution.evolution_base import EvolutionBase
from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
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
)


class VarQte(EvolutionBase, ABC):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191
    Algorithms that use variational variational_principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state.
    """

    def __init__(
        self,
        variational_principle: VariationalPrinciple,
        ode_function_generator: AbstractOdeFunctionGenerator,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        ode_solver_callable: OdeSolver = RK45,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        allowed_imaginary_part: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
    ):
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            backend: Backend used to evaluate the quantum circuit outputs.
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
            lse_solver_callable: Linear system of equations solver that follows a NumPy
                np.linalg.lstsq interface.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
        """
        super().__init__()
        self._variational_principle = variational_principle

        self._backend = backend
        # we define separate instances of CircuitSamplers as it caches aggressively according
        # to its documentation
        self._init_samplers()

        self._ode_function_generator = ode_function_generator
        self._ode_solver_callable = ode_solver_callable
        self._lse_solver_callable = lse_solver_callable
        self._allowed_imaginary_part = allowed_imaginary_part
        self._allowed_num_instability_error = allowed_num_instability_error

    def _evolve_helper(
        self,
        init_state_param_dict: Dict[Parameter, Union[float, complex]],
        hamiltonian: OperatorBase,
        time: float,
        t_param: Parameter,
        error_calculator=None,  # TODO will be supported in another PR
        initial_state: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        observable: Optional[OperatorBase] = None,
    ) -> OperatorBase:
        """
        Helper method for performing time evolution. Works both for imaginary and real case.

        Args:
            init_state_param_dict: Parameter dictionary with initial values for a given
                parametrized state/ansatz.
            hamiltonian:
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQte)
                The coefficient of the operator (operator.coeff) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a
                ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved. Not supported by VarQte.
            t_param: Time parameter in case of a time-dependent Hamiltonian.

        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective time evolution.

        Raises:
            TypeError: If observable is provided - not supported by this algorithm.
        """
        if observable is not None:
            raise TypeError(
                "Observable argument provided. Observable evolution not supported by VarQte."
            )
        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameters_values = list(init_state_param_dict.values())

        metric_tensor = self._variational_principle._get_metric_tensor(
            initial_state, init_state_parameters
        )
        evolution_grad = self._variational_principle._get_evolution_grad(
            hamiltonian, initial_state, init_state_parameters
        )

        linear_solver = VarQteLinearSolver(
            metric_tensor,
            evolution_grad,
            self._lse_solver_callable,
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._energy_sampler,
            self._allowed_imaginary_part,
        )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        self._ode_function_generator._lazy_init(
            linear_solver,
            error_calculator,
            t_param,
            init_state_param_dict,
        )

        ode_solver = VarQteOdeSolver(
            init_state_parameters_values, self._ode_function_generator, self._ode_solver_callable
        )
        parameter_values = ode_solver._run(time)
        param_dict_from_ode = dict(zip(init_state_parameters, parameter_values))

        # if self._state_circ_sampler:
        #     return self._state_circ_sampler.convert(initial_state, param_dict_from_ode)
        return initial_state.assign_parameters(param_dict_from_ode)

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
            initial_state = self._state_circ_sampler.convert(state, param_dict)
        else:
            initial_state = state.assign_parameters(param_dict)
        return initial_state.eval().primitive.data

    def _init_samplers(self) -> None:
        """Creates all possible samplers if a backend is present."""
        self._operator_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._state_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._h_squared_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._grad_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._metric_circ_sampler = CircuitSampler(self._backend) if self._backend else None
        self._energy_sampler = CircuitSampler(self._backend) if self._backend else None

    def _hamiltonian_power(
        self, hamiltonian: OperatorBase, initial_state, power: int
    ) -> OperatorBase:
        """
        Calculates a Hamiltonian raised to a given power.

        Args:
            power: Power to which a Hamiltonian operator should be raised.

        Returns:
            Hamiltonian raised to a given power.
        """
        h_power = hamiltonian**power
        h_power = ComposedOp([~StateFn(h_power.reduce()), StateFn(initial_state)])
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
