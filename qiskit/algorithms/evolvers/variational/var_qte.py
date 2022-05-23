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
from functools import partial
from typing import Optional, Union, Dict, List, Callable

import numpy as np
from scipy.integrate import RK45, OdeSolver

from qiskit import QuantumCircuit
from qiskit.algorithms import EvolutionProblem
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    StateFn,
    CircuitSampler,
    OperatorBase,
    ExpectationBase,
)
from qiskit.utils.backend_utils import is_aer_provider
from .solvers.ode.ode_function_factory import OdeFunctionFactory

from .solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)
from .variational_principles.variational_principle import (
    VariationalPrinciple,
)
from .solvers.ode.var_qte_ode_solver import (
    VarQTEOdeSolver,
)


class VarQTE(ABC):
    """Variational Quantum Time Evolution.

    Algorithms that use variational variational_principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state.

    References:

        [1] Benjamin, Simon C. et al. (2019).
        Theory of variational quantum simulation. `<https://doi.org/10.22331/q-2019-10-07-191>`_
    """

    def __init__(
        self,
        variational_principle: VariationalPrinciple,
        ode_function_factory: OdeFunctionFactory,
        ode_solver: OdeSolver = RK45,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = partial(
            np.linalg.lstsq, rcond=1e-2
        ),
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_function_factory: Factory for the ODE function.
            ode_solver: ODE solver callable that follows a SciPy ``OdeSolver`` interface.
            lse_solver: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                expectation values of ``EvolutionProblem.aux_operators``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
        """
        super().__init__()
        self.variational_principle = variational_principle
        self._quantum_instance = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance
        self.expectation = expectation
        self.ode_function_factory = ode_function_factory
        self.ode_solver = ode_solver
        self.lse_solver = lse_solver
        self.imag_part_tol = imag_part_tol
        self.num_instability_tol = num_instability_tol

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Returns quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance, Backend]) -> None:
        """Sets quantum_instance"""
        if not isinstance(quantum_instance, QuantumInstance):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance
        self._circuit_sampler = CircuitSampler(
            quantum_instance, param_qobj=is_aer_provider(quantum_instance.backend)
        )

    def _evolve(
        self,
        init_state_param_dict: Dict[Parameter, complex],
        hamiltonian: OperatorBase,
        time: float,
        t_param: Parameter,
        initial_state: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        error_calculator=None,
    ) -> OperatorBase:
        r"""
        Helper method for performing time evolution. Works both for imaginary and real case.

        Args:
            init_state_param_dict: Parameter dictionary with initial values for a given
                parametrized state/ansatz. If no initial parameter values are provided, they are
                initialized uniformly at random.
            hamiltonian:
                Operator used for Variational Quantum Imaginary Time Evolution (VarQTE).
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a
                ``ComboFn``.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            initial_state: Quantum state to be evolved.
            error_calculator: Not yet supported. Calculator of errors for error-based ODE functions.


        Returns:
            Result of the evolution which is a quantum circuit with bound parameters as an
            evolved state.
        """

        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameters_values = list(init_state_param_dict.values())

        metric_tensor = self.variational_principle.calc_metric_tensor(
            initial_state, init_state_parameters
        )
        evolution_grad = self.variational_principle.calc_evolution_grad(
            hamiltonian, initial_state, init_state_parameters
        )

        linear_solver = VarQTELinearSolver(
            metric_tensor,
            evolution_grad,
            self.lse_solver,
            self._circuit_sampler,
            self.imag_part_tol,
        )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        ode_function = self.ode_function_factory.build(
            linear_solver,
            error_calculator,
            t_param,
            init_state_param_dict,
        )

        ode_solver = VarQTEOdeSolver(init_state_parameters_values, ode_function, self.ode_solver)
        parameter_values = ode_solver.run(time)
        param_dict_from_ode = dict(zip(init_state_parameters, parameter_values))

        return initial_state.assign_parameters(param_dict_from_ode)

    def bind_parameters_to_state(
        self,
        state: Union[QuantumCircuit, StateFn],
        param_dict: Dict[Parameter, complex],
    ) -> None:
        r"""
        Bind parameters in a given quantum state to values provided. Uses a ``CircuitSampler`` if
        available.

        Args:
            state: Parametrized quantum state to be bound.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        """
        if self._circuit_sampler:
            initial_state = self._circuit_sampler.convert(state, param_dict)
        else:
            initial_state = state.assign_parameters(param_dict)
        return initial_state.eval().primitive.data

    @staticmethod
    def _create_init_state_param_dict(
        param_value_dict: Dict[Parameter, complex],
        init_state_parameters: List[Parameter],
    ) -> Dict[Parameter, complex]:
        r"""
        Looks for parameters present in an initial state (an ansatz) in a ``param_value_dict``
        provided. Based on that, it creates a new dictionary containing only parameters present
        in an initial state and their respective values. If no value for an initial state parameter
        is present, it is chosen uniformly at random.

        Args:
            param_value_dict: Dictionary which relates parameter values to the parameters.
            init_state_parameters: Parameters present in a quantum state.

        Returns:
            Dictionary that maps parameters of an initial state to some values.
        """
        if param_value_dict is None:
            init_state_parameter_values = np.random.random(len(init_state_parameters))
        else:
            init_state_parameter_values = []
            for param in init_state_parameters:
                if param in param_value_dict.keys():
                    init_state_parameter_values.append(param_value_dict[param])
                else:
                    init_state_parameter_values.append(np.random.random(len(init_state_parameters)))
        init_state_param_dict = dict(zip(init_state_parameters, init_state_parameter_values))
        return init_state_param_dict

    def _validate_aux_ops(self, evolution_problem: EvolutionProblem) -> None:
        if evolution_problem.aux_operators is not None and (
            self.quantum_instance is None or self.expectation is None
        ):
            raise ValueError(
                "aux_operators where provided for evaluations but no ``expectation`` or "
                "``quantum_instance`` was provided."
            )
