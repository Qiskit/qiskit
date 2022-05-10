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

from qiskit.algorithms.evolvers.variational.solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)
from qiskit.algorithms.evolvers.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.evolvers.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)
from qiskit.algorithms.evolvers.variational.solvers.ode.var_qte_ode_solver import (
    VarQTEOdeSolver,
)


class VarQTE(ABC):
    """Variational Quantum Time Evolution.
       https://doi.org/10.22331/q-2019-10-07-191
    Algorithms that use variational variational_principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state.
    """

    def __init__(
        self,
        variational_principle: VariationalPrinciple,
        ode_function_generator: AbstractOdeFunctionGenerator,
        ode_solver_callable: OdeSolver = RK45,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_function_generator: Generates the ODE function.
            ode_solver_callable: ODE solver callable that follows a SciPy ``OdeSolver`` interface.
            lse_solver_callable: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                expectation values of ``EvolutionProblem.aux_operators``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
            quantum_instance: Backend used to evaluate the quantum circuit outputs.
        """
        super().__init__()
        self._variational_principle = variational_principle
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._backend = quantum_instance
        self._expectation = expectation
        self._circuit_sampler = CircuitSampler(self._backend) if self._backend else None
        self._ode_function_generator = ode_function_generator
        self._ode_solver_callable = ode_solver_callable
        self._lse_solver_callable = lse_solver_callable
        self._imag_part_tol = imag_part_tol
        self._allowed_num_instability_error = allowed_num_instability_error

    def _evolve_helper(
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
                parametrized state/ansatz.
            hamiltonian:
                Operator used for Variational Quantum Imaginary Time Evolution (VarQTE)
                The coefficient of the operator (``operator.coeff``) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a
                ``ComboFn``.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            initial_state: Quantum state to be evolved.
            error_calculator: Calculator of errors for error-based ODE functions.


        Returns:
            Result of the evolution which is a quantum circuit with bound parameters as an
            evolved state.
        """

        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameters_values = list(init_state_param_dict.values())

        metric_tensor = self._variational_principle.calc_metric_tensor(
            initial_state, init_state_parameters
        )
        evolution_grad = self._variational_principle.calc_evolution_grad(
            hamiltonian, initial_state, init_state_parameters
        )

        linear_solver = VarQTELinearSolver(
            metric_tensor,
            evolution_grad,
            self._lse_solver_callable,
            self._circuit_sampler,
            self._imag_part_tol,
        )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        self._ode_function_generator._lazy_init(
            linear_solver,
            error_calculator,
            t_param,
            init_state_param_dict,
        )

        ode_solver = VarQTEOdeSolver(
            init_state_parameters_values, self._ode_function_generator, self._ode_solver_callable
        )
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

    # TODO handle the case where quantum state params are not present in a dictionary; possibly
    #  rename the dictionary because it not only relates to a Hamiltonian but also to a state
    def _create_init_state_param_dict(
        self,
        hamiltonian_value_dict: Dict[Parameter, complex],
        init_state_parameters: List[Parameter],
    ) -> Dict[Parameter, complex]:
        r"""
        Looks for parameters present in an initial state (an ansatz) in a ``hamiltonian_value_dict``
        provided. Based on that, it creates a new dictionary containing only parameters present
        in an initial state and their respective values. If no ``hamiltonian_value_dict`` is
        present, values are chosen uniformly at random.

        Args:
            hamiltonian_value_dict: Dictionary which relates parameter values to the parameters.
            init_state_parameters: Parameters present in a quantum state.

        Returns:
            Dictionary that maps parameters of an initial state to some values.
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

    def _validate_aux_ops(self, evolution_problem: EvolutionProblem) -> None:
        if evolution_problem.aux_operators is not None and (
            self._backend is None or self._expectation is None
        ):
            raise ValueError(
                "aux_operators where provided for evaluations but no ``expectation`` or "
                "``quantum_instance`` was provided."
            )
