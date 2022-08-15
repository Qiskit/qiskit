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
from typing import Optional, Union, Dict, List, Any, Type, Callable

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.algorithms.evolvers.evolution_result import EvolutionResult
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    CircuitSampler,
    OperatorBase,
    ExpectationBase,
)
from qiskit.utils.backend_utils import is_aer_provider
from .solvers.ode.forward_euler_solver import ForwardEulerSolver
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

    Algorithms that use variational principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state prepared by a parameterized quantum circuit.

    References:

        [1] Benjamin, Simon C. et al. (2019).
        Theory of variational quantum simulation. `<https://doi.org/10.22331/q-2019-10-07-191>`_
    """

    def __init__(
        self,
        ansatz: Union[OperatorBase, QuantumCircuit],
        variational_principle: VariationalPrinciple,
        initial_parameters: Optional[
            Union[Dict[Parameter, complex], List[complex], np.ndarray]
        ] = None,
        ode_solver: Union[Type[OdeSolver], str] = ForwardEulerSolver,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        num_timesteps: Optional[int] = None,
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            ansatz: Ansatz to be used for variational time evolution.
            variational_principle: Variational Principle to be used.
            initial_parameters: Initial parameter values for an ansatz. If ``None`` provided,
                they are initialized uniformly at random.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            num_timesteps: The number of timesteps to take. If None, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                a metric tensor and an evolution gradient and, if provided, expectation values of
                ``EvolutionProblem.aux_operators``.
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
        self.ansatz = ansatz
        self.variational_principle = variational_principle
        self.initial_parameters = initial_parameters
        self._quantum_instance = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance
        self.expectation = expectation
        self.num_timesteps = num_timesteps
        self.lse_solver = lse_solver
        # OdeFunction abstraction kept for potential extensions - unclear at the moment;
        # currently hidden from the user
        self._ode_function_factory = OdeFunctionFactory(lse_solver=lse_solver)
        self.ode_solver = ode_solver
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

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given
        operator.

        Args:
            evolution_problem: Instance defining an evolution problem.
        Returns:
            Result of the evolution which includes a quantum circuit with bound parameters as an
            evolved state and, if provided, observables evaluated on the evolved state using
            a ``quantum_instance`` and ``expectation`` provided.

        Raises:
            ValueError: If no ``initial_state`` is included in the ``evolution_problem``.
        """
        self._validate_aux_ops(evolution_problem)

        if evolution_problem.initial_state is not None:
            raise ValueError("initial_state provided but not applicable to VarQTE.")

        init_state_param_dict = self._create_init_state_param_dict(
            self.initial_parameters, self.ansatz.parameters
        )

        error_calculator = None  # TODO will be supported in another PR

        evolved_state = self._evolve(
            init_state_param_dict,
            evolution_problem.hamiltonian,
            evolution_problem.time,
            evolution_problem.t_param,
            error_calculator,
        )

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = eval_observables(
                self.quantum_instance,
                evolved_state,
                evolution_problem.aux_operators,
                self.expectation,
            )

        return EvolutionResult(evolved_state, evaluated_aux_ops)

    def _evolve(
        self,
        init_state_param_dict: Dict[Parameter, complex],
        hamiltonian: OperatorBase,
        time: float,
        t_param: Optional[Parameter] = None,
        error_calculator: Any = None,
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
            error_calculator: Not yet supported. Calculator of errors for error-based ODE functions.

        Returns:
            Result of the evolution which is a quantum circuit with bound parameters as an
            evolved state.
        """

        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameters_values = list(init_state_param_dict.values())

        linear_solver = VarQTELinearSolver(
            self.variational_principle,
            hamiltonian,
            self.ansatz,
            init_state_parameters,
            t_param,
            self._ode_function_factory.lse_solver,
            self.imag_part_tol,
            self.expectation,
            self._quantum_instance,
        )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        ode_function = self._ode_function_factory._build(
            linear_solver, error_calculator, init_state_param_dict, t_param
        )

        ode_solver = VarQTEOdeSolver(
            init_state_parameters_values, ode_function, self.ode_solver, self.num_timesteps
        )
        parameter_values = ode_solver.run(time)
        param_dict_from_ode = dict(zip(init_state_parameters, parameter_values))

        return self.ansatz.assign_parameters(param_dict_from_ode)

    @staticmethod
    def _create_init_state_param_dict(
        param_values: Union[Dict[Parameter, complex], List[complex], np.ndarray],
        init_state_parameters: List[Parameter],
    ) -> Dict[Parameter, complex]:
        r"""
        If ``param_values`` is a dictionary, it looks for parameters present in an initial state
        (an ansatz) in a ``param_values``. Based on that, it creates a new dictionary containing
        only parameters present in an initial state and their respective values.
        If ``param_values`` is a list of values, it creates a new dictionary containing
        parameters present in an initial state and their respective values.
        If no ``param_values`` is provided, parameter values are chosen uniformly at random.

        Args:
            param_values: Dictionary which relates parameter values to the parameters or a list of
                values.
            init_state_parameters: Parameters present in a quantum state.

        Returns:
            Dictionary that maps parameters of an initial state to some values.

        Raises:
            ValueError: If the dictionary with parameter values provided does not include all
                parameters present in the initial state or if the list of values provided is not the
                same length as the list of parameters.
            TypeError: If an unsupported type of ``param_values`` provided.
        """
        if param_values is None:
            init_state_parameter_values = np.random.random(len(init_state_parameters))
        elif isinstance(param_values, dict):
            init_state_parameter_values = []
            for param in init_state_parameters:
                if param in param_values.keys():
                    init_state_parameter_values.append(param_values[param])
                else:
                    raise ValueError(
                        f"The dictionary with parameter values provided does not "
                        f"include all parameters present in the initial state."
                        f"Parameters present in the state: {init_state_parameters}, "
                        f"parameters in the dictionary: "
                        f"{list(param_values.keys())}."
                    )
        elif isinstance(param_values, (list, np.ndarray)):
            if len(init_state_parameters) != len(param_values):
                raise ValueError(
                    f"Initial state has {len(init_state_parameters)} parameters and the"
                    f" list of values has {len(param_values)} elements. They should be"
                    f"equal in length."
                )
            init_state_parameter_values = param_values
        else:
            raise TypeError(f"Unsupported type of param_values provided: {type(param_values)}.")

        init_state_param_dict = dict(zip(init_state_parameters, init_state_parameter_values))
        return init_state_param_dict

    def _validate_aux_ops(self, evolution_problem: EvolutionProblem) -> None:
        if evolution_problem.aux_operators is not None:
            if self.quantum_instance is None:
                raise ValueError(
                    "aux_operators where provided for evaluations but no ``quantum_instance`` "
                    "was provided."
                )

            if self.expectation is None:
                raise ValueError(
                    "aux_operators where provided for evaluations but no ``expectation`` "
                    "was provided."
                )
