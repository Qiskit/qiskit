# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Time Evolution Interface"""
from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Callable, Sequence
from typing import Type

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .solvers.ode.forward_euler_solver import ForwardEulerSolver
from .solvers.ode.ode_function_factory import OdeFunctionFactory
from .solvers.ode.var_qte_ode_solver import VarQTEOdeSolver
from .solvers.var_qte_linear_solver import VarQTELinearSolver

from .variational_principles.variational_principle import VariationalPrinciple
from .var_qte_result import VarQTEResult

from ..time_evolution_problem import TimeEvolutionProblem

from ...observables_evaluator import estimate_observables


class VarQTE(ABC):
    """Variational Quantum Time Evolution.

    Algorithms that use variational principles to compute a time evolution for a given
    Hermitian operator (Hamiltonian) and a quantum state prepared by a parameterized quantum
    circuit.

    Attributes:
            ansatz (QuantumCircuit): Ansatz to be used for variational time evolution.
            initial_parameters (Mapping[Parameter, float] | Sequence[float]): Initial
                parameter values for an ansatz.
            variational_principle (VariationalPrinciple): Variational Principle to be used.
            estimator (BaseEstimator): An estimator primitive used for calculating expectation
                values of ``TimeEvolutionProblem.aux_operators``.
            ode_solver(Type[OdeSolver] | str): ODE solver callable that implements a SciPy
                ``OdeSolver`` interface or a string indicating a valid method offered by SciPy.
            lse_solver (Callable[[np.ndarray, np.ndarray], np.ndarray] | None): Linear system
                of equations solver callable. It accepts ``A`` and ``b`` to solve ``Ax=b``
                and returns ``x``.
            num_timesteps (int | None): The number of timesteps to take. If None, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
            imag_part_tol (float): Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol (float): The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
    References:

        [1] Benjamin, Simon C. et al. (2019).
        Theory of variational quantum simulation. `<https://doi.org/10.22331/q-2019-10-07-191>`_
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: Mapping[Parameter, float] | Sequence[float],
        variational_principle: VariationalPrinciple,
        estimator: BaseEstimator,
        ode_solver: Type[OdeSolver] | str = ForwardEulerSolver,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_timesteps: int | None = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
    ) -> None:
        r"""
        Args:
            ansatz: Ansatz to be used for variational time evolution.
            initial_parameters: Initial parameter values for an ansatz.
            variational_principle: Variational Principle to be used.
            estimator: An estimator primitive used for calculating expectation values of
                TimeEvolutionProblem.aux_operators.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``.
            num_timesteps: The number of timesteps to take. If None, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
        """
        super().__init__()
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.variational_principle = variational_principle
        self.estimator = estimator
        self.num_timesteps = num_timesteps
        self.lse_solver = lse_solver
        self.ode_solver = ode_solver
        self.imag_part_tol = imag_part_tol
        self.num_instability_tol = num_instability_tol
        # OdeFunction abstraction kept for potential extensions - unclear at the moment;
        # currently hidden from the user
        self._ode_function_factory = OdeFunctionFactory()

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> VarQTEResult:
        """Apply Variational Quantum Time Evolution to the given operator.

        Args:
            evolution_problem: Instance defining an evolution problem.
        Returns:
            Result of the evolution which includes a quantum circuit with bound parameters as an
            evolved state and, if provided, observables evaluated on the evolved state.

        Raises:
            ValueError: If ``initial_state`` is included in the ``evolution_problem``.
        """
        self._validate_aux_ops(evolution_problem)

        if evolution_problem.initial_state is not None:
            raise ValueError(
                "An initial_state was provided to the TimeEvolutionProblem but this is not "
                "supported by VarQTE. Please remove this state from the problem definition "
                "and set VarQTE.initial_parameters with the corresponding initial parameter "
                "values instead."
            )

        init_state_param_dict = self._create_init_state_param_dict(
            self.initial_parameters, self.ansatz.parameters
        )

        # unwrap PauliSumOp (in the future this will be deprecated)
        if isinstance(evolution_problem.hamiltonian, PauliSumOp):
            hamiltonian = (
                evolution_problem.hamiltonian.primitive * evolution_problem.hamiltonian.coeff
            )
        else:
            hamiltonian = evolution_problem.hamiltonian

        evolved_state, param_values, time_points = self._evolve(
            init_state_param_dict,
            hamiltonian,
            evolution_problem.time,
            evolution_problem.t_param,
        )

        observables = []
        if evolution_problem.aux_operators is not None:
            for values in param_values:
                # cannot batch evaluation because estimate_observables
                # only accepts single circuits
                evol_state = self.ansatz.assign_parameters(
                    dict(zip(init_state_param_dict.keys(), values))
                )
                observable = estimate_observables(
                    self.estimator,
                    evol_state,
                    evolution_problem.aux_operators,
                )
                observables.append(observable)

        # TODO: deprecate returning evaluated_aux_ops.
        #  As these are the observables for the last time step.
        evaluated_aux_ops = observables[-1] if len(observables) > 0 else None

        return VarQTEResult(
            evolved_state, evaluated_aux_ops, observables, time_points, param_values
        )

    def _evolve(
        self,
        init_state_param_dict: Mapping[Parameter, float],
        hamiltonian: BaseOperator,
        time: float,
        t_param: Parameter | None = None,
    ) -> tuple[QuantumCircuit | None, Sequence[Sequence[float]], Sequence[float]]:
        r"""
        Helper method for performing time evolution. Works both for imaginary and real case.

        Args:
            init_state_param_dict: Parameter dictionary with initial values for a given
                parametrized state/ansatz.
            hamiltonian: Operator used for Variational Quantum Time Evolution (VarQTE).
            time: Total time of evolution.
            t_param: Time parameter in case of a time-dependent Hamiltonian.

        Returns:
            Result of the evolution which is a quantum circuit with bound parameters as an
            evolved state.
        """

        init_state_parameters = list(init_state_param_dict.keys())
        init_state_parameter_values = list(init_state_param_dict.values())

        linear_solver = VarQTELinearSolver(
            self.variational_principle,
            hamiltonian,
            self.ansatz,
            init_state_parameters,
            t_param,
            self.lse_solver,
            self.imag_part_tol,
        )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        ode_function = self._ode_function_factory._build(
            linear_solver, init_state_param_dict, t_param
        )

        ode_solver = VarQTEOdeSolver(
            init_state_parameter_values, ode_function, self.ode_solver, self.num_timesteps
        )
        final_param_values, param_values, time_points = ode_solver.run(time)
        param_dict_from_ode = dict(zip(init_state_parameters, final_param_values))

        return self.ansatz.assign_parameters(param_dict_from_ode), param_values, time_points

    @staticmethod
    def _create_init_state_param_dict(
        param_values: Mapping[Parameter, float] | Sequence[float],
        init_state_parameters: Sequence[Parameter],
    ) -> Mapping[Parameter, float]:
        r"""
        If ``param_values`` is a dictionary, it looks for parameters present in an initial state
        (an ansatz) in a ``param_values``. Based on that, it creates a new dictionary containing
        only parameters present in an initial state and their respective values.
        If ``param_values`` is a list of values, it creates a new dictionary containing
        parameters present in an initial state and their respective values.

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
        if isinstance(param_values, Mapping):
            init_state_parameter_values: Sequence[float] = []
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
        elif isinstance(param_values, (Sequence, np.ndarray)):
            if len(init_state_parameters) != len(param_values):
                raise ValueError(
                    f"Initial state has {len(init_state_parameters)} parameters and the"
                    f" list of values has {len(param_values)} elements. They should be"
                    f" equal in length."
                )
            init_state_parameter_values = param_values
        else:
            raise TypeError(f"Unsupported type of param_values provided: {type(param_values)}.")

        init_state_param_dict = dict(zip(init_state_parameters, init_state_parameter_values))
        return init_state_param_dict

    def _validate_aux_ops(self, evolution_problem: TimeEvolutionProblem) -> None:
        if evolution_problem.aux_operators is not None and self.estimator is None:
            raise ValueError(
                "aux_operators were provided for evaluations but no ``estimator`` was provided."
            )
