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

"""Variational Quantum Real Time Evolution algorithm."""
from functools import partial
from typing import Optional, Callable, Union

import numpy as np
from scipy.integrate import OdeSolver

from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.algorithms.evolvers import EvolutionProblem, EvolutionResult
from qiskit.algorithms.evolvers.real_evolver import RealEvolver
from qiskit.opflow import (
    StateFn,
    ExpectationBase,
)
from qiskit.utils import QuantumInstance
from .solvers.ode.ode_function_factory import OdeFunctionFactory
from .var_qte import VarQTE
from .variational_principles.real_variational_principle import (
    RealVariationalPrinciple,
)


class VarQRTE(RealEvolver, VarQTE):
    """Variational Quantum Real Time Evolution algorithm."""

    def __init__(
        self,
        variational_principle: RealVariationalPrinciple,
        ode_function_factory: OdeFunctionFactory,
        ode_solver: Union[OdeSolver, str] = "RK45",
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
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
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
        super().__init__(
            variational_principle,
            ode_function_factory,
            ode_solver,
            lse_solver,
            expectation,
            imag_part_tol,
            num_instability_tol,
            quantum_instance,
        )

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Apply Variational Quantum Real Time Evolution (VarQRTE) w.r.t. the given operator.

        Args:
            evolution_problem: Instance defining an evolution problem. If no initial parameter
                values are provided in ``param_value_dict``, they are initialized uniformly at
                random.

        Returns:
            Result of the evolution which includes a quantum circuit with bound parameters as an
            evolved state and, if provided, observables evaluated on the evolved state using
            a ``quantum_instance`` and ``expectation`` provided.
        """
        self._validate_aux_ops(evolution_problem)
        init_state_param_dict = self._create_init_state_param_dict(
            evolution_problem.param_value_dict,
            list(evolution_problem.initial_state.parameters),
        )

        error_calculator = None  # TODO will be supported in another PR

        evolved_state = super()._evolve(
            init_state_param_dict,
            evolution_problem.hamiltonian,
            evolution_problem.time,
            evolution_problem.t_param,
            evolution_problem.initial_state,
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
