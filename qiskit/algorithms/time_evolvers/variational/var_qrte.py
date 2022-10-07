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
from __future__ import annotations

from typing import Type, Callable

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.algorithms.time_evolvers.real_time_evolver import RealTimeEvolver
from qiskit.circuit import Parameter
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from . import RealMcLachlanPrinciple
from .solvers.ode.forward_euler_solver import ForwardEulerSolver
from .variational_principles import RealVariationalPrinciple
from .var_qte import VarQTE


class VarQRTE(VarQTE, RealTimeEvolver):
    """Variational Quantum Real Time Evolution algorithm.

    .. code-block::python

        from qiskit.algorithms import TimeEvolutionProblem, VarQITE
        from qiskit.circuit.library import EfficientSU2
        from qiskit.algorithms.time_evolvers.variational import RealMcLachlanPrinciple
        from qiskit.quantum_info import SparsePauliOp
        import numpy as np

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        ansatz = EfficientSU2(observable.num_qubits, reps=1)
        init_param_values = np.zeros(len(ansatz.parameters))
        for i in range(len(ansatz.parameters)):
            init_param_values[i] = np.pi / 2
        var_principle = RealMcLachlanPrinciple()
        time = 1
        evolution_problem = TimeEvolutionProblem(observable, time)
        var_qrte = VarQRTE(ansatz, var_principle, init_param_values)
        evolution_result = var_qite.evolve(evolution_problem)
    """

    def __init__(
        self,
        ansatz: BaseOperator | QuantumCircuit,
        variational_principle: RealVariationalPrinciple | None = None,
        initial_parameters: dict[Parameter, complex] | list[complex] | np.ndarray | None = None,
        estimator: BaseEstimator | None = None,
        ode_solver: Type[OdeSolver] | str = ForwardEulerSolver,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_timesteps: int | None = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
    ) -> None:
        r"""
        Args:
            ansatz: Ansatz to be used for variational time evolution.
            variational_principle: Variational Principle to be used. Defaults to
                ``RealMcLachlanPrinciple``.
            initial_parameters: Initial parameter values for an ansatz. If ``None`` provided,
                they are initialized uniformly at random.
            estimator: An estimator primitive used for calculating expectation values of
                TimeEvolutionProblem.aux_operators.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            num_timesteps: The number of timesteps to take. If ``None``, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
        """
        if variational_principle is None:
            variational_principle = RealMcLachlanPrinciple()
        super().__init__(
            ansatz,
            variational_principle,
            estimator,
            initial_parameters,
            ode_solver,
            lse_solver=lse_solver,
            num_timesteps=num_timesteps,
            imag_part_tol=imag_part_tol,
            num_instability_tol=num_instability_tol,
        )
