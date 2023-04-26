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

"""Variational Quantum Real Time Evolution algorithm."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Type, Callable

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BaseEstimator

from .solvers.ode.forward_euler_solver import ForwardEulerSolver

from .variational_principles import RealVariationalPrinciple, RealMcLachlanPrinciple
from .var_qte import VarQTE

from ..real_time_evolver import RealTimeEvolver


class VarQRTE(VarQTE, RealTimeEvolver):
    """Variational Quantum Real Time Evolution algorithm.

    .. code-block::python

        import numpy as np

        from qiskit.algorithms import TimeEvolutionProblem, VarQRTE
        from qiskit.circuit.library import EfficientSU2
        from qiskit.algorithms.time_evolvers.variational import RealMcLachlanPrinciple
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info import SparsePauliOp, Pauli
        from qiskit.primitives import Estimator

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
        init_param_values = np.ones(len(ansatz.parameters)) * np.pi/2
        var_principle = RealMcLachlanPrinciple()
        time = 1

        # without evaluating auxiliary operators
        evolution_problem = TimeEvolutionProblem(observable, time)
        var_qrte = VarQRTE(ansatz, init_param_values, var_principle)
        evolution_result = var_qrte.evolve(evolution_problem)

        # evaluating auxiliary operators
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)
        var_qrte = VarQRTE(ansatz, init_param_values, var_principle, Estimator())
        evolution_result = var_qrte.evolve(evolution_problem)
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: Mapping[Parameter, float] | Sequence[float],
        variational_principle: RealVariationalPrinciple | None = None,
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
            initial_parameters: Initial parameter values for an ansatz.
            variational_principle: Variational Principle to be used. Defaults to
                ``RealMcLachlanPrinciple``.
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
            initial_parameters,
            variational_principle,
            estimator,
            ode_solver,
            lse_solver=lse_solver,
            num_timesteps=num_timesteps,
            imag_part_tol=imag_part_tol,
            num_instability_tol=num_instability_tol,
        )
