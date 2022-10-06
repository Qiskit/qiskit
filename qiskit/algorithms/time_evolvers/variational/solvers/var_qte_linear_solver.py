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

"""Class for solving linear equations for Quantum Time Evolution."""
from __future__ import annotations

from typing import Callable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.time_evolvers.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        var_principle: VariationalPrinciple,
        hamiltonian: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        gradient_params: list[Parameter],
        t_param: Parameter | None = None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        imag_part_tol: float = 1e-7,
    ) -> None:
        """
        Args:
            var_principle: Variational Principle to be used.
            hamiltonian: TODO update types
                Operator used for Variational Quantum Time Evolution.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a
                ``ComboFn``.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            gradient_params: List of parameters with respect to which gradients should be computed.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
        """
        self._var_principle = var_principle
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._gradient_params = gradient_params
        self._bind_params = gradient_params
        self._time_param = t_param
        self.lse_solver = lse_solver
        self._imag_part_tol = imag_part_tol

    @property
    def lse_solver(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Returns an LSE solver callable."""
        return self._lse_solver

    @lse_solver.setter
    def lse_solver(self, lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None) -> None:
        """Sets an LSE solver. Uses a ``np.linalg.lstsq`` callable if ``None`` provided."""
        if lse_solver is None:
            lse_solver = lambda a, b: np.linalg.lstsq(a, b, rcond=1e-2)[0]

        self._lse_solver = lse_solver

    def solve_lse(
        self,
        param_dict: dict[Parameter, complex],
        time_value: float | None = None,
    ) -> (list | np.ndarray, list | np.ndarray, np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.

        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            time_value: Time value that will be bound to ``t_param``. It is required if ``t_param``
                is not ``None``.

        Returns:
            Solution to the LSE, A from Ax=b, b from Ax=b.
        """
        param_values = list(param_dict.values())
        metric_tensor_lse_lhs = self._var_principle.metric_tensor(self._ansatz, param_values)
        if self._time_param is not None:
            # TODO this turns into OperatorBase
            self._hamiltonian = self._hamiltonian.assign_parameters({self._time_param: time_value})

        evolution_grad_lse_rhs = self._var_principle.evolution_grad(
            self._hamiltonian,
            self._ansatz,
            param_dict,
            self._bind_params,
            self._gradient_params,
            param_values,
        )

        x = self._lse_solver(metric_tensor_lse_lhs, evolution_grad_lse_rhs)

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs
