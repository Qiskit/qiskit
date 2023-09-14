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

"""Class for solving linear equations for Quantum Time Evolution."""
from __future__ import annotations

from collections.abc import Mapping, Sequence, Callable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..variational_principles import VariationalPrinciple


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        var_principle: VariationalPrinciple,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        gradient_params: Sequence[Parameter] | None = None,
        t_param: Parameter | None = None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        imag_part_tol: float = 1e-7,
    ) -> None:
        """
        Args:
            var_principle: Variational Principle to be used.
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.

        Raises:
            TypeError: If t_param is provided and Hamiltonian is not of type SparsePauliOp.
        """
        self._var_principle = var_principle
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._gradient_params = gradient_params
        self._bind_params = gradient_params
        self._time_param = t_param
        self.lse_solver = lse_solver
        self._imag_part_tol = imag_part_tol

        if self._time_param is not None and not isinstance(self._hamiltonian, SparsePauliOp):
            raise TypeError(
                f"A time parameter {t_param} has been specified, so a time-dependent "
                f"hamiltonian is expected. The operator provided is of type {type(self._hamiltonian)}, "
                f"which might not support parametrization. "
                f"Please provide the parametrized hamiltonian as a SparsePauliOp."
            )

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
        param_dict: Mapping[Parameter, float],
        time_value: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.

        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            time_value: Time value that will be bound to ``t_param``. It is required if ``t_param``
                is not ``None``.

        Returns:
            Solution to the LSE, A from Ax=b, b from Ax=b.

        Raises:
            ValueError: If no time value is provided for time dependent hamiltonians.

        """
        param_values = list(param_dict.values())
        metric_tensor_lse_lhs = self._var_principle.metric_tensor(self._ansatz, param_values)
        hamiltonian = self._hamiltonian

        if self._time_param is not None:
            if time_value is not None:
                hamiltonian = hamiltonian.assign_parameters([time_value])
            else:
                raise ValueError(
                    "Providing a time_value is required for time-dependent hamiltonians, "
                    f"but got time_value = {time_value}. "
                    "Please provide a time_value to the solve_lse method."
                )

        evolution_grad_lse_rhs = self._var_principle.evolution_gradient(
            hamiltonian, self._ansatz, param_values, self._gradient_params
        )

        x = self._lse_solver(metric_tensor_lse_lhs, evolution_grad_lse_rhs)

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs
