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

from typing import Union, List, Dict, Optional, Callable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms.evolvers.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitSampler,
    OperatorBase,
    ExpectationBase,
)
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        var_principle: VariationalPrinciple,
        hamiltonian: OperatorBase,
        ansatz: QuantumCircuit,
        gradient_params: List[Parameter],
        t_param: Optional[Parameter] = None,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        imag_part_tol: float = 1e-7,
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        """
        Args:
            var_principle: Variational Principle to be used.
            hamiltonian:
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
            expectation: An instance of ``ExpectationBase`` used for calculating a metric tensor
                and an evolution gradient. If ``None`` provided, a ``PauliExpectation`` is used.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
        """
        self._var_principle = var_principle
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._gradient_params = gradient_params
        self._bind_params = gradient_params + [t_param] if t_param else gradient_params
        self._time_param = t_param
        self.lse_solver = lse_solver
        self._quantum_instance = None
        self._circuit_sampler = None
        self._imag_part_tol = imag_part_tol
        self._expectation = expectation
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

    @property
    def lse_solver(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Returns an LSE solver callable."""
        return self._lse_solver

    @lse_solver.setter
    def lse_solver(
        self, lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    ) -> None:
        """Sets an LSE solver. Uses a ``np.linalg.lstsq`` callable if ``None`` provided."""
        if lse_solver is None:
            lse_solver = lambda a, b: np.linalg.lstsq(a, b, rcond=1e-2)[0]

        self._lse_solver = lse_solver

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

    def solve_lse(
        self,
        param_dict: Dict[Parameter, complex],
        time_value: Optional[float] = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
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
        if self._time_param is not None:
            param_values.append(time_value)

        metric_tensor_lse_lhs = self._var_principle.metric_tensor(
            self._ansatz,
            self._bind_params,
            self._gradient_params,
            param_values,
            self._expectation,
            self._quantum_instance,
        )
        evolution_grad_lse_rhs = self._var_principle.evolution_grad(
            self._hamiltonian,
            self._ansatz,
            self._circuit_sampler,
            param_dict,
            self._bind_params,
            self._gradient_params,
            param_values,
            self._expectation,
            self._quantum_instance,
        )

        x = self._lse_solver(metric_tensor_lse_lhs, evolution_grad_lse_rhs)

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs
