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
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler, OperatorBase, QFI, CircuitStateFn, StateFn
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider
from ..calculators.evolution_grad_calculator import (
    eval_grad_result,
)


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        ansatz: Union[StateFn, QuantumCircuit],
        qfi: QFI,
        gradient_params: List[Parameter],
        evolution_grad: OperatorBase,
        t_param=None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        quantum_instance: Optional[QuantumInstance] = None,
        imag_part_tol: float = 1e-7,
    ) -> None:
        """
        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            qfi: A Quantum Fisher Information instance used to calculate a metric tensor for the
                left-hand side of an ODE.
            gradient_params: List of parameters with respect to which gradients should be computed.
            evolution_grad: A parametrized operator that represents the right-hand side of an ODE.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            lse_solver: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
        """
        self._ansatz = ansatz
        self._qfi = qfi
        self._gradient_params = gradient_params
        bind_params = gradient_params + [t_param] if t_param else gradient_params
        self._qfi_gradient_callable = qfi.gradient_wrapper(
            CircuitStateFn(ansatz), bind_params, gradient_params, quantum_instance
        )
        self._evolution_grad = evolution_grad
        self._time_param = t_param
        self._lse_solver = lse_solver
        self._quantum_instance = None
        self._circuit_sampler = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance
        self._imag_part_tol = imag_part_tol

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
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.

        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            time_value: Time value that will be bound to ``t_param``. It is required if ``t_param``
                is not ``None``.

        Returns:
            Solution to the LSE, A from Ax=b, b from Ax=b.
        """

        metric_tensor_lse_lhs = 0.25 * self._qfi_gradient_callable(list(param_dict.values()))
        evolution_grad_lse_rhs = self._calc_lse_rhs(param_dict, t_param, time_value)

        x = self._lse_solver(metric_tensor_lse_lhs, evolution_grad_lse_rhs)[0]

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs

    def _calc_lse_rhs(
        self,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
    ) -> OperatorBase:

        grad = self._evolution_grad

        if t_param is not None:
            time_dict = {t_param: time_value}
            grad = self._evolution_grad.bind_parameters(time_dict)

        evolution_grad_lse_rhs = eval_grad_result(
            grad,
            param_dict,
            self._circuit_sampler,
            self._imag_part_tol,
        )

        return evolution_grad_lse_rhs
