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
from qiskit.opflow import (
    CircuitSampler,
    OperatorBase,
    QFI,
    CircuitStateFn,
    StateFn,
    Gradient,
)
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_aer_provider


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        ansatz: Union[StateFn, QuantumCircuit],
        qfi: QFI,
        gradient_params: List[Parameter],
        evolution_grad: Gradient,
        modified_hamiltonian_callable: OperatorBase,
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
        self._bind_params = gradient_params + [t_param] if t_param else gradient_params
        # print(bind_params)
        # print(gradient_params)
        self._qfi_gradient_callable = qfi.gradient_wrapper(
            CircuitStateFn(ansatz), self._bind_params, gradient_params, quantum_instance
        )
        # print(gradient_operator)
        self._evolution_grad = evolution_grad
        self._modified_hamiltonian_callable = modified_hamiltonian_callable
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
        param_values = list(param_dict.values())
        if self._time_param is not None:
            param_values.append(time_value)
            t_dict = {t_param: time_value}
        # print("Param vals")
        # print(param_values)
        metric_tensor_lse_lhs = 0.25 * self._qfi_gradient_callable(param_values)
        modified_hamiltonian = self._modified_hamiltonian_callable(param_dict)
        #print(modified_hamiltonian)
        grad_callable = self._evolution_grad.gradient_wrapper(
            modified_hamiltonian, self._bind_params, self._gradient_params, self._quantum_instance
        )
        #print("Grad call")
        evolution_grad_lse_rhs = 0.5 * grad_callable(param_values)

        # print(metric_tensor_lse_lhs)
        # print(evolution_grad_lse_rhs)
        # print(type(evolution_grad_lse_rhs))

        if self._time_param is not None:
            bound_evolution_grad_lse_rhs = np.zeros(len(evolution_grad_lse_rhs), dtype=complex)
            for i, param_expr in enumerate(evolution_grad_lse_rhs):
                bound_evolution_grad_lse_rhs[i] = param_expr.assign(
                    self._time_param, time_value
                ).__complex__()
            # print(bound_evolution_grad_lse_rhs)
            # print(type(bound_evolution_grad_lse_rhs[0]))
            evolution_grad_lse_rhs = bound_evolution_grad_lse_rhs

        #print(-evolution_grad_lse_rhs)
        x = self._lse_solver(metric_tensor_lse_lhs, evolution_grad_lse_rhs)[0]

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs

    # def _calc_lse_rhs(
    #     self,
    #     param_dict: Dict[Parameter, complex],
    #     t_param: Optional[Parameter] = None,
    #     time_value: Optional[float] = None,
    # ) -> OperatorBase:
    #
    #     grad = self._evolution_grad_callable
    #
    #     if t_param is not None:
    #         time_dict = {t_param: time_value}
    #         grad = self._evolution_grad_callable.bind_parameters(time_dict)
    #
    #     evolution_grad_lse_rhs = eval_grad_result(
    #         grad,
    #         param_dict,
    #         self._circuit_sampler,
    #         self._imag_part_tol,
    #     )
    #
    #     return evolution_grad_lse_rhs
