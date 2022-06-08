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

"""Class for a Variational Principle."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    QFI,
    Gradient,
    CircuitStateFn,
)
from qiskit.opflow.gradients.circuit_gradients import LinComb


class VariationalPrinciple(ABC):
    """A Variational Principle class. It determines the time propagation of parameters in a
    quantum state provided as a parametrized quantum circuit (ansatz)."""

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
    ) -> None:
        """
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        self._qfi_method = qfi_method
        self.qfi = QFI(qfi_method)
        self._grad_method = grad_method
        if self._grad_method == "lin_comb":
            self._grad_method = LinComb()
        self._evolution_gradient = Gradient(self._grad_method)
        self._qfi_gradient_callable = None

    def calc_metric_tensor(
        self, ansatz, bind_params, gradient_params, quantum_instance, param_values
    ) -> np.ndarray:
        """
        Created a QFI instance according to the rules of this variational principle. It will be used
        to calculate a metric tensor required in the ODE.

        Returns:
            QFI instance.
        """
        if self._qfi_gradient_callable is None:
            self._qfi_gradient_callable = self.qfi.gradient_wrapper(
                CircuitStateFn(ansatz), bind_params, gradient_params, quantum_instance
            )
        metric_tensor_lse_lhs = 0.25 * self._qfi_gradient_callable(param_values)

        return metric_tensor_lse_lhs

    def calc_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        circuit_sampler,
        param_dict,
        bind_params,
        gradient_params,
        quantum_instance,
        param_values,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Returns:
            Transformed evolution gradient.
        """
        # TODO calculate and sample energy separately
        modified_hamiltonian = self.modify_hamiltonian(
            hamiltonian, ansatz, circuit_sampler, param_dict
        )
        # TODO calculate energy separately, cache Hamiltonian only grad_callable and apply energy
        #  post factum
        grad_callable = self._evolution_gradient.gradient_wrapper(
            modified_hamiltonian, bind_params, gradient_params, quantum_instance
        )
        evolution_grad_lse_rhs = 0.5 * grad_callable(param_values)

        return evolution_grad_lse_rhs

    @abstractmethod
    def modify_hamiltonian(self, hamiltonian, ansatz, circuit_sampler, param_dict):
        pass
