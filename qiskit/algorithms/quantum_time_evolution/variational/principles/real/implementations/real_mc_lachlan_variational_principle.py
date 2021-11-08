# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Union, Dict, Optional, List

from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    evolution_grad_calculator,
    metric_tensor_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.opflow import CircuitQFI, OperatorBase, StateFn, SummedOp, Y, I, \
    PauliExpectation, CircuitSampler


class RealMcLachlanVariationalPrinciple(RealVariationalPrinciple):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ):
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
        """
        super().__init__(
            qfi_method,
        )

    def _get_raw_metric_tensor(
        self,
        ansatz,
        parameters: List[Parameter],
        # param_dict: Dict[Parameter, Union[float, complex]],
    ):
        raw_metric_tensor_real = metric_tensor_calculator.calculate(
            ansatz, parameters, self._qfi_method
        )

        return raw_metric_tensor_real

    def _get_raw_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        parameters: List[Parameter],
    ):

        def raw_evolution_grad_imag(param_dict: Dict,
                                    backend: Optional[Union[BaseBackend, QuantumInstance]] = None):
            energy = ~StateFn(hamiltonian) @ StateFn(ansatz)
            energy = PauliExpectation().convert(energy)
            #TODO rewrite to be more efficient
            if backend is not None:
                energy = CircuitSampler(backend).convert(energy, param_dict).eval()
            else:
                energy = energy.assign_parameters(param_dict).eval()

            hamiltonian_ = SummedOp([hamiltonian, -(1) * energy * I ^ hamiltonian.num_qubits])

            return evolution_grad_calculator.calculate(hamiltonian_, ansatz, parameters,
                                                       self._grad_method, basis=-1j * Y)

        return raw_evolution_grad_imag

    @staticmethod
    def _calc_metric_tensor(
        raw_metric_tensor_real: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return raw_metric_tensor_real.bind_parameters(param_dict) / 4.0

    @staticmethod
    def _calc_evolution_grad(
        raw_evolution_grad_imag: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return raw_evolution_grad_imag.bind_parameters(param_dict)

    def _calc_nat_grad(
        self,
        raw_operator: OperatorBase,
        param_dict: Dict[Parameter, Union[float, complex]],
        regularization: Optional[str] = None,
    ) -> OperatorBase:
        return super()._calc_nat_grad(raw_operator, param_dict, regularization)
