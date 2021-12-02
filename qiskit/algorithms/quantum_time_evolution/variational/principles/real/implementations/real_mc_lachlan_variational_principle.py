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
from typing import Union, Dict, List, Callable

from qiskit import QuantumCircuit
from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    evolution_grad_calculator,
    metric_tensor_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    StateFn,
    SummedOp,
    Y,
    I,
    PauliExpectation,
    CircuitSampler,
    ListOp,
    OperatorBase,
)


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

    def _get_metric_tensor(
        self,
        ansatz: QuantumCircuit,
        parameters: List[Parameter],
    ) -> ListOp:
        raw_metric_tensor_real = metric_tensor_calculator.calculate(
            ansatz, parameters, self._qfi_method
        )

        return raw_metric_tensor_real * 0.25  # QFI/4

    def _get_evolution_grad(
        self,
        hamiltonian,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> Callable:
        def raw_evolution_grad_imag(
            param_dict: Dict[Parameter, Union[float, complex]], energy_sampler: CircuitSampler
        ) -> OperatorBase:
            energy = ~StateFn(hamiltonian) @ StateFn(ansatz)
            energy = PauliExpectation().convert(energy)

            if energy_sampler:
                energy = energy_sampler.convert(energy, param_dict).eval()
            else:
                energy = energy.assign_parameters(param_dict).eval()

            energy_term = I ^ hamiltonian.num_qubits
            energy_term *= -1
            energy_term *= energy
            hamiltonian_ = SummedOp([hamiltonian, energy_term]).reduce()
            basis_operator = Y
            grad = (
                evolution_grad_calculator.calculate(
                    hamiltonian_, ansatz, parameters, self._grad_method, basis=basis_operator
                )
                * 0.5
            )  # Im(...)
            return grad

        return raw_evolution_grad_imag
