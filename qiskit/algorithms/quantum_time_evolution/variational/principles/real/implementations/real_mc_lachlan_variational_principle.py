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
from typing import Union

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.real_error_calculator import (
    RealErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.opflow import CircuitQFI, CircuitGradient, OperatorBase


class RealMcLachlanVariationalPrinciple(RealVariationalPrinciple):
    def __init__(
        self,
        observable,
        ansatz,
        parameters,
        error_calculator: RealErrorCalculator,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
        is_error_supported: bool = False,
    ):
        super().__init__(
            observable,
            ansatz,
            parameters,
            error_calculator,
            qfi_method,
            grad_method,
            is_error_supported,
        )

    @staticmethod
    def _calc_metric_tensor(raw_metric_tensor: OperatorBase) -> OperatorBase:
        return VariationalPrinciple.op_real_part(raw_metric_tensor)

    @staticmethod
    def _calc_evolution_grad(raw_evolution_grad: OperatorBase) -> OperatorBase:
        # TODO verify
        return VariationalPrinciple.op_imag_part(raw_evolution_grad)
