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
from typing import Union, Dict

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitQFI, CircuitGradient, OperatorBase


class ImaginaryMcLachlanVariationalPrinciple(ImaginaryVariationalPrinciple):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
    ):
        super().__init__(
            qfi_method,
            grad_method,
        )

    @staticmethod
    def _calc_metric_tensor(
        raw_metric_tensor: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return VariationalPrinciple.op_real_part(raw_metric_tensor.bind_parameters(param_dict))

    @staticmethod
    def _calc_evolution_grad(
        raw_evolution_grad: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return -VariationalPrinciple.op_real_part(raw_evolution_grad.bind_parameters(param_dict))
