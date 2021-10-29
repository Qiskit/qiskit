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
from typing import Optional, Union, List, Dict

import numpy as np

from qiskit.circuit import ParameterVector, ParameterExpression, Parameter
from qiskit.opflow import (
    NaturalGradient,
    OperatorBase,
    CircuitGradient,
    CircuitQFI,
    CircuitSampler,
)
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


def calculate(
    operator: OperatorBase,
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    grad_method: Union[str, CircuitGradient] = "lin_comb",
    qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    regularization: str = None,
):
    nat_grad = NaturalGradient(
        grad_method=grad_method,
        qfi_method=qfi_method,
        regularization=regularization,
    ).convert(operator * 0.5, parameters)

    return nat_grad


def eval_nat_grad_result(
    nat_grad,
    param_dict: Dict[Parameter, Union[float, complex]],
    nat_grad_circ_sampler: CircuitSampler,
    backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
):
    if backend is not None:
        nat_grad_result = nat_grad_circ_sampler.convert(nat_grad, params=param_dict).eval()
    else:
        nat_grad_result = nat_grad.assign_parameters(param_dict).eval()

    if any(np.abs(np.imag(nat_grad_item)) > 1e-8 for nat_grad_item in nat_grad_result):
        raise Warning("The imaginary part of the gradient are non-negligible.")

    return nat_grad_result
