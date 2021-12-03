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
from typing import Union, Optional, List, Dict

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression, Parameter
from qiskit.opflow import StateFn, Gradient, CircuitGradient, OperatorBase, Z, CircuitSampler
from qiskit.opflow.gradients.circuit_gradients import LinComb


def calculate(
    observable: OperatorBase,
    ansatz: Union[StateFn, QuantumCircuit],
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    grad_method: Union[str, CircuitGradient],
    basis: OperatorBase = Z,
) -> OperatorBase:
    """Calculates a parametrized evolution gradient object."""
    operator = ~StateFn(observable) @ StateFn(ansatz)
    if grad_method == "lin_comb":
        return LinComb().convert(operator, parameters, aux_meas_op=basis)
    return Gradient(grad_method).convert(operator, parameters)


def eval_grad_result(
    grad: Union[OperatorBase, callable],
    param_dict: Dict[Parameter, Union[float, complex]],
    grad_circ_sampler: CircuitSampler = None,
    energy_sampler: CircuitSampler = None,
) -> OperatorBase:
    """Binds a parametrized evolution grad object to parameters values provided. Uses circuit
    samplers if available."""
    # TODO would be nicer to somehow get rid of this if statement
    if isinstance(grad, OperatorBase):
        grad_result = grad
    else:
        grad_result = grad(param_dict, energy_sampler)

    if grad_circ_sampler:
        grad_result = grad_circ_sampler.convert(grad_result, param_dict)
    else:
        grad_result = grad_result.assign_parameters(param_dict)
    grad_result = grad_result.eval()
    if any(np.abs(np.imag(grad_item)) > 1e-8 for grad_item in grad_result):
        raise Warning("The imaginary part of the gradient are non-negligible.")

    return grad_result
