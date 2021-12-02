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
    operator = ~StateFn(observable) @ StateFn(ansatz)
    if grad_method == "lin_comb":
        return LinComb().convert(operator, parameters, aux_meas_op=basis)
    return Gradient(grad_method).convert(operator, parameters)


def eval_evolution_grad(
    evolution_grad: OperatorBase,
    param_dict: Dict[Parameter, Union[float, complex]],
    grad_circ_sampler: CircuitSampler,
) -> np.ndarray:
    if grad_circ_sampler:
        grad_res = np.array(grad_circ_sampler.convert(evolution_grad, params=param_dict).eval())
    else:
        grad_res = np.array(evolution_grad.assign_parameters(param_dict).eval())
    return grad_res
