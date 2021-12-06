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
from typing import Union, Optional, List, Dict, Callable

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
    """
    Calculates a parametrized evolution gradient object.
    Args:
        observable: Observable for which an evolution gradient should be calculated,
                    e.g., a Hamiltonian of a system.
        ansatz: Quantum state to be evolved.
        parameters: Parameters with respect to which gradients should be computed.
        grad_method: The method used to compute the state gradient. Can be either
                    ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
        basis: Basis with respect to which evolution gradient is calculated. In case of a default
                Z basis a real part of an evolution gradient is returned. In case of a Y basis,
                an imaginary part of an evolution gradient is returned.
    Returns:
        Parametrized evolution gradient as an OperatorBase.
    """
    operator = ~StateFn(observable) @ StateFn(ansatz)
    if grad_method == "lin_comb":
        return LinComb().convert(operator, parameters, aux_meas_op=basis)
    return Gradient(grad_method).convert(operator, parameters)


def eval_grad_result(
    grad: Union[OperatorBase, Callable[[Dict[Parameter, float], CircuitSampler], OperatorBase]],
    param_dict: Dict[Parameter, Union[float, complex]],
    grad_circ_sampler: Optional[CircuitSampler] = None,
    energy_sampler: Optional[CircuitSampler] = None,
) -> OperatorBase:
    """Binds a parametrized evolution grad object to parameters values provided. Uses circuit
    samplers if available.
    Args:
        grad: Either an evolution gradient as an OperatorBase to be evaluated or a callable that
            constructs an OperatorBase from a dictionairy of parameters and evalues and potentially a
            CircuitSampler.
        param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        grad_circ_sampler: CircuitSampler for evolution gradients.
        energy_sampler: CircuitSampler for energy.
    Returns:
        Evolution gradient with all parameters bound.
    """
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
