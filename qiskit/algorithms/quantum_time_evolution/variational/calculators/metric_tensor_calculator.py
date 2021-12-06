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

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression, Parameter
from qiskit.opflow import QFI, CircuitQFI, CircuitStateFn, OperatorBase, Z, ListOp, CircuitSampler
from qiskit.opflow.gradients.circuit_qfis import LinCombFull

"""Calculates and binds metric tensors for Variational Quantum Time Evolution."""


def calculate(
    ansatz: QuantumCircuit,
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    basis: OperatorBase = Z,
    phase_fix: bool = True,
) -> ListOp:
    """
    Calculates a parametrized metric tensor object.
    Args:
        ansatz: Quantum state to be evolved.
        parameters: Parameters with respect to which gradients should be computed.
        qfi_method: The method used to compute the QFI. Can be either
                    ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
        basis: Basis with respect to which metric tensor is calculated. In case of a default Z
                basis a real part of a metric tensor is returned. In case of a Y basis,
                an imaginary part
                of a metric tensor is returned.
        phase_fix: Flag that indicates whether a phase fix should be applied. It depends on the
                    type of a variational principle.
    Returns:
        Parametrized metric tensor as a ListOp.
    """
    operator = CircuitStateFn(ansatz)

    if qfi_method == "lin_comb_full":
        return LinCombFull().convert(operator, parameters, aux_meas_op=basis, phase_fix=phase_fix)

    return QFI(qfi_method).convert(operator, parameters)


def eval_metric_result(
    metric: OperatorBase,
    param_dict: Dict[Parameter, Union[float, complex]],
    metric_circ_sampler: CircuitSampler,
) -> OperatorBase:
    """
    Binds a parametrized metric tensor object to parameters values provided. Uses a circuit
    sampler if available.
    Args:
        metric: Metric tensor to be evaluated.
        param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        metric_circ_sampler: CircuitSampler for metric tensors.
    Returns:
        Metric tensor with all parameters bound.
    """
    if metric_circ_sampler:
        metric_result = metric_circ_sampler.convert(metric, params=param_dict).eval()
    else:
        metric_result = metric.assign_parameters(param_dict).eval()

    return metric_result
