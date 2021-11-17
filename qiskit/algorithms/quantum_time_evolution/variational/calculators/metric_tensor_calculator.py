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
from qiskit.opflow import QFI, CircuitQFI, CircuitStateFn, OperatorBase, Z, CircuitSampler
from qiskit.opflow.gradients.circuit_qfis import LinCombFull
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


def calculate(
    ansatz: QuantumCircuit,
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    basis: OperatorBase = Z,
    phase_fix: bool = True,
):
    operator = CircuitStateFn(ansatz)

    if qfi_method == "lin_comb_full":
        return LinCombFull().convert(operator, parameters, aux_meas_op=basis, phase_fix=phase_fix)

    return QFI(qfi_method).convert(operator, parameters)


def eval_metric_tensor(
    metric_tensor,
    param_dict: Dict[Parameter, Union[float, complex]],
    metric_circ_sampler: CircuitSampler,
):
    if metric_circ_sampler:
        metric_res = np.array(metric_circ_sampler.convert(metric_tensor, params=param_dict).eval())
    else:
        metric_res = np.array(metric_tensor.assign_parameters(param_dict).eval())
    return metric_res
