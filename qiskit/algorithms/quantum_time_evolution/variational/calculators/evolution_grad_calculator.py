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
from typing import Union, Optional, List

from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.opflow import StateFn, Gradient, CircuitGradient, OperatorBase, Z
from qiskit.opflow.gradients.circuit_gradients import LinComb


def calculate(
    observable: OperatorBase,
    ansatz: OperatorBase,
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    grad_method: Union[str, CircuitGradient],
    basis: OperatorBase = Z,
):
    operator = ~StateFn(observable) @ StateFn(ansatz)
    if grad_method == "lin_comb":
        return LinComb().convert(operator, parameters, aux_meas_op=basis)
    return Gradient(grad_method).convert(operator, parameters)
