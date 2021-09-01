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

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary\
    .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real\
    .real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.opflow import StateFn, NaturalGradient, OpflowError


def calculate(
    variational_principle: VariationalPrinciple,
    param_dict,
    regularization=None,
):
    # TODO consider passing operator immediately, also in evolution_grad and metric_tensor
    observable = variational_principle._hamiltonian
    ansatz = variational_principle._ansatz
    operator = ~StateFn(observable) @ StateFn(ansatz)
    parameters = variational_principle._parameters
    grad_method = variational_principle._grad_method
    qfi_method = variational_principle._qfi_method

    # VarQRTE
    if isinstance(variational_principle, RealVariationalPrinciple):
        nat_grad = _calc_op_natural_gradient(
            grad_method, operator, parameters, qfi_method, regularization
        )
    # VarQITE
    elif isinstance(variational_principle, ImaginaryVariationalPrinciple):
        nat_grad = _calc_op_natural_gradient(
            grad_method, -operator, parameters, qfi_method, regularization
        )
    else:
        raise OpflowError(
            f"Unrecognized variational principle provided, of type {type(variational_principle)}."
        )

    nat_grad_result = nat_grad.assign_parameters(param_dict).eval()

    imaginary_part_threshold = 1e-8
    if any(
        np.abs(np.imag(nat_grad_item)) > imaginary_part_threshold
        for nat_grad_item in nat_grad_result
    ):
        raise Warning("The imaginary part of the gradient are non-negligible.")

    print("nat grad result", nat_grad_result)
    return NaturalGradient(grad_method).convert(operator, parameters)


def _calc_op_natural_gradient(grad_method, operator, parameters, qfi_method, regularization):
    nat_grad = NaturalGradient(
        grad_method=grad_method,
        qfi_method=qfi_method,
        regularization=regularization,
    ).convert(operator * 0.5, parameters)
    return nat_grad
