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

"""Set of method for validating input to TrotterQrte algorithm."""

import numbers
from typing import Union

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.opflow import (
    OperatorBase,
    StateFn,
    SummedOp,
    PauliOp,
)


def _is_op_bound(operator: Union[SummedOp, PauliOp, OperatorBase]) -> None:
    """Checks if an operator provided has all parameters bound.
    Args:
        operator: Operator to be checked.
    Raises:
        ValueError: If an operator has unbound parameters.
    """
    if len(operator.parameters) > 0:
        raise ValueError(
            f"Did not manage to bind all parameters in the Hamiltonian, "
            f"these parameters encountered: {operator.parameters}."
        )


def _validate_input(initial_state: StateFn, observable: OperatorBase) -> None:
    """Validates if one and only one among initial_state and observable is provided."""
    if initial_state is None and observable is None:
        raise ValueError(
            "TrotterQrte requires an initial state or an observable to be evolved; None "
            "provided."
        )
    if initial_state is not None and observable is not None:
        raise ValueError(
            "TrotterQrte requires an initial state or an observable to be evolved; both "
            "provided."
        )


def _validate_hamiltonian_form(hamiltonian: Union[SummedOp, PauliOp, OperatorBase]):
    """Validates that a Hamiltonian is of a correct type and with expected dependence on
    parameters.
    Args:
        hamiltonian: Hamiltonian to be validated.
    Raises:
        ValueError: if an invalid Hamiltonian is provided.
    """
    if isinstance(hamiltonian, SummedOp):
        if isinstance(hamiltonian.coeff, ParameterExpression):
            raise ValueError(
                "The coefficient multiplying the whole Hamiltonian cannot be a "
                "ParameterExpression."
            )
        for op in hamiltonian.oplist:
            if not _is_linear_with_single_param(op):
                raise ValueError(
                    "Hamiltonian term has a coefficient that is not a linear function of a "
                    "single parameter. It is not supported."
                )
    elif isinstance(hamiltonian, (PauliOp, OperatorBase)):
        if not _is_linear_with_single_param(hamiltonian):
            raise ValueError(
                "Hamiltonian term has a coefficient that is not a linear function of a "
                "single parameter. It is not supported."
            )
    else:
        raise ValueError("Hamiltonian not a SummedOp which is the only option supported.")


def _is_linear_with_single_param(operator: OperatorBase) -> bool:
    """Checks if an operator provided is linear w.r.t. one and only one parameter.
    Args:
        operator: Operator to be checked.
    Returns:
        True or False depending on whether an operator is linear in a single param and only contains'
        a single param.
    Raises:
        ValueError: If an operator contains more than 1 parameter.
    """
    if (
        not isinstance(operator.coeff, ParameterExpression)
        and not isinstance(operator.coeff, Parameter)
        or len(operator.coeff.parameters) == 0
    ):
        return True
    if len(operator.coeff.parameters) > 1:
        raise ValueError(
            "Term of a Hamiltonian has a coefficient that depends on several "
            "parameters. Only dependence on a single parameter is allowed."
        )
    single_parameter_expression = operator.coeff
    parameter = list(single_parameter_expression.parameters)[0]
    gradient = single_parameter_expression.gradient(parameter)
    return isinstance(gradient, numbers.Number)
