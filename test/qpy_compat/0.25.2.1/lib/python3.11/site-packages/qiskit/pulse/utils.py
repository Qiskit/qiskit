# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module for common pulse programming utilities."""
from typing import List, Dict, Union
import warnings

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import UnassignedDurationError, QiskitError
from qiskit.utils.deprecation import deprecate_func, deprecate_function


def format_meas_map(meas_map: List[List[int]]) -> Dict[int, List[int]]:
    """
    Return a mapping from qubit label to measurement group given the nested list meas_map returned
    by a backend configuration. (Qubits can not always be measured independently.) Sorts the
    measurement group for consistency.

    Args:
        meas_map: Groups of qubits that get measured together, for example: [[0, 1], [2, 3, 4]]
    Returns:
        Measure map in map format
    """
    qubit_mapping = {}
    for sublist in meas_map:
        sublist.sort()
        for q in sublist:
            qubit_mapping[q] = sublist
    return qubit_mapping


def format_parameter_value(
    operand: ParameterExpression,
    decimal: int = 10,
) -> Union[ParameterExpression, complex]:
    """Convert ParameterExpression into the most suitable data type.

    Args:
        operand: Operand value in arbitrary data type including ParameterExpression.
        decimal: Number of digit to round returned value.

    Returns:
        Value casted to non-parameter data type, when possible.
    """
    try:
        # value is assigned.
        # note that ParameterExpression directly supports __complex__ via sympy or symengine
        evaluated = complex(operand)
        # remove truncation error
        evaluated = np.round(evaluated, decimals=decimal)
        # typecast into most likely data type
        if np.isreal(evaluated):
            evaluated = float(evaluated.real)
            if evaluated.is_integer():
                evaluated = int(evaluated)
        else:
            warnings.warn(
                "Assignment of complex values to ParameterExpression in Qiskit Pulse objects is "
                "now pending deprecation. This will align the Pulse module with other modules "
                "where such assignment wasn't possible to begin with. The typical use case for complex "
                "parameters in the module was the SymbolicPulse library. As of Qiskit-Terra "
                "0.23.0 all library pulses were converted from complex amplitude representation"
                " to real representation using two floats (amp,angle), as used in the "
                "ScalableSymbolicPulse class. This eliminated the need for complex parameters. "
                "Any use of complex parameters (and particularly custom-built pulses) should be "
                "converted in a similar fashion to avoid the use of complex parameters.",
                PendingDeprecationWarning,
            )

        return evaluated
    except TypeError:
        # value is not assigned
        pass

    return operand


def instruction_duration_validation(duration: int):
    """Validate instruction duration.

    Args:
        duration: Instruction duration value to validate.

    Raises:
        UnassignedDurationError: When duration is unassigned.
        QiskitError: When invalid duration is assigned.
    """
    if isinstance(duration, ParameterExpression):
        raise UnassignedDurationError(
            "Instruction duration {} is not assigned. "
            "Please bind all durations to an integer value before playing in the Schedule, "
            "or use ScheduleBlock to align instructions with unassigned duration."
            "".format(repr(duration))
        )

    if not isinstance(duration, (int, np.integer)) or duration < 0:
        raise QiskitError(
            f"Instruction duration must be a non-negative integer, got {duration} instead."
        )


@deprecate_func(
    additional_msg="Instead, use 'qiskit.utils.deprecate_func'.",
    since="0.22.0",
)
def deprecated_functionality(func):
    """A decorator that raises deprecation warning without showing alternative method."""
    return deprecate_function(
        f"Calling {func.__name__} is being deprecated and will be removed soon. "
        "No alternative method will be provided with this change. "
        "If there is any practical usage of this functionality, please write "
        "an issue in Qiskit/qiskit-terra repository.",
        category=DeprecationWarning,
        stacklevel=2,
        since="0.22.0",
    )(func)
