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
import functools
import warnings
from typing import List, Dict, Union

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import UnassignedDurationError, QiskitError


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


@functools.lru_cache(maxsize=None)
def format_parameter_value(
    operand: Union[ParameterExpression],
) -> Union[ParameterExpression, complex]:
    """Convert ParameterExpression into the most suitable data type.

    Args:
        operand: Operand value in arbitrary data type including ParameterExpression.

    Returns:
        Value casted to non-parameter data type, when possible.
    """
    # to evaluate parameter expression object, sympy srepr function is used.
    # this function converts the parameter object into string with tiny round error.
    # therefore evaluated value is not completely equal to the assigned value.
    # however this error can be ignored in practice though we need to be careful for unittests.
    # i.e. "pi=3.141592653589793" will be evaluated as "3.14159265358979"
    # no DAC that recognizes the resolution of 1e-15 but they are AlmostEqual in tests.
    from sympy import srepr

    math_expr = srepr(operand).replace("*I", "j")
    try:
        # value is assigned
        evaluated = complex(math_expr)
        if not np.iscomplex(evaluated):
            evaluated = float(evaluated.real)
            if evaluated.is_integer():
                evaluated = int(evaluated)
        return evaluated
    except ValueError:
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
            "Instruction duration must be a non-negative integer, "
            "got {} instead.".format(duration)
        )


def deprecated_functionality(func):
    """A decorator that raises deprecation warning without showing alternative method."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Calling {func.__name__} is being deprecated and will be removed soon. "
            "No alternative method will be provided with this change. "
            "If there is any practical usage of this functionality, please write "
            "an issue in Qiskit/qiskit-terra repository.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
