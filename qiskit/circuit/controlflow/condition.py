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

"""Functions for dealing with classical conditions."""

from typing import Tuple, Union

from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.exceptions import CircuitError


def validate_condition(
    condition: Tuple[Union[ClassicalRegister, Clbit], int]
) -> Tuple[Union[ClassicalRegister, Clbit], int]:
    """Validate that a condition is in a valid format and return it, but raise if it is invalid.

    Args:
        condition: the condition to be tested for validity.

    Raises:
        CircuitError: if the condition is not in a valid format.

    Returns:
        The same condition as passed, if it was valid.
    """
    try:
        bits, value = condition
        if isinstance(bits, (ClassicalRegister, Clbit)) and isinstance(value, int):
            return (bits, value)
    except (TypeError, ValueError):
        pass
    raise CircuitError(
        "A classical condition should be a 2-tuple of `(ClassicalRegister | Clbit, int)`,"
        f" but received '{condition!r}'."
    )


def condition_bits(condition: Tuple[Union[ClassicalRegister, Clbit], int]) -> Tuple[Clbit, ...]:
    """Return the classical resources used by ``condition`` as a tuple of :obj:`.Clbit`.

    This is useful when the exact set of bits is required, rather than the logical grouping of
    :obj:`.ClassicalRegister`, such as when determining circuit blocking.

    Args:
        condition: the valid condition to extract the bits from.

    Returns:
        a tuple of all classical bits used in the condition.
    """
    return (condition[0],) if isinstance(condition[0], Clbit) else tuple(condition[0])


def condition_registers(
    condition: Tuple[Union[ClassicalRegister, Clbit], int]
) -> Tuple[ClassicalRegister, ...]:
    """Return any classical registers used by ``condition`` as a tuple of :obj:`.ClassicalRegister`.

    This is useful as a quick method for extracting the registers from a condition, if any exist.
    The output might be empty if the condition is on a single bit.

    Args:
        condition: the valid condition to extract any registers from.

    Returns:
        a tuple of all classical registers used in the condition.
    """
    return (condition[0],) if isinstance(condition[0], ClassicalRegister) else ()
