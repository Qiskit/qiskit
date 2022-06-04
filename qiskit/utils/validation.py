# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Validation module
"""

from typing import Set


def validate_in_set(name: str, value: object, values: Set[object]) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        values: set that should contain value.
    Raises:
        ValueError: invalid value
    """
    if value not in values:
        raise ValueError(f"{name} must be one of '{values}', was '{value}'.")


def validate_min(name: str, value: float, minimum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum:
        raise ValueError(f"{name} must have value >= {minimum}, was {value}")


def validate_min_exclusive(name: str, value: float, minimum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum:
        raise ValueError(f"{name} must have value > {minimum}, was {value}")


def validate_max(name: str, value: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value > maximum:
        raise ValueError(f"{name} must have value <= {maximum}, was {value}")


def validate_max_exclusive(name: str, value: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value >= maximum:
        raise ValueError(f"{name} must have value < {maximum}, was {value}")


def validate_range(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must have value >= {minimum} and <= {maximum}, was {value}")


def validate_range_exclusive(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum or value >= maximum:
        raise ValueError(f"{name} must have value > {minimum} and < {maximum}, was {value}")


def validate_range_exclusive_min(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum or value > maximum:
        raise ValueError(f"{name} must have value > {minimum} and <= {maximum}, was {value}")


def validate_range_exclusive_max(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum or value >= maximum:
        raise ValueError(f"{name} must have value >= {minimum} and < {maximum}, was {value}")
