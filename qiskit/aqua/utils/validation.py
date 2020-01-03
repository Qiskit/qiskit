# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
For now this module validates parameters based on its schema.
Once all json schema is eliminated from classes, this module will be removed
"""

from typing import Dict, Set
import numpy as np
import jsonschema
from qiskit.aqua import AquaError


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
        raise ValueError("{} must be one of '{}', was '{}'.".format(name, values, value))


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
        raise ValueError('{} must have value >= {}, was {}'.format(name, minimum, value))


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
        raise ValueError('{} must have value > {}, was {}'.format(name, minimum, value))


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
        raise ValueError('{} must have value <= {}, was {}'.format(name, maximum, value))


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
        raise ValueError('{} must have value < {}, was {}'.format(name, maximum, value))


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
        raise ValueError(
            '{} must have value >= {} and <= {}, was {}'.format(name, minimum, maximum, value))


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
        raise ValueError(
            '{} must have value > {} and < {}, was {}'.format(name, minimum, maximum, value))


def validate_range_exclusive_min(name: str, value: float,
                                 minimum: float, maximum: float) -> None:
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
        raise ValueError(
            '{} must have value > {} and <= {}, was {}'.format(name, minimum, maximum, value))


def validate_range_exclusive_max(name: str, value: float,
                                 minimum: float, maximum: float) -> None:
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
        raise ValueError(
            '{} must have value >= {} and < {}, was {}'.format(name, minimum, maximum, value))


def validate(args_dict: Dict[str, object], schema_dict: Dict[str, object]) -> None:
    """ validate json data according to a schema"""
    if schema_dict is None:
        return

    properties_dict = schema_dict.get('properties', None)
    if properties_dict is None:
        return

    json_dict = {}
    for property_name, _ in properties_dict.items():
        if property_name in args_dict:
            value = args_dict[property_name]
            if isinstance(value, np.ndarray):
                value = value.tolist()

            json_dict[property_name] = value
    try:
        jsonschema.validate(json_dict, schema_dict)
    except jsonschema.exceptions.ValidationError as vex:
        raise AquaError(vex.message)
