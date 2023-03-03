# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Primitive abstract base class."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.providers import Options


class BasePrimitive(ABC):
    """Primitive abstract base class."""

    def __init__(self, options: dict | None = None):
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

    ################################################################################
    ## PROPERTIES
    ################################################################################
    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._run_options

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._run_options.update_options(**fields)

    ################################################################################
    ## VALIDATION
    ################################################################################
    @staticmethod
    def _validate_circuits(
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        elif not isinstance(circuits, Sequence) or not all(
            isinstance(cir, QuantumCircuit) for cir in circuits
        ):
            raise TypeError("Invalid circuits, expected Sequence[QuantumCircuit].")
        elif not isinstance(circuits, tuple):
            circuits = tuple(circuits)
        if len(circuits) == 0:
            raise ValueError("No circuits were provided.")
        return circuits

    @staticmethod
    def _validate_parameter_values(
        parameter_values: Sequence[Sequence[float] | Mapping[ParameterExpression, float]]
        | Sequence[float]
        | Mapping[ParameterExpression, float]
        | float
        | None,
        parameter_views: Sequence[ParameterView],
    ) -> tuple[tuple[float, ...], ...]:
        # Allow optional
        if parameter_values is None:
            parameter_values = [()] * len(parameter_views)

        # Allow single value
        if _isreal(parameter_values):
            parameter_values = [[parameter_values]]
        elif isinstance(parameter_values, (Sequence, np.ndarray)) and not any(
            isinstance(vector, (Sequence, Mapping, np.ndarray)) for vector in parameter_values
        ):
            parameter_values = [parameter_values]

        # Support numpy ndarray
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()
        elif isinstance(parameter_values, Sequence):
            parameter_values = [
                vector.tolist() if isinstance(vector, np.ndarray) else vector
                for vector in parameter_values
            ]

        # Support Mapping
        if isinstance(parameter_values, Mapping):
            parameter_values = [[parameter_values[parameter] for parameter in parameter_views[0]]]
        elif isinstance(parameter_values, Sequence):
            parameter_values = [
                [_validate_and_getitem(value, parameter, i) for parameter in parameter_view]
                if isinstance(value, Mapping)
                else value
                for i, (value, parameter_view) in enumerate(zip(parameter_values, parameter_views))
            ]

        # Validation
        if (
            not isinstance(parameter_values, Sequence)
            or not all(isinstance(vector, Sequence) for vector in parameter_values)
            or not all(all(_isreal(value) for value in vector) for vector in parameter_values)
        ):
            raise TypeError("Invalid parameter values, expected Sequence[Sequence[float]].")

        return tuple(tuple(float(value) for value in vector) for vector in parameter_values)

    @staticmethod
    def _cross_validate_circuits_parameter_values(
        circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]
    ) -> None:
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )
        for i, (circuit, vector) in enumerate(zip(circuits, parameter_values)):
            if len(vector) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(vector)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )


def _isint(obj: Any) -> bool:
    """Check if object is int."""
    int_types = (int, np.integer)
    return isinstance(obj, int_types) and not isinstance(obj, bool)


def _isreal(obj: Any) -> bool:
    """Check if object is a real number: int or float except ``±Inf`` and ``NaN``."""
    float_types = (float, np.floating)
    return _isint(obj) or isinstance(obj, float_types) and float("-Inf") < obj < float("Inf")


def _validate_and_getitem(parameter_dict, parameter, i):
    if parameter not in parameter_dict:
        raise ValueError(
            f"Parameter {parameter} in {i}-th circuit can not be found in {parameter_dict}."
        )
    value = parameter_dict[parameter]
    if not _isreal(value):
        raise ValueError(
            f"The value of dict must be a real number, but {value} was given in {i}-th parameter_value."
        )
    return value
