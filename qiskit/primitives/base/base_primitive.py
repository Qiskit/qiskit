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
from collections.abc import Sequence
from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import BasePrimitiveOptions, BasePrimitiveOptionsLike
from qiskit.providers import Options
from qiskit.utils.deprecation import deprecate_func

from . import validation


class BasePrimitiveV1(ABC):
    """Primitive abstract base class."""

    def __init__(self, options: dict | None = None):
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

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

    @staticmethod
    @deprecate_func(since="0.46.0")
    def _validate_circuits(
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        return validation._validate_circuits(circuits)

    @staticmethod
    @deprecate_func(since="0.46.0")
    def _validate_parameter_values(
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None,
        default: Sequence[Sequence[float]] | Sequence[float] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        return validation._validate_parameter_values(parameter_values, default=default)

    @staticmethod
    @deprecate_func(since="0.46.0")
    def _cross_validate_circuits_parameter_values(
        circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]
    ) -> None:
        return validation._cross_validate_circuits_parameter_values(
            circuits, parameter_values=parameter_values
        )


BasePrimitive = BasePrimitiveV1


class BasePrimitiveV2(ABC):
    """Primitive abstract base class version 2."""

    version = 2
    _options_class: type[BasePrimitiveOptions] = BasePrimitiveOptions

    def __init__(self, options: BasePrimitiveOptionsLike | None = None):
        self._options = self._options_class()
        if options:
            self._options.update(options)

    @property
    def options(self) -> BasePrimitiveOptions:
        """Options for the primitive"""
        return self._options
