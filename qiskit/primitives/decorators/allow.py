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

"""Allow decorators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import wraps
import sys
from typing import overload

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base_estimator import BaseEstimator
from ..base_sampler import BaseSampler
from ..estimator_result import EstimatorResult
from ..sampler_result import SamplerResult

if sys.version_info >= (3, 8):
    # pylint: disable=no-name-in-module, ungrouped-imports
    from typing import Protocol
else:
    from typing_extensions import Protocol


class PrimitiveDecorator(Protocol):
    """Callback Protocol for decorator in primitives.

    .. automethod:: qiskit.primitives.decorators.allow.PrimitiveDecorator.__call__
    """

    @overload
    def __call__(self, cls: type[BaseEstimator]) -> type[BaseEstimator]:
        ...

    @overload
    def __call__(self, cls: type[BaseSampler]) -> type[BaseSampler]:
        ...

    def __call__(self, cls):
        """
        Function call of decorator.

        Args:
            cls: class to be decorated.

        Returns:
            The decorated class or decorator.
        """
        ...


@overload
def allow_optional(cls: type[BaseEstimator]) -> type[BaseEstimator]:
    ...


@overload
def allow_optional(cls: type[BaseSampler]) -> type[BaseSampler]:
    ...


@overload
def allow_optional(cls: None) -> PrimitiveDecorator:
    ...


def allow_optional(cls=None) -> type[BaseEstimator] | type[BaseSampler] | PrimitiveDecorator:
    """
    Allow optional. If indices are not given, indices are the list of all circuits (and observables)
    i.e. ``[0, 1, ..., len(circuits) - 1]`` (and ``[0, 1, ..., len(observables) - 1]``).
    If parameter values are not given, the parameter values are circuit indices length empty lists.

    Args:
        cls: class to be decorated.

    Returns:
        The decorated class or decorator.
    """

    if cls is None:
        return _allow_optional

    return _allow_optional(cls)


def _allow_optional(cls):
    original_call_method = cls.__call__

    if issubclass(cls, BaseEstimator):

        @wraps(cls.__call__)
        def wrapper(
            self: BaseEstimator,
            circuit_indices: Sequence[int] | None = None,
            observable_indices: Sequence[int] | None = None,
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> EstimatorResult:
            if circuit_indices is None:
                circuit_indices = list(range(len(self.circuits)))
            if observable_indices is None:
                observable_indices = list(range(len(self.observables)))
            if parameter_values is None:
                parameter_values = [[]] * len(circuit_indices)
            return original_call_method(
                self, circuit_indices, observable_indices, parameter_values, **run_options
            )

    elif issubclass(cls, BaseSampler):

        @wraps(cls.__call__)
        def wrapper(
            self: BaseSampler,
            circuit_indices: Sequence[int] | None = None,
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> SamplerResult:
            if circuit_indices is None:
                circuit_indices = list(range(len(self._circuits)))
            if parameter_values is None:
                parameter_values = [[]] * len(circuit_indices)
            return original_call_method(self, circuit_indices, parameter_values, **run_options)

    else:
        raise TypeError(
            "allow_optional decorator can be used for subclass of BaseEstimator or BaseSampler."
        )

    setattr(cls, "__call__", wrapper)
    return cls


@overload
def allow_broadcasting(cls: type[BaseEstimator]) -> type[BaseEstimator]:
    ...


@overload
def allow_broadcasting(cls: type[BaseSampler]) -> type[BaseSampler]:
    ...


@overload
def allow_broadcasting(cls: None) -> PrimitiveDecorator:
    ...


def allow_broadcasting(cls=None) -> type[BaseEstimator] | type[BaseSampler] | PrimitiveDecorator:
    """
    Allow broadcasting. Broadcasting means that if the number of circuits (and observables) is one
    and indices are not given, it generates as many indices as the number of parameters, i.e.
    ``[0] * len(parameter_values)``.

    Args:
        cls: class to be decorated.

    Returns:
        The decorated class or decorator.
    """

    if cls is None:
        return _allow_broadcasting

    return _allow_broadcasting(cls)


def _allow_broadcasting(cls):
    original_call_method = cls.__call__

    if issubclass(cls, BaseEstimator):

        @wraps(cls.__call__)
        def wrapper(
            self: BaseEstimator,
            circuit_indices: Sequence[int] | None = None,
            observable_indices: Sequence[int] | None = None,
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> EstimatorResult:
            if (
                circuit_indices is None
                and len(self._circuits) == 1
                and observable_indices is None
                and len(self._observables) == 1
                and parameter_values is not None
            ):
                circuit_indices = [0] * len(parameter_values)
                observable_indices = [0] * len(parameter_values)
            return original_call_method(
                self, circuit_indices, observable_indices, parameter_values, **run_options
            )

    elif issubclass(cls, BaseSampler):

        @wraps(cls.__call__)
        def wrapper(
            self: BaseSampler,
            circuit_indices: Sequence[int] | None = None,
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> SamplerResult:
            if (
                circuit_indices is None
                and parameter_values is not None
                and len(self._circuits) == 1
            ):
                circuit_indices = [0] * len(parameter_values)

            return original_call_method(self, circuit_indices, parameter_values, **run_options)

    else:
        raise TypeError(
            "allow_broadcasting decorator can be used for subclass of BaseEstimator or BaseSampler."
        )

    setattr(cls, "__call__", wrapper)
    return cls


@overload
def allow_objects(cls: type[BaseEstimator]) -> type[BaseEstimator]:
    ...


@overload
def allow_objects(cls: type[BaseSampler]) -> type[BaseSampler]:
    ...


@overload
def allow_objects(cls: None) -> PrimitiveDecorator:
    ...


def allow_objects(cls=None) -> type[BaseEstimator] | type[BaseSampler] | PrimitiveDecorator:
    """
    Allow objects as inputs of indices instead of integer.

    Args:
        cls: class to be decorated.

    Returns:
        The decorated class or decorator.
    """

    if cls is None:
        return _allow_objects

    return _allow_objects(cls)


def _allow_objects(cls):
    original_call_method = cls.__call__
    original_init_method = cls.__init__

    if issubclass(cls, BaseEstimator):

        @wraps(cls.__init__)
        def init_wrapper(
            self: BaseEstimator,
            circuits: Iterable[QuantumCircuit],
            observables: Iterable[SparsePauliOp],
            *args,
            parameters: Iterable[Iterable[Parameter]] | None = None,
            **kwargs,
        ):
            if isinstance(circuits, QuantumCircuit):
                circuits = [circuits]
            else:
                circuits = list(circuits)
            if isinstance(observables, (PauliSumOp, BaseOperator)):
                observables = [observables]
            else:
                observables = list(observables)
            self.__circuit_ids = [id(i) for i in circuits]
            self.__observable_ids = [id(i) for i in observables]
            original_init_method(self, circuits, observables, parameters, *args, **kwargs)

        @wraps(cls.__call__)
        def wrapper(
            self: BaseEstimator,
            circuit_indices: Sequence[int | QuantumCircuit],
            observable_indices: Sequence[int | SparsePauliOp],
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> EstimatorResult:
            try:
                circuit_indices = [
                    next(
                        map(
                            lambda x: x[0],
                            filter(
                                lambda x: x[1] == id(i),  # pylint: disable=cell-var-from-loop
                                enumerate(self.__circuit_ids),
                            ),
                        )
                    )
                    if not isinstance(i, (int, np.integer))
                    else i
                    for i in circuit_indices
                ]
            except StopIteration as err:
                raise QiskitError("The object id does not match.") from err
            try:
                observable_indices = [
                    next(
                        map(
                            lambda x: x[0],
                            filter(
                                lambda x: x[1] == id(i),  # pylint: disable=cell-var-from-loop
                                enumerate(self.__observable_ids),
                            ),
                        )
                    )
                    if not isinstance(i, (int, np.integer))
                    else i
                    for i in observable_indices
                ]
            except StopIteration as err:
                raise QiskitError("The object id does not match.") from err
            return original_call_method(
                self, circuit_indices, observable_indices, parameter_values, **run_options
            )

    elif issubclass(cls, BaseSampler):

        @wraps(cls.__init__)
        def init_wrapper(
            self: BaseEstimator,
            circuits: Iterable[QuantumCircuit],
            *args,
            parameters: Iterable[Iterable[Parameter]] | None = None,
            **kwargs,
        ):
            self._circuit_ids = [id(i) for i in circuits]
            original_init_method(self, circuits, parameters, *args, **kwargs)

        @wraps(cls.__call__)
        def wrapper(
            self: BaseSampler,
            circuit_indices: Sequence[int | QuantumCircuit],
            parameter_values: Sequence[Sequence[float]] | None = None,
            **run_options,
        ) -> SamplerResult:
            try:
                circuit_indices = [
                    next(
                        map(
                            lambda x: x[0],
                            filter(
                                lambda x: x[1] == id(i),  # pylint: disable=cell-var-from-loop
                                enumerate(self._circuit_ids),
                            ),
                        )
                    )
                    if not isinstance(i, (int, np.integer))
                    else i
                    for i in circuit_indices
                ]
            except StopIteration as err:
                raise QiskitError("The object id does not match.") from err

            return original_call_method(self, circuit_indices, parameter_values, **run_options)

    else:
        raise TypeError(
            "allow_objects decorator can be used for subclass of BaseEstimator or BaseSampler."
        )

    setattr(cls, "__call__", wrapper)
    setattr(cls, "__init__", init_wrapper)
    return cls
