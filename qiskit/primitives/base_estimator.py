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
"""
Estimator base class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp

from .estimator_result import EstimatorResult


@dataclass(frozen=True)
class Group:
    """The dataclass represents indices of circuit and observable."""

    circuit_index: int
    observable_index: int


class BaseEstimator(ABC):
    """
    Estimator base class.
    """

    def __init__(
        self,
        circuits: list[QuantumCircuit],
        observables: list[SparsePauliOp],
        backend: Backend,
    ):
        self._circuits = circuits
        self._observables = observables
        self._backend = backend

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, ex_type, ex_value, trace):
        ...

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.__call__ = cls.run

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Quantum Circuits that represents quantum states.

        Returns:
            quantum states
        """
        return self._circuits

    @property
    def observables(self) -> list[SparsePauliOp]:
        """
        SparsePauliOp that represents observable

        Returns:
            observable
        """
        return self._observables

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @abstractmethod
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        grouping: Optional[list[Union[Group, tuple[int, int]]]] = None,
        **run_options,
    ) -> EstimatorResult:
        """
        Run the estimation.

        Args:
            parameters: parameters to be bound.
            run_options: backend runtime options used for circuit execution.
            grouping: the list of Group or tuple of circuit index and observable index.

        Returns:
            The result of Estimator.
        """
        ...
