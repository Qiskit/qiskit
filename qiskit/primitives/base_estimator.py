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
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp

from .results import EstimatorResult


class BaseEstimator(ABC):
    """
    Estimator base class.
    """

    @abstractmethod
    def __init__(
        self,
        circuits: list[QuantumCircuit],
        observables: list[SparsePauliOp],
        backend: Backend,
    ):
        ...

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, ex_type, ex_value, trace):
        ...

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.__call__ = cls.run

    @abstractmethod
    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Quantum Circuits that represents quantum states.

        Returns:
            quantum states
        """
        ...

    @abstractmethod
    @property
    def observables(self) -> list[SparsePauliOp]:
        """
        SparsePauliOp that represents observable

        Returns:
            observable
        """
        ...

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        ...

    @abstractmethod
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        **run_options,
    ) -> EstimatorResult:
        """
        Run the estimation.

        Args:
            parameters: parameters to be bound.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The result of Estimator.
        """
        ...
