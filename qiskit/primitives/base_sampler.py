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
Sampler class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend

from .sampler_result import SamplerResult


class BaseSampler(ABC):
    """
    Sampler base class
    """

    def __init__(
        self,
        circuits: list[QuantumCircuit],
        backend: Backend,
    ):
        """
        Args:
            circuits: circuits to be executed
            backend: a backend or a backend wrapper
        """
        self._circuits = circuits
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
        """A list of quantum circuits to be executed

        Returns:
            a list of quantum circuits
        """
        return self._circuits

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
        **run_options,
    ) -> SamplerResult:
        """
        Run the sampling.

        Args:
            parameters: parameters to be bound.
            run_options: backend runtime options used for circuit execution.


        Returns:
            The result of Sampler.
        """
        ...
