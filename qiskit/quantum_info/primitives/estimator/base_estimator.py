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
"""
Expectation value base class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union, cast

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.primitives.framework.base_primitive import (
    BasePrimitive,
    PreprocessedCircuits,
)
from qiskit.result import Result
from qiskit.transpiler import PassManager

from ..results import EstimatorResult
from ..results.base_result import BaseResult
from ..framework.utils import Group, init_circuit, init_observable


class BaseEstimator(BasePrimitive, ABC):
    """
    Expectation Value class
    """

    def __init__(
        self,
        circuits: Union[QuantumCircuit, list[Union[QuantumCircuit]]],
        observables: Union[BaseOperator, PauliSumOp, list[Union[BaseOperator, PauliSumOp]]],
        backend: Backend,
        transpile_options: Optional[dict] = None,
        bound_pass_manager: Optional[PassManager] = None,
    ):
        """ """
        super().__init__(
            backend=backend,
            transpile_options=transpile_options,
            bound_pass_manager=bound_pass_manager,
        )

        # Initialize quantum circuits
        if isinstance(circuits, list):
            self._circuits = [init_circuit(circuit) for circuit in circuits]
        else:
            self._circuits = [init_circuit(circuits)]

        # Initialize observables
        if isinstance(observables, list):
            self._observables = [init_observable(observable) for observable in observables]
        else:
            self._observables = [init_observable(observables)]

        self._grouping = [
            Group(i, j) for i in range(len(self._circuits)) for j in range(len(self._observables))
        ]
        self._transpiled_circuits_cache: dict[tuple[Group], list[QuantumCircuit]] = {}

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
    def grouping(self) -> list[Group]:
        """
        Group of circuits and observables.

        Returns:
            grouping
        """
        return self._grouping

    def set_transpile_options(self, **fields) -> BaseEstimator:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        self._transpiled_circuits_cache = {}
        super().set_transpile_options(**fields)
        return self

    @property
    def preprocessed_circuits(
        self,
    ) -> PreprocessedCircuits:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        self._preprocessed_circuits = self._preprocessing(self.circuits, self.observables)
        return super().preprocessed_circuits

    # pylint: disable=arguments-differ
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        grouping: Optional[list[Union[Group, tuple[int, int]]]] = None,
        **run_options,
    ) -> EstimatorResult:
        if grouping is not None:
            self._grouping = [g if isinstance(g, Group) else Group(g[0], g[1]) for g in grouping]

        return cast(EstimatorResult, super().run(parameters, **run_options))

    @abstractmethod
    def _preprocessing(
        self, circuits: list[QuantumCircuit], observables: list[SparsePauliOp]
    ) -> Union[list[QuantumCircuit], list[tuple[QuantumCircuit, list[QuantumCircuit]]]]:
        return NotImplemented

    @abstractmethod
    def _postprocessing(self, result: Union[dict, BaseResult, Result]) -> EstimatorResult:
        return NotImplemented

    def _transpile(self):

        if tuple(self._grouping) in self._transpiled_circuits_cache:
            self._transpiled_circuits = self._transpiled_circuits_cache[tuple(self._grouping)]
        else:
            super()._transpile()
            self._transpiled_circuits_cache[tuple(self._grouping)] = self._transpiled_circuits
