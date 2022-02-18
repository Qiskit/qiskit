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
Evaluator class base class
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union, cast

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info.primitives.results.base_result import BaseResult
from qiskit.result import Result, BaseReadoutMitigator
from qiskit.transpiler import PassManager

PreprocessedCircuits = Union[
    "list[QuantumCircuit]", "list[tuple[QuantumCircuit, list[QuantumCircuit]]]"
]


class BasePrimitive(ABC):
    """
    Base class for primitives.
    """

    def __init__(
        self,
        backend: Backend,
        mitigator: Optional[BaseReadoutMitigator] = None,
        transpile_options: Optional[dict] = None,
        bound_pass_manager: Optional[PassManager] = None,
    ):
        """
        Args:
            backend: backend
        """
        self._backend: Union[Backend, BaseReadoutMitigator] = backend
        self._mitigator: Optional[BaseReadoutMitigator] = mitigator
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: Optional[PreprocessedCircuits] = None
        self._transpiled_circuits: Optional[list[QuantumCircuit]] = None

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self._is_closed = True

    def __call__(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        **run_options,
    ) -> BaseResult:
        return self.run(parameters, **run_options)

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> BasePrimitive:
        """Set options values for the evaluator.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._run_options.update_options(**fields)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> BasePrimitive:
        """Set the transpiler options for transpiler.
        Args:
            fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()

        self._transpile_options.update_options(**fields)
        return self

    @property
    def backend(self) -> Backend:
        """Backend

        Returns:
            backend
        """
        return self._backend.backend

    @property
    def preprocessed_circuits(self) -> Optional[PreprocessedCircuits]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
        self._transpile()
        return self._transpiled_circuits

    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        **run_options,
    ) -> BaseResult:
        """
        Returns:
            The running result.
        Raises:
            QiskitError: if the instance has been closed.
            TypeError: if the shape of parameters is invalid.
        """
        self._check_is_closed()

        transpiled_circuits = self.transpiled_circuits

        # preprocessing and transpile
        # parameters: NoneType
        if parameters is None:
            parameters = cast("list[list[float]]", [[]] * len(self.preprocessed_circuits))
        # parameters: list[float]
        elif isinstance(parameters[0], (int, float)):
            parameters = cast("list[list[float]]", [parameters] * len(self.preprocessed_circuits))
        # parameters: list[list[float]]
        elif len(self.preprocessed_circuits) == 1:
            transpiled_circuits = transpiled_circuits * len(parameters)
        elif len(parameters) != len(self.preprocessed_circuits):
            raise TypeError("The number of parameters and circuits must be same.")

        # Bind parameters
        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        bound_circuits = [
            circuit.bind_parameters(parameter)  # type: ignore
            for parameter, circuit in zip(parameters, transpiled_circuits)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        result = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        return self._postprocessing(result)

    def _transpile(self):
        self._transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(
                self.preprocessed_circuits,
                self.backend,
                **self.transpile_options.__dict__,
            ),
        )

    @abstractmethod
    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> BaseResult:
        return NotImplemented

    def _check_is_closed(self):
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return cast("list[QuantumCircuit]", self._bound_pass_manager.run(circuits))
