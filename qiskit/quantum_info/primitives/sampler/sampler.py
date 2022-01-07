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

from collections import Counter
from typing import Optional, Union, cast

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler import PassManager

from ..backends import BackendWrapper
from ..framework.base_primitive import BasePrimitive
from ..results import SamplerResult
from ..results.base_result import BaseResult
from ..utils import init_circuit


class Sampler(BasePrimitive):
    """
    Sampler class
    """

    def __init__(
        self,
        backend: Union[Backend, BackendWrapper],
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
        transpile_options: Optional[dict] = None,
        bound_pass_manager: Optional[PassManager] = None,
    ):
        """
        Args:
            backend: a backend or a backend wrapper
            circuits: circuits to be executed
        """
        super().__init__(
            backend=backend,
            transpile_options=transpile_options,
            bound_pass_manager=bound_pass_manager,
        )
        if circuits is None:
            self._circuits = None
        elif isinstance(circuits, list):
            self._circuits = [init_circuit(circuit) for circuit in circuits]
        else:
            self._circuits = [init_circuit(circuits)]

        self._skip_transpilation = False

    @classmethod
    def from_backend(cls, backend: Union[Backend, BackendWrapper, Sampler]) -> "Sampler":
        """Generates a Sampler based on a backend

        Args:
            backend: a backend, a backend wrapper, or a sampler

        Returns:
            a sampler
        """
        if not isinstance(backend, Sampler):
            return cls(backend=backend)
        return backend

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """A list of quantum circuits to be executed

        Returns:
            a list of quatum circuits
        """
        return self._circuits

    @property
    def preprocessed_circuits(self) -> Optional[list[QuantumCircuit]]:
        return self._circuits

    # pylint: disable=arguments-differ
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        circuits: Optional[list[QuantumCircuit]] = None,
        **run_options,
    ) -> SamplerResult:
        if circuits is not None:
            self._circuits = circuits
        return cast(SamplerResult, super().run(parameters, **run_options))

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend.backend

    def set_skip_transpilation(self):
        """Once the method is called, the transpilation will be skiped."""
        self._skip_transpilation = True

    def _get_quasis(
        self, results: list[Result], num_circuits: int = -1
    ) -> tuple[list[QuasiDistribution], float]:
        """Converts a list of results to quasi-probabilities and the total number of shots

        Args:
            results: a list of results
            num_circuits: the number of circuits. If -1, `len(self._circuits)` will be used.

        Returns:
            a list of quasi-probabilities and the total number of shots

        Raises:
            QiskitError: if inputs are empty
        """
        if len(results) == 0:
            raise QiskitError("Empty result")
        list_counts = self._backend.get_counts(results)
        if num_circuits == -1:
            num_circuits = len(self._circuits)
        counters: list[Counter] = [Counter() for _ in range(num_circuits)]
        i = 0
        for counts in list_counts:
            for count in counts:
                counters[i % num_circuits].update(count)
                i += 1
        shots = sum(counters[0].values())
        quasis = [QuasiDistribution({k: v / shots for k, v in c.items()}) for c in counters]
        return quasis, shots

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> SamplerResult:
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        raw_results = [result]
        quasis, shots = self._get_quasis(raw_results)
        metadata = [res.header.metadata for result in raw_results for res in result.results]

        return SamplerResult(
            quasi_dists=quasis,
            shots=shots,
            raw_results=raw_results,
            metadata=metadata,
        )

    def _transpile(self):
        if self._skip_transpilation:
            self._transpiled_circuits = self.preprocessed_circuits

        else:
            self._transpiled_circuits = cast(
                "list[QuantumCircuit]",
                transpile(
                    self.preprocessed_circuits,
                    self.backend,
                    **self.transpile_options.__dict__,
                ),
            )
