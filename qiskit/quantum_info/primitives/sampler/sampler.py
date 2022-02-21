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

from typing import Optional, Union, cast

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import BaseReadoutMitigator, QuasiDistribution, Result
from qiskit.transpiler import PassManager

from ..framework.base_primitive import BasePrimitive
from ..framework.utils import init_circuit
from ..results import SamplerResult


class Sampler(BasePrimitive):
    """
    Sampler class
    """

    def __init__(
        self,
        backend: Backend,
        mitigator: Optional[BaseReadoutMitigator] = None,
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
        bound_pass_manager: Optional[PassManager] = None,
    ):
        """
        Args:
            backend: a backend or a backend wrapper
            circuits: circuits to be executed
        """
        super().__init__(
            backend=backend,
            mitigator=mitigator,
            bound_pass_manager=bound_pass_manager,
        )
        if circuits is None:
            self._circuits = None
        elif isinstance(circuits, list):
            self._circuits = [init_circuit(circuit) for circuit in circuits]
        else:
            self._circuits = [init_circuit(circuits)]

        self._skip_transpilation = False

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """A list of quantum circuits to be executed

        Returns:
            a list of quantum circuits
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
        return self._backend

    def _postprocessing(self, result: Result) -> SamplerResult:
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

        shots = sum(counts[0].values())

        quasis = []
        for c in counts:
            if self._mitigator is None:
                quasis.append(QuasiDistribution({k: v / shots for k, v in c.items()}))
            else:
                quasis.append(self._mitigator.quasi_probabilities(c))

        metadata = [res.header.metadata for res in result.results]

        return SamplerResult(
            quasi_dists=quasis,
            shots=shots,
            raw_results=result,
            metadata=metadata,
        )

    def _transpile(self):
        self._transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(
                self.preprocessed_circuits,
                self.backend,
                **self.transpile_options.__dict__,
            ),
        )
