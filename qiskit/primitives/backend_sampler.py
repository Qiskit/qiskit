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

"""Sampler implementation for an artibtrary Backend object."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
from typing import cast, Any

from qiskit.exceptions import QiskitError
from qiskit.providers.options import Options
from qiskit.result import QuasiDistribution
from .base_sampler import BaseSampler
from .sampler_result import SamplerResult
from .primitive_job import PrimitiveJob


class BackendSampler(BaseSampler):
    """A :class:`~.BaseSampler` implementation that provides an interface for leveraging
    the sampler interface from any backend.

    This class provides a sampler interface from any backend and doesn't do
    any measurement mitigation, it just computes the probability distribution
    from the counts.
    """

    def __init__(
        self,
        backend: "Backend" = None,
        circuits: "QuantumCircuit" | Iterable["QuantumCircuit"] = None,
        parameters: Iterable[Iterable["Parameter"]] | None = None,
        bound_pass_manager: "PassManager" | None = None,
        skip_transpilation: bool = False,
    ):
        """Initialize a new BackendSampler

        Args:
            backend: Required: the backend to run the sampler primitive on
            circuits: The Quantum circuits to be executed.
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.
            bound_pass_manager: An optional pass manager to run after
                parameter binding.
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this objected is called.
        Raises:
            ValueError: If backend is not provided
        """

        super().__init__(circuits, parameters)
        if backend is None:
            raise ValueError("A backend is required to use BackendSampler")
        self._backend = backend
        self._run_options = self._backend.options
        self._is_closed = False
        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager
        self._preprocessed_circuits: list["QuantumCircuit"] | None = None
        self._transpiled_circuits: list["QuantumCircuit"] | None = None
        self._skip_transpilation = skip_transpilation

    @property
    def preprocessed_circuits(self) -> list["QuantumCircuit"]:
        """
        Preprocessed quantum circuits produced by preprocessing
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
        return list(self._circuits)

    @property
    def transpiled_circuits(self) -> list["QuantumCircuit"]:
        """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
        if self._skip_transpilation:
            self._transpiled_circuits = list(self._circuits)
        elif self._transpiled_circuits is None:
            # Only transpile if have not done so yet
            self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> "Backend":
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> BackendSampler:
        """Set options values for the evaluator.
        Args:
            **fields: The fields to update the options
        Returns:
            self
        """
        self._check_is_closed()
        self._run_options.update_options(**fields)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> BackendSampler:
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()

        self._transpile_options.update_options(**fields)
        return self

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        self._check_is_closed()

        # This line does the actual transpilation
        transpiled_circuits = self.transpiled_circuits
        bound_circuits = [
            transpiled_circuits[i]
            if len(value) == 0
            else transpiled_circuits[i].bind_parameters((dict(zip(self._parameters[i], value))))
            for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        result = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        return self._postprocessing(result, bound_circuits)

    def close(self):
        self._is_closed = True

    def _postprocessing(self, result: "Result", circuits: list["QuantumCircuit"]) -> SamplerResult:

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

        shots = sum(counts[0].values())

        probabilies = []
        metadata: list[dict[str, Any]] = [{}] * len(circuits)
        for count in counts:
            prob_dist = {k: v / shots for k, v in count.int_outcomes().items()}
            probabilies.append(QuasiDistribution(prob_dist))
            for metadatum in metadata:
                metadatum["shots"] = shots
        return SamplerResult(probabilies, metadata)

    def _transpile(self):
        from qiskit.compiler import transpile

        self._transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(
                self.preprocessed_circuits,
                self.backend,
                **self.transpile_options.__dict__,
            ),
        )

    def _check_is_closed(self):
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return cast("list[QuantumCircuit]", self._bound_pass_manager.run(circuits))

    def _run(
        self,
        circuits: Sequence["QuantumCircuit"],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> PrimitiveJob:
        circuit_indices = []
        for circuit in circuits:
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[id(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        job = PrimitiveJob(self._call, circuit_indices, parameter_values, **run_options)
        job.submit()
        return job
