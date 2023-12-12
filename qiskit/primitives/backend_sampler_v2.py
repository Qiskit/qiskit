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

"""Sampler implementation for an arbitrary Backend object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler.passmanager import PassManager

from .backend_estimator import _prepare_counts, _run_circuits
from .base import BaseSamplerV2, SamplerResult
from .containers import (
    BasePrimitiveOptions,
    BasePrimitiveOptionsLike,
    BitArray,
    PrimitiveResult,
    PubResult,
    SamplerPub,
    SamplerPubLike,
    make_data_bin,
)
from .containers.bit_array import _min_num_bytes
from .containers.dataclasses import mutable_dataclass
from .primitive_job import PrimitiveJob


@mutable_dataclass
class ExecutionOptions(BasePrimitiveOptions):
    """Options for execution."""

    shots: int = 1  # TODO: discuss the default number of shots
    seed: Optional[Union[int, np.random.Generator]] = None


@mutable_dataclass
class Options(BasePrimitiveOptions):
    """Options for the primitives.

    Args:
        execution: Execution time options. See :class:`ExecutionOptions` for all available options.
    """

    execution: ExecutionOptions = Field(default_factory=ExecutionOptions)
    transpilation: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BackendSampler(BaseSamplerV2):
    """A :class:`~.BaseSampler` implementation that provides an interface for
    leveraging the sampler interface from any backend.

    This class provides a sampler interface from any backend and doesn't do
    any measurement mitigation, it just computes the probability distribution
    from the counts. It facilitates using backends that do not provide a
    native :class:`~.BaseSampler` implementation in places that work with
    :class:`~.BaseSampler`, such as algorithms in :mod:`qiskit.algorithms`
    including :class:`~.qiskit.algorithms.minimum_eigensolvers.SamplingVQE`.
    However, if you're using a provider that has a native implementation of
    :class:`~.BaseSampler`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.
    """

    _options_class = Options
    options: Options

    def __init__(
        self,
        *,
        backend: BackendV1 | BackendV2,
        options: Optional[BasePrimitiveOptionsLike] = None,
        pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        """Initialize a new BackendSampler

        Args:
            backend: Required: the backend to run the sampler primitive on
            options: Default options.
            pass_manager: An optional pass manager to use for the internal compilation
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this objected is called.
        Raises:
            ValueError: If backend is not provided
        """
        if options is None:
            options = Options()
        elif not isinstance(options, Options):
            options = Options(**options)
        super().__init__(options=options)
        self._backend = backend
        self._circuits = []
        self._parameters = []
        self._transpile_options = Options()
        self._pass_manager = pass_manager
        self._transpiled_circuits: list[QuantumCircuit] = []
        self._skip_transpilation = skip_transpilation

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        """
        return self._transpiled_circuits

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @property
    def transpile_options(self) -> Dict[str, Any]:
        """Return the transpiler options for transpiling the circuits."""
        return self.options.transpilation

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        """
        self._options.transpilation.update(**fields)

    def _transpile(self, circuits: List[QuantumCircuit]) -> None:
        if self._skip_transpilation:
            ret = circuits
        elif self._pass_manager:
            ret = self._pass_manager.run(circuits)
        else:
            from qiskit.compiler import transpile

            ret = transpile(
                circuits,
                self.backend,
                **self.options.transpilation,
            )
        self._transpiled_circuits = ret if isinstance(ret, list) else [ret]

    def run(self, pubs: Iterable[SamplerPubLike]) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        job = PrimitiveJob(self._run, pubs)
        job.submit()
        return job

    def _run(self, pubs: Iterable[SamplerPubLike]) -> PrimitiveResult[PubResult]:
        coerced_pubs = [SamplerPub.coerce(pub) for pub in pubs]
        for pub in coerced_pubs:
            pub.validate()

        shots = self.options.execution.shots

        self._transpile([pub.circuit for pub in coerced_pubs])

        results = []
        for pub, circuit in zip(coerced_pubs, self._transpiled_circuits):
            meas_info, max_num_bits = _analyze_circuit(pub.circuit)
            max_num_bytes = _min_num_bytes(max_num_bits)
            parameter_values = pub.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)
            arrays = {
                item.creg_name: np.zeros(
                    bound_circuits.shape + (shots, item.num_bytes), dtype=np.uint8
                )
                for item in meas_info
            }
            flatten_circuits = np.ravel(bound_circuits).tolist()
            result_memory, _ = _run_circuits(
                flatten_circuits, self._backend, memory=True, **self.options.execution.__dict__
            )
            memory_list = _prepare_memory(result_memory, max_num_bits, max_num_bytes)

            for samples, index in zip(memory_list, np.ndindex(*bound_circuits.shape)):
                for item in meas_info:
                    ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                    arrays[item.creg_name][index] = ary

            data_bin_cls = make_data_bin(
                [(item.creg_name, BitArray) for item in meas_info],
                shape=bound_circuits.shape,
            )
            meas = {
                item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                for item in meas_info
            }
            data_bin = data_bin_cls(**meas)
            results.append(PubResult(data_bin, metadata={"shots": shots}))
        return PrimitiveResult(results)


def _analyze_circuit(circuit: QuantumCircuit) -> Tuple[List[_MeasureInfo], int]:
    meas_info = []
    start = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        meas_info.append(
            _MeasureInfo(
                creg_name=name,
                num_bits=num_bits,
                num_bytes=_min_num_bytes(num_bits),
                start=start,
            )
        )
        start += num_bits
    return meas_info, start


def _prepare_memory(results: List[Result], num_bits: int, num_bytes: int) -> NDArray[np.uint8]:
    lst = []
    for res in results:
        for exp in res.results:
            data = b"".join(int(i, 16).to_bytes(num_bytes, "big") for i in exp.data.memory)
            data = np.frombuffer(data, dtype=np.uint8).reshape(-1, num_bytes)
            lst.append(data)
    ary = np.array(lst, copy=False)
    return np.unpackbits(ary, axis=-1, bitorder="big")


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, start: int
) -> NDArray[np.uint8]:
    # samples of `Backend.run(memory=True)` will be the order of
    # clbit_last, ..., clbit_1, clbit_0
    # place samples in the order of clbit_start+num_bits-1, ..., clbit_start+1, clbit_start
    if start == 0:
        ary = samples[:, -start - num_bits :]
    else:
        ary = samples[:, -start - num_bits : -start]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1, bitorder="big")
    return ary
