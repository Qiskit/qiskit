# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sampler V2 implementation for an arbitrary BackendV2 object."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    make_data_bin,
)
from qiskit.primitives.containers.bit_array import _min_num_bytes
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers.backend import BackendV2
from qiskit.result import Result


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BackendSamplerV2(BaseSamplerV2):
    """
    Implementation of :class:`BaseSamplerV2` using a backend.

    This class provides a SamplerV2 interface from any :class:`~.BackendV2` backend
    and doesn't do any measurement mitigation, it just computes the bitstrings.

    This sampler supports providing arrays of parameter value sets to
    bind against a single circuit.

    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    .. note::

        This class requires a backend that supports ``memory`` option.

    """

    def __init__(
        self,
        *,
        backend: BackendV2,
        default_shots: int = 1024,
    ):
        """
        Args:
            backend: Required: the backend to run the sampler primitive on
            default_shots: The default shots for the sampler if not specified during run.
        """
        super().__init__()
        self._backend = backend
        self._default_shots = default_shots

    @property
    def backend(self) -> BackendV2:
        """Returns the backend which this sampler object based on."""
        return self._backend

    @property
    def default_shots(self) -> int:
        """Return the default shots"""
        return self._default_shots

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[PubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: SamplerPub) -> PubResult:
        meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
        bound_circuits = pub.parameter_values.bind_all(pub.circuit)
        arrays = {
            item.creg_name: np.zeros(
                bound_circuits.shape + (pub.shots, item.num_bytes), dtype=np.uint8
            )
            for item in meas_info
        }
        flatten_circuits = np.ravel(bound_circuits).tolist()
        result_memory, _ = _run_circuits(
            flatten_circuits, self._backend, memory=True, shots=pub.shots
        )
        memory_list = _prepare_memory(result_memory, max_num_bytes)

        for samples, index in zip(memory_list, np.ndindex(*bound_circuits.shape)):
            for item in meas_info:
                ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                arrays[item.creg_name][index] = ary

        data_bin_cls = make_data_bin(
            [(item.creg_name, BitArray) for item in meas_info],
            shape=bound_circuits.shape,
        )
        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info
        }
        data_bin = data_bin_cls(**meas)
        return PubResult(data_bin, metadata={"shots": pub.shots})


def _analyze_circuit(circuit: QuantumCircuit) -> tuple[list[_MeasureInfo], int]:
    meas_info = []
    max_num_bits = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        start = circuit.find_bit(creg[0]).index
        meas_info.append(
            _MeasureInfo(
                creg_name=name,
                num_bits=num_bits,
                num_bytes=_min_num_bytes(num_bits),
                start=start,
            )
        )
        max_num_bits = max(max_num_bits, start + num_bits)
    return meas_info, _min_num_bytes(max_num_bits)


def _prepare_memory(results: list[Result], num_bytes: int) -> NDArray[np.uint8]:
    lst = []
    for res in results:
        for exp in res.results:
            if hasattr(exp.data, "memory") and exp.data.memory:
                data = b"".join(int(i, 16).to_bytes(num_bytes, "big") for i in exp.data.memory)
                data = np.frombuffer(data, dtype=np.uint8).reshape(-1, num_bytes)
            else:
                # no measure in a circuit
                data = np.zeros((exp.shots, num_bytes), dtype=np.uint8)
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
