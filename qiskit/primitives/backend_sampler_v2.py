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

"""Sampler V2 implementation for an arbitrary Backend object."""

from __future__ import annotations

import warnings
from collections import defaultdict
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
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.result import Result


@dataclass
class Options:
    """Options for :class:`~.BackendSamplerV2`"""

    default_shots: int = 1024
    """The default shots to use if none are specified in :meth:`~.run`.
    Default: 1024.
    """

    seed_simulator: int | None = None
    """The seed to use in the simulator. If None, a random seed will be used.
    Default: None.
    """


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BackendSamplerV2(BaseSamplerV2):
    """Evaluates bitstrings for provided quantum circuits

    The :class:`~.BackendSamplerV2` class is a generic implementation of the
    :class:`~.BaseSamplerV2` interface that is used to wrap a :class:`~.BackendV2`
    (or :class:`~.BackendV1`) object in the class :class:`~.BaseSamplerV2` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseSamplerV2` implementation in places that work with
    :class:`~.BaseSamplerV2`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseSamplerV2`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.

    This class does not perform any measurement or gate mitigation.

    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    The options for :class:`~.BackendSamplerV2` consist of the following items.

    * ``default_shots``: The default shots to use if none are specified in :meth:`~run`.
      Default: 1024.

    * ``seed_simulator``: The seed to use in the simulator. If None, a random seed will be used.
      Default: None.

    .. note::

        This class requires a backend that supports ``memory`` option.

    """

    def __init__(
        self,
        *,
        backend: BackendV1 | BackendV2,
        options: dict | None = None,
    ):
        """
        Args:
            backend: The backend to run the primitive on.
            options: The options to control the default shots (``default_shots``) and
                the random seed for the simulator (``seed_simulator``).
        """
        self._backend = backend
        self._options = Options(**options) if options else Options()

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """Returns the backend which this sampler object based on."""
        return self._backend

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if shots is None:
            shots = self._options.default_shots
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
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results)

    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[PubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        # prepare circuits
        bound_circuits = [pub.parameter_values.bind_all(pub.circuit) for pub in pubs]
        flatten_circuits = []
        for circuits in bound_circuits:
            flatten_circuits.extend(np.ravel(circuits).tolist())

        # run circuits
        results, _ = _run_circuits(
            flatten_circuits,
            self._backend,
            memory=True,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
        )
        result_memory = _prepare_memory(results)

        # pack memory to an ndarray of uint8
        results = []
        start = 0
        for pub, bound in zip(pubs, bound_circuits):
            meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
            end = start + bound.size
            results.append(
                self._postprocess_pub(
                    result_memory[start:end], shots, bound.shape, meas_info, max_num_bytes
                )
            )
            start = end

        return results

    def _postprocess_pub(
        self,
        result_memory: list[list[str]],
        shots: int,
        shape: tuple[int, ...],
        meas_info: list[_MeasureInfo],
        max_num_bytes: int,
    ) -> PubResult:
        """Converts the memory data into an array of bit arrays with the shape of the pub."""
        arrays = {
            item.creg_name: np.zeros(shape + (shots, item.num_bytes), dtype=np.uint8)
            for item in meas_info
        }
        memory_array = _memory_array(result_memory, max_num_bytes)

        for samples, index in zip(memory_array, np.ndindex(*shape)):
            for item in meas_info:
                ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                arrays[item.creg_name][index] = ary

        data_bin_cls = make_data_bin(
            [(item.creg_name, BitArray) for item in meas_info],
            shape=shape,
        )
        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info
        }
        data_bin = data_bin_cls(**meas)
        return PubResult(data_bin, metadata={})


def _analyze_circuit(circuit: QuantumCircuit) -> tuple[list[_MeasureInfo], int]:
    """Analyzes the information for each creg in a circuit."""
    meas_info = []
    max_num_bits = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        if num_bits != 0:
            start = circuit.find_bit(creg[0]).index
        else:
            start = 0
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


def _prepare_memory(results: list[Result]) -> list[list[str]]:
    """Joins splitted results if exceeding max_experiments"""
    lst = []
    for res in results:
        for exp in res.results:
            if hasattr(exp.data, "memory") and exp.data.memory:
                lst.append(exp.data.memory)
            else:
                # no measure in a circuit
                lst.append(["0x0"] * exp.shots)
    return lst


def _memory_array(results: list[list[str]], num_bytes: int) -> NDArray[np.uint8]:
    """Converts the memory data into an array in an unpacked way."""
    lst = []
    for memory in results:
        if num_bytes > 0:
            data = b"".join(int(i, 16).to_bytes(num_bytes, "big") for i in memory)
            data = np.frombuffer(data, dtype=np.uint8).reshape(-1, num_bytes)
        else:
            # no measure in a circuit
            data = np.zeros((len(memory), num_bytes), dtype=np.uint8)
        lst.append(data)
    ary = np.asarray(lst)
    return np.unpackbits(ary, axis=-1, bitorder="big")


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, start: int
) -> NDArray[np.uint8]:
    """Converts an unpacked array of the memory data into a packed array."""
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
