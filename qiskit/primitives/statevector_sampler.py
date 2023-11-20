# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Statevector Sampler class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from qiskit import ClassicalRegister, QiskitError, QuantumCircuit
from qiskit.quantum_info import Statevector

from .base import BaseSamplerV2
from .base.validation import _has_measure
from .containers import BasePrimitiveOptions, BasePrimitiveOptionsLike, SamplerTask, TaskResult
from .containers.bit_array import BitArray
from .containers.data_bin import make_databin
from .containers.options import mutable_dataclass
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


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


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    packed_size: int
    qreg_indices: List[int]


class StatevectorSampler(BaseSamplerV2[PrimitiveJob[List[TaskResult]]]):
    """
    Simple implementation of :class:`BaseSamplerV2` with Statevector.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
    """

    _options_class = Options

    def __init__(self, *, options: Optional[BasePrimitiveOptionsLike] = None):
        """
        Args:
            options: Options including shots, seed.
        """
        if options is None:
            options = Options()
        elif not isinstance(options, Options):
            options = Options(**options)
        super().__init__(options=options)

    def _run(self, tasks: List[SamplerTask]) -> PrimitiveJob[List[TaskResult]]:
        job = PrimitiveJob(self._run_task, tasks)
        job.submit()
        return job

    def _run_task(self, tasks: List[SamplerTask]) -> List[TaskResult]:
        shots = self.options.execution.shots
        if shots is None:
            raise QiskitError("`shots` should be a positive integer")
        seed = self.options.execution.seed

        results = []
        for task in tasks:
            circuit, qargs, meas_info = _preprocess_circuit(task.circuit)
            parameter_values = task.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)
            arrays = {
                item.creg_name: np.zeros(
                    bound_circuits.shape + (shots, item.packed_size), dtype=np.uint8
                )
                for item in meas_info
            }

            for index in np.ndindex(*bound_circuits.shape):
                bound_circuit = bound_circuits[index]
                final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
                final_state.seed(seed)
                samples = final_state.sample_memory(shots=shots, qargs=qargs)
                for item in meas_info:
                    ary = _samples_to_packed_array(samples, item.num_bits, item.qreg_indices)
                    arrays[item.creg_name][index] = ary

            data_bin_cls = make_databin(
                [(item.creg_name, BitArray) for item in meas_info],
                shape=bound_circuits.shape,
            )
            meas = {
                item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                for item in meas_info
            }
            data_bin = data_bin_cls(**meas)
            results.append(TaskResult(data_bin, metadata={"shots": shots}))

        return results


def _preprocess_circuit(circuit: QuantumCircuit):
    mapping = _final_measurement_mapping(circuit)
    qargs = sorted(set(mapping.values()))
    circuit = circuit.remove_final_measurements(inplace=False)
    if _has_measure(circuit):
        raise QiskitError("StatevectorSampler cannot handle mid-circuit measurements")
    num_qubits = circuit.num_qubits
    num_bits_dict = {key[0].name: key[0].size for key in mapping}
    # num_qubits is used as sentinel to fill 0 in _samples_to_packed_array
    indices = {key: [num_qubits] * val for key, val in num_bits_dict.items()}
    for key, qreg in mapping.items():
        creg, ind = key
        indices[creg.name][ind] = qreg
    meas_info = [
        _MeasureInfo(
            creg_name=name,
            num_bits=num_bits,
            qreg_indices=indices[name],
            packed_size=num_bits // 8 + (num_bits % 8 > 0),
        )
        for name, num_bits in num_bits_dict.items()
    ]
    return circuit, qargs, meas_info


def _samples_to_packed_array(
    samples: NDArray[str], num_bits: int, indices: List[int]
) -> NDArray[np.uint8]:
    # samples of `Statevector.sample_memory` will be the order of
    # qubit_0, qubit_1, ..., qubit_last
    ary = np.array([np.fromiter(sample, dtype=np.uint8) for sample in samples])
    # pad 0 in the rightmost to be used for the sentinel introduced by _preprocess_circuit
    ary = np.pad(ary, ((0, 0), (0, 1)), constant_values=0)
    # place samples in the order of clbit_last, ..., clbit_1, clbit_0
    ary = ary[:, indices[::-1]]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = (8 - num_bits % 8) % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1)
    return ary


def _final_measurement_mapping(circuit: QuantumCircuit) -> Dict[Tuple[ClassicalRegister, int], int]:
    """Return the final measurement mapping for the circuit.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of classical bits to qubits for final measurements.
    """
    active_qubits = set(range(circuit.num_qubits))
    active_cbits = set(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit[::-1]:
        if item.operation.name == "measure":
            loc = circuit.find_bit(item.clbits[0])
            cbit = loc.index
            creg = loc.registers[0]
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[creg] = qbit
                active_cbits.remove(cbit)
        elif item.operation.name not in ["barrier", "delay"]:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    return mapping
