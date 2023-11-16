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
Sampler class
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

from .base import BaseSamplerV2
from .containers import BasePrimitiveOptions, BasePrimitiveOptionsLike, SamplerTask, TaskResult
from .containers.bit_array import BitArray
from .containers.options import mutable_dataclass
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


@mutable_dataclass
class ExecutionOptions(BasePrimitiveOptions):
    """Options for execution."""

    shots: Optional[int] = None
    seed: Optional[Union[int, np.random.Generator]] = None


@mutable_dataclass
class Options(BasePrimitiveOptions):
    """Options for the primitives.

    Args:
        execution: Execution time options. See :class:`ExecutionOptions` for all available options.
    """

    execution: ExecutionOptions = Field(default_factory=ExecutionOptions)


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

    def _run(self, tasks: list[SamplerTask]) -> PrimitiveJob[list[TaskResult]]:
        job = PrimitiveJob(self._run_task, tasks)
        job.submit()
        return job

    def _run_task(self, tasks: list[SamplerTask]) -> list[TaskResult]:
        shots = self.options.execution.shots or 1

        results = []
        for task in tasks:
            circuit, qargs, num_bits, q_indices, packed_sizes = self._preprocess_circuit(
                task.circuit
            )
            parameter_values = task.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)
            arrays = {
                name: np.zeros(bound_circuits.shape + (shots, packed_size), dtype=np.uint8)
                for name, packed_size in packed_sizes.items()
            }
            for index in np.ndindex(*bound_circuits.shape):
                bound_circuit = bound_circuits[index]
                final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
                samples = final_state.sample_memory(shots=shots, qargs=qargs)
                for name in num_bits:
                    ary = self._samples_to_packed_array(samples, num_bits[name], q_indices[name])
                    arrays[name][index] = ary
            meas = {name: BitArray(arrays[name], num_bits[name]) for name in num_bits}
            results.append(TaskResult(meas, metadata={"shots": shots}))

        return results

    @staticmethod
    def _preprocess_circuit(circuit: QuantumCircuit):
        mapping = _final_measurement_mapping(circuit)
        qargs = list(mapping.values())
        circuit = circuit.remove_final_measurements(inplace=False)
        num_qubits = circuit.num_qubits
        num_bits = {key[0].name: key[0].size for key in mapping}
        # num_qubits is used as sentinel to fill 0
        indices = {key: [num_qubits] * val for key, val in num_bits.items()}
        for key, qreg in mapping.items():
            creg, ind = key
            indices[creg.name][ind] = qreg
        packed_sizes = {
            name: num_bits // 8 + (num_bits % 8 > 0) for name, num_bits in num_bits.items()
        }
        return circuit, qargs, num_bits, indices, packed_sizes

    @staticmethod
    def _samples_to_packed_array(
        samples: NDArray[str], num_bits: int, indices: list[int]
    ) -> NDArray[np.uint8]:
        pad_size = (8 - num_bits % 8) % 8
        ary = np.array([np.fromiter(sample, dtype=np.uint8) for sample in samples])
        # pad 0 to be used for the sentinel introduced by _preprocess_circuit
        ary = np.pad(ary, ((0, 0), (0, 1)), constant_values=0)
        ary = ary[:, indices]
        ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
        ary = np.packbits(ary, axis=-1)
        return ary


def _final_measurement_mapping(circuit: QuantumCircuit) -> dict[tuple[ClassicalRegister, int], int]:
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
    for item in circuit._data[::-1]:
        if item.operation.name == "measure":
            loc = circuit.find_bit(item.clbits[0])
            cbit = loc.index
            creg = loc.registers[0]
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[creg] = qbit
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item.operation.name not in ["barrier", "delay"]:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    return mapping
