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

from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from .base import BaseSampler, SamplerResult
from .primitive_job import PrimitiveJob
from .utils import (
    _circuit_key,
    bound_circuit_to_instruction,
    final_measurement_mapping,
    init_circuit,
)


class Sampler(BaseSampler[PrimitiveJob[SamplerResult]]):
    """
    Sampler class.

    :class:`~Sampler` is a reference implementation of :class:`~BaseSampler`.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the probabilities.
          Otherwise, it samples from multinomial distributions.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the multinomial distribution. If shots is None, this
          option is ignored.
    """

    def __init__(self, *, options: dict | None = None):
        """
        Args:
            options: Default options.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(options=options)
        self._qargs_list = []
        self._circuit_ids = {}

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        shots = run_options.pop("shots", None)
        seed = run_options.pop("seed", None)
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        bound_circuits = []
        qargs_list = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits.append(
                self._circuits[i]
                if len(value) == 0
                else self._circuits[i].bind_parameters(dict(zip(self._parameters[i], value)))
            )
            qargs_list.append(self._qargs_list[i])
        probabilities = [
            Statevector(bound_circuit_to_instruction(circ)).probabilities_dict(
                qargs=qargs, decimals=16
            )
            for circ, qargs in zip(bound_circuits, qargs_list)
        ]
        if shots is not None:
            for i, prob_dict in enumerate(probabilities):
                counts = rng.multinomial(shots, np.fromiter(prob_dict.values(), dtype=float))
                probabilities[i] = {
                    key: count / shots for key, count in zip(prob_dict.keys(), counts) if count > 0
                }
            for metadatum in metadata:
                metadatum["shots"] = shots
        quasis = [QuasiDistribution(p, shots=shots) for p in probabilities]

        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        circuit_indices = []
        for circuit in circuits:
            key = _circuit_key(circuit)
            index = self._circuit_ids.get(key)
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[key] = len(self._circuits)
                circuit, qargs = self._preprocess_circuit(circuit)
                self._circuits.append(circuit)
                self._qargs_list.append(qargs)
                self._parameters.append(circuit.parameters)
        job = PrimitiveJob(self._call, circuit_indices, parameter_values, **run_options)
        job.submit()
        return job

    @staticmethod
    def _preprocess_circuit(circuit: QuantumCircuit):
        circuit = init_circuit(circuit)
        q_c_mapping = final_measurement_mapping(circuit)
        if set(range(circuit.num_clbits)) != set(q_c_mapping.values()):
            raise QiskitError(
                "Some classical bits are not used for measurements."
                f" the number of classical bits ({circuit.num_clbits}),"
                f" the used classical bits ({set(q_c_mapping.values())})."
            )
        c_q_mapping = sorted((c, q) for q, c in q_c_mapping.items())
        qargs = [q for _, q in c_q_mapping]
        circuit = circuit.remove_final_measurements(inplace=False)
        return circuit, qargs
