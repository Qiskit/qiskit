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

from collections.abc import Iterable, Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from .base_sampler import BaseSampler
from .sampler_result import SamplerResult
from .statevector_primitive import StatevectorPrimitive
from .utils import final_measurement_mapping, init_circuit


class Sampler(BaseSampler, StatevectorPrimitive):
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

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        """
        Args:
            circuits: circuits to be executed
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)
        q_c_mappings = [final_measurement_mapping(circuit) for circuit in circuits]
        self._qargs_list = []
        for circuit, q_c_mapping in zip(circuits, q_c_mappings):
            if set(range(circuit.num_clbits)) != set(q_c_mapping.values()):
                raise QiskitError(
                    "some classical bits are not used for measurements."
                    f" the number of classical bits {circuit.num_clbits},"
                    f" the used classical bits {set(q_c_mapping.values())}."
                )
            c_q_mapping = sorted((c, q) for q, c in q_c_mapping.items())
            self._qargs_list.append([q for _, q in c_q_mapping])
        circuits = tuple(circuit.remove_final_measurements(inplace=False) for circuit in circuits)
        super().__init__(circuits, parameters)
        self._is_closed = False

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        # Parse input
        states = [
            self._build_statevector(circuit_index, tuple(values))
            for circuit_index, values in zip(circuits, parameter_values)
        ]
        qargs_list = [self._qargs_list[i] for i in circuits]
        shots = run_options.pop("shots", None)
        rng = self._parse_rng_from_seed(run_options.pop("seed", None))

        # Results
        raw_results = [
            self._compute_result(state, qargs, shots, rng) for state, qargs in zip(states, qargs_list)
        ]
        probabilities, metadata = zip(*raw_results)
        quasis = [QuasiDistribution(dict(enumerate(p))) for p in probabilities]
        return SamplerResult(quasis, metadata)

    def close(self):
        self._is_closed = True

    def _compute_result(
        self, state: Statevector, qargs: list[int], shots: int, rng: np.random.Generator
    ) -> tuple[float, dict]:
        probability = state.probabilities(qargs=qargs)
        metadatum = {}
        if shots is not None:
            probability = rng.multinomial(shots, probability) / shots
            metadatum["shots"] = shots
        return probability, metadatum
