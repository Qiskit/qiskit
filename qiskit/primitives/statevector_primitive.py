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
Common functionality for primitives based on the Statevector construct
"""
from functools import lru_cache

from numpy.random import Generator, default_rng
from qiskit.circuit import Parameter, QuantumCircuit

cache = lru_cache(maxsize=None)


class StatevectorPrimitive:
    """
    Common functionality for primitives based on the Statevector construct.

    It provides caching capabilities for Statevector construction.
    """

    def _bind_circuit_parameters(
        self, circuit_index: int, parameter_values: tuple[float]
    ) -> QuantumCircuit:
        parameters = self._parameters[circuit_index]
        if len(parameter_values) != len(parameters):
            raise ValueError(
                f"The number of values ({len(parameter_values)}) does not match "
                f"the number of parameters ({len(parameters)})."
            )
        circuit = self._circuits[circuit_index]
        if not parameter_values:
            return circuit
        parameter_mapping = dict(zip(parameters, parameter_values))
        return circuit.bind_parameters(parameter_mapping)

    @cache  # Enables memoization (tuples are hashable)
    def _build_statevector(self, circuit_index: int, parameter_values: tuple[float]) -> Statevector:
        circuit = self._bind_circuit_parameters(circuit_index, parameter_values)
        return Statevector(circuit)

    def _parse_rng_from_seed(self, seed: None | int | Generator):
        if seed is None:
            return default_rng()
        elif isinstance(seed, Generator):
            return seed
        else:
            return default_rng(seed)
