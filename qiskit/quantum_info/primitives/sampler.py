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
from typing import Optional

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.quantum_info.states import Statevector
from qiskit.result import QuasiDistribution

from .utils import init_circuit


class Sampler(BaseSampler):
    """
    Sampler class
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        """
        Args:
            circuits: circuits to be executed
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = [init_circuit(circuit).remove_final_measurements(False) for circuit in circuits]
        super().__init__(circuits, parameters)
        self._is_closed = False

    def __call__(
        self,
        circuits: Optional[Sequence[int]] = None,
        parameters: Optional[Sequence[Sequence[float]]] = None,
        **run_options,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if circuits is None and parameters is not None and len(self._circuits) == 1:
            circuits = [0] * len(parameters)
        if circuits is None:
            circuits = list(range(len(self._circuits)))
        if parameters is None:
            parameters = [[] for _ in self._circuits]

        bound_circuits = [
            self._circuits[i].bind_parameters((dict(zip(self._parameters[i], value))))
            for i, value in zip(circuits, parameters)
        ]
        probabilities = [Statevector(circ).probabilities() for circ in bound_circuits]
        quasis = [QuasiDistribution(dict(enumerate(p))) for p in probabilities]

        return SamplerResult(quasis, [])

    def close(self):
        self._is_closed = True
