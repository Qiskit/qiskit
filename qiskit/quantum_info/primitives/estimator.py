# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Expectation value class
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .utils import PauliSumOp, init_circuit, init_observable


class Estimator(BaseEstimator):
    """
    Evaluates expectation value using pauli rotation gates.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Optional[Iterable[Iterable[Parameter]]] = None,
    ):
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = [init_circuit(circuit) for circuit in circuits]

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = [observables]
        observables = [init_observable(observable) for observable in observables]

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
        )
        self._is_closed = False

    def __call__(
        self,
        circuits: Optional[Sequence[int]] = None,
        observables: Optional[Sequence[int]] = None,
        parameters: Optional[Sequence[Sequence[float]] | Sequence[float]] = None,
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if parameters and not isinstance(parameters[0], Sequence):
            parameters = [parameters]
        if (
            circuits is None
            and len(self._circuits) == 1
            and observables is None
            and len(self._observables) == 1
            and parameters is not None
        ):
            circuits = [0] * len(parameters)
            observables = [0] * len(parameters)
        if circuits is None:
            circuits = list(range(len(self._circuits)))
        if observables is None:
            observables = list(range(len(self._observables)))
        if parameters is None:
            parameters = [[]] * len(circuits)

        bound_circuits = [
            self._circuits[i].bind_parameters((dict(zip(self._parameters[i], value))))
            for i, value in zip(circuits, parameters)
        ]
        sorted_observables = [self._observables[i] for i in observables]
        expectation_values = [
            Statevector(circ).expectation_value(obs)
            for circ, obs in zip(bound_circuits, sorted_observables)
        ]
        expectation_values = np.real_if_close(expectation_values)

        return EstimatorResult(expectation_values, metadata={})

    def close(self):
        self._is_closed = True
