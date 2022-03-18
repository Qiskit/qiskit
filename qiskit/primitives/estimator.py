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
Estimator class
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator import BaseEstimator
from .estimator_result import EstimatorResult
from .utils import init_circuit, init_observable


class Estimator(BaseEstimator):
    """
    Estimator class
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
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
        circuits: Sequence[int] | None = None,
        observables: Sequence[int] | None = None,
        parameters: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if parameters and not isinstance(parameters[0], Sequence):
            parameters = cast("Sequence[float]", parameters)
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
        if len(circuits) != len(parameters):
            raise QiskitError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter sets ({len(parameters)})."
            )

        bound_circuits = []
        for i, value in zip(circuits, parameters):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits.append(
                self._circuits[i].bind_parameters(dict(zip(self._parameters[i], value)))
            )
        sorted_observables = [self._observables[i] for i in observables]
        expectation_values = []
        for circ, obs in zip(bound_circuits, sorted_observables):
            if circ.num_qubits != obs.num_qubits:
                raise QiskitError(
                    f"The number of qubits of a circuit ({circ.num_qubits}) does not match "
                    f"the number of qubits of a observable ({obs.num_qubits})."
                )
            expectation_values.append(Statevector(circ).expectation_value(obs))

        return EstimatorResult(np.real_if_close(expectation_values), [])

    def close(self):
        self._is_closed = True
