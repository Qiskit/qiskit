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
        circuit_indices: Sequence[int] | None = None,
        observable_indices: Sequence[int] | None = None,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()
        if parameter_values and not isinstance(parameter_values[0], (np.ndarray, Sequence)):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]
        if circuit_indices is None:
            circuit_indices = list(range(len(self._circuits)))
        if observable_indices is None:
            observable_indices = list(range(len(self._observables)))
        if parameter_values is None:
            parameter_values = [[]] * len(circuit_indices)
        if len(circuit_indices) != len(observable_indices):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of observable indices ({len(observable_indices)})."
            )
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        bound_circuits = []
        for i, value in zip(circuit_indices, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits.append(
                self._circuits[i].bind_parameters(dict(zip(self._parameters[i], value)))
            )
        sorted_observables = [self._observables[i] for i in observable_indices]
        expectation_values = []
        for circ, obs in zip(bound_circuits, sorted_observables):
            if circ.num_qubits != obs.num_qubits:
                raise QiskitError(
                    f"The number of qubits of a circuit ({circ.num_qubits}) does not match "
                    f"the number of qubits of a observable ({obs.num_qubits})."
                )
            expectation_values.append(Statevector(circ).expectation_value(obs))

        return EstimatorResult(np.real_if_close(expectation_values), [{}] * len(expectation_values))

    def close(self):
        self._is_closed = True
