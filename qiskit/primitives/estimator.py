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
from typing import Any

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator import BaseEstimator
from .estimator_result import EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import init_circuit, init_observable


class Estimator(BaseEstimator):
    """
    Reference implementation of :class:`BaseEstimator`.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit] | None = None,
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp] | None = None,
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        if circuits is not None:
            circuits = tuple(init_circuit(circuit) for circuit in circuits)

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        if observables is not None:
            observables = tuple(init_observable(observable) for observable in observables)

        super().__init__(
            circuits=circuits,
            observables=observables,  # type: ignore
            parameters=parameters,
        )
        self._is_closed = False

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        shots = run_options.pop("shots", None)
        seed = run_options.pop("seed", None)
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[dict[str, Any]] = [{}] * len(circuits)

        bound_circuits = []
        for i, value in zip(circuits, parameter_values):
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
        for circ, obs, metadatum in zip(bound_circuits, sorted_observables, metadata):
            if circ.num_qubits != obs.num_qubits:
                raise QiskitError(
                    f"The number of qubits of a circuit ({circ.num_qubits}) does not match "
                    f"the number of qubits of a observable ({obs.num_qubits})."
                )
            final_state = Statevector(circ)
            expectation_value = final_state.expectation_value(obs)
            if shots is None:
                expectation_values.append(expectation_value)
            else:
                expectation_value = np.real_if_close(expectation_value)
                sq_obs = (obs @ obs).simplify()
                sq_exp_val = np.real_if_close(final_state.expectation_value(sq_obs))
                variance = sq_exp_val - expectation_value**2
                standard_deviation = np.sqrt(variance / shots)
                expectation_value_with_error = rng.normal(expectation_value, standard_deviation)
                expectation_values.append(expectation_value_with_error)
                metadatum["variance"] = variance
                metadatum["shots"] = shots

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def close(self):
        self._is_closed = True

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[ParameterView],
        **run_options,
    ) -> PrimitiveJob:
        circuit_indices = []
        for i, circuit in enumerate(circuits):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[id(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(parameters[i])
        observable_indices = []
        for observable in observables:
            index = self._observable_ids.get(id(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[id(observable)] = len(self._observables)
                self._observables.append(init_observable(observable))
        job = PrimitiveJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job
