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

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator import BaseEstimator
from .estimator_result import EstimatorResult
from .statevector_primitive import StatevectorPrimitive
from .utils import init_circuit, init_observable, rng_from_seed


class Estimator(BaseEstimator, StatevectorPrimitive):
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
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        observables = tuple(init_observable(observable) for observable in observables)

        super().__init__(
            circuits=circuits,
            observables=observables,
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

        # Parse input
        states = [
            self._build_statevector(circuit_index, tuple(values))
            for circuit_index, values in zip(circuits, parameter_values)
        ]
        observables = [self._observables[i] for i in observables]
        shots = run_options.pop("shots", None)
        rng = rng_from_seed(run_options.pop("seed", None))

        # Results
        raw_results = [
            self._compute_result(state, observable, shots, rng)
            for state, observable in zip(states, observables)
        ]
        expectation_values, metadata = zip(*raw_results)
        return EstimatorResult(np.array(expectation_values), metadata)

    def close(self):
        self._is_closed = True

    def _compute_result(
        self,
        state: Statevector,
        observable: BaseOperator | PauliSumOp,
        shots: int,
        rng: np.random.Generator,
    ) -> tuple[float, dict]:
        if state.num_qubits != observable.num_qubits:
            raise QiskitError(
                f"The number of qubits of a circuit ({state.num_qubits}) does not match "
                f"the number of qubits of a observable ({observable.num_qubits})."
            )
        expectation_value = np.real_if_close(state.expectation_value(observable))
        metadatum = {}
        if shots is not None:
            sq_obs = (observable @ observable).simplify()
            sq_exp_val = np.real_if_close(state.expectation_value(sq_obs))
            variance = sq_exp_val - expectation_value**2
            standard_error = np.sqrt(variance / shots)
            expectation_value = rng.normal(expectation_value, standard_error)
            metadatum["variance"] = variance
            metadatum["shots"] = shots
        return float(expectation_value), metadatum
