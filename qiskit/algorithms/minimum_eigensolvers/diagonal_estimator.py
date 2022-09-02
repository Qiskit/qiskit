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

"""Expectation value for a diagonal observable using a sampler primitive."""

from collections.abc import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.primitives.utils import init_observable
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


def diagonal_estimation(
    sampler: BaseSampler,
    observable: PauliSumOp | BaseOperator | list[PauliSumOp | BaseOperator],
    circuit: QuantumCircuit | list[QuantumCircuit],
    values: np.ndarray | list[np.ndarray] | None = None,
    aggregation: float | Callable[[list[tuple[float, float]]], float] | None = None,
    return_best_measurement: bool = False,
    **run_options,
) -> list[complex] | tuple[list[complex], list[dict]]:
    r"""Evaluate a the expectation of quantum state with respect to a diagonal operator.

    Args:
        sampler: The sampler used to evaluate the circuits.
        observable: The diagonal operator.
        circuit: The circuits preparing the quantum states. Note that this circuit must
            contain measurements already.
        values: The parameter values for the circuits. Can be a list of values which
            will be evaluated in a batch. If the observable and circuit are a single object and
            the values are a list of arrays, the observable and circuit are broadcasted to
            the size of the values.
        aggregation: The aggregation function to aggregate the measurement outcomes. If a float
            this specified the CVaR :math:`\alpha` parameter.
        return_best_measurement: If True, return a dict specifying the best measurement along
            to the expectation value.
        run_options: Run options for the sampler.

    Returns:
        A tuple containing a list of expectation values and a list of the best measurements in
        each expectation value.
    """
    # TODO check if observables are all diagonal

    if values is None:
        values = [np.array([])]
    elif not isinstance(values, list):
        values = [values]

    # broadcast if necessary
    if not isinstance(circuit, list) and not isinstance(observable, list):
        observable = init_observable(observable)  # only do this conversion once, before broadcast
        num_batches = len(values)
        observables = [observable] * num_batches
        circuits = [circuit] * num_batches
    else:
        observables = [init_observable(obs) for obs in observable]
        circuits = circuit

    samples = sampler.run(circuits, values, **run_options).result().quasi_dists

    # a list of dictionaries containing: {state: (measurement probability, value)}
    evaluations = [
        {
            state: (probability, _evaluate_sparsepauli(state, observable))
            for state, probability in sampled.items()
        }
        for observable, sampled in zip(observables, samples)
    ]

    if not callable(aggregation):
        aggregation = _get_cvar_aggregation(aggregation)

    results = [aggregation(evaluated.values()) for evaluated in evaluations]

    if not return_best_measurement:
        return results

    # get the best measurements
    best_measurements = []
    num_qubits = circuits[0].num_qubits
    for evaluated in evaluations:
        best_result = min(evaluated.items(), key=lambda x: x[1][1])
        best_measurements.append(
            {
                "state": best_result[0],
                "bitstring": bin(best_result[0])[2:].zfill(num_qubits),
                "value": best_result[1][1],
                "probability": best_result[1][0],
            }
        )

    return results, best_measurements


def _get_cvar_aggregation(alpha):
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):

        def aggregate(measurements):
            return sum(probability * value for probability, value in measurements)

    else:

        def aggregate(measurements):
            # sort by values
            sorted_measurements = sorted(measurements, key=lambda x: x[1])

            accumulated_percent = 0  # once alpha is reached, stop
            cvar = 0
            for probability, value in sorted_measurements:
                cvar += value * max(probability, alpha - accumulated_percent)
                accumulated_percent += probability
                if accumulated_percent >= alpha:
                    break

            return cvar / alpha

    return aggregate


def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> complex:
    return sum(
        coeff * _evaluate_bitstring(state, paulistring)
        for paulistring, coeff in observable.label_iter()
    )


def _evaluate_bitstring(state: int, paulistring: str) -> float:
    """Evaluate a bitstring on a Pauli label."""
    n = len(paulistring) - 1
    return np.prod(
        [-1 if state & (1 << (n - i)) else 1 for i, pauli in enumerate(paulistring) if pauli == "Z"]
    )
