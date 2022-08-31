import numpy as np
from collections.abc import Callable
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.primitives.utils import init_observable
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


def diagonal_estimation(
    sampler: BaseSampler,
    observable: PauliSumOp | BaseOperator,
    circuit: QuantumCircuit,
    values: np.ndarray | list[np.ndarray] | None = None,
    aggregation: float | Callable[[list[tuple[float, float]]], float] | None = None,
    **run_options,
) -> tuple[list[complex], list[dict]]:
    r"""Evaluate a the expectation of quantum state with respect to a diagonal operator.

    Args:
        sampler: The sampler used to evaluate the circuits.
        observable: The diagonal operator.
        circuit: The circuits preparing the quantum states. Note that this circuit must
            contain measurements already.
        values: The parameter values for the circuits. Can be a list of values which
            will be evaluated in a batch.
        aggregation: The aggregation function to aggregate the measurement outcomes. If a float
            this specified the CVaR :math:`\alpha` parameter.

    Returns:
        A tuple containing a list of expectation values and a list of the best measurements in
        each expectation value.
    """
    # value_batches = np.reshape(values, (-1, circuit.num_parameters)).tolist()
    if values is None:
        values = [np.array([])]
    elif not isinstance(values, list):
        values = [values]

    num_batches = len(values) if isinstance(values, list) else 1

    samples = sampler.run(num_batches * [circuit], values, **run_options).result().quasi_dists
    observable = init_observable(observable)

    # a list of dictionaries containing: {state: (measurement probability, value)}
    evaluations = [
        {
            state: (probability, evaluate_sparsepauli(state, observable))
            for state, probability in sampled.items()
        }
        for sampled in samples
    ]

    # get the best measurements
    best_measurements = []
    for evaluated in evaluations:
        best_result = min(evaluated.items(), key=lambda x: x[1][1])
        best_measurements.append(
            {
                "state": best_result[0],
                "bitstring": bin(best_result[0])[2:].zfill(circuit.num_qubits),
                "value": best_result[1][1],
                "probability": best_result[1][0],
            }
        )

    if not callable(aggregation):
        aggregation = get_cvar_aggregation(aggregation)

    results = [aggregation(evaluated.values()) for evaluated in evaluations]

    return results, best_measurements


def get_cvar_aggregation(alpha):
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


def evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> complex:
    return sum(
        coeff * evaluate_bitstring(state, paulistring)
        for paulistring, coeff in observable.label_iter()
    )


def evaluate_bitstring(state: int, paulistring: str) -> float:
    """Evaluate a bitstring on a Pauli label."""
    n = len(paulistring) - 1
    return np.prod(
        [-1 if state & (1 << (n - i)) else 1 for i, pauli in enumerate(paulistring) if pauli == "Z"]
    )
