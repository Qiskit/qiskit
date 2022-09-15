# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Evaluator of observables for algorithms."""
from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from .exceptions import AlgorithmError
from .list_or_dict import ListOrDict
from ..primitives import EstimatorResult, BaseEstimator
from ..quantum_info.operators.base_operator import BaseOperator


def estimate_observables(
    estimator: BaseEstimator,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator | PauliSumOp],
    threshold: float = 1e-12,
) -> ListOrDict[tuple[complex, tuple[complex, int]]]:
    """
    Accepts a sequence of operators and calculates their expectation values - means
    and standard deviations. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: An unparametrized quantum circuit representing a quantum state that
            expectation values are computed against.
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list or a dictionary of tuples (mean, (variance, shots)).

    Raises:
        ValueError: If a ``quantum_state`` with free parameters is provided.
        AlgorithmError: If a primitive job is not successful.
    """

    if (
        isinstance(quantum_state, QuantumCircuit)  # State cannot be parametrized
        and len(quantum_state.parameters) > 0
    ):
        raise ValueError(
            "A parametrized representation of a quantum_state was provided. It is not "
            "allowed - it cannot have free parameters."
        )
    if isinstance(observables, dict):
        observables_list = list(observables.values())
    else:
        observables_list = observables

    observables_list = _handle_zero_ops(observables_list)
    quantum_state = [quantum_state] * len(observables)
    try:
        estimator_job = estimator.run(quantum_state, observables_list)
        expectation_values = estimator_job.result().values
    except Exception as exc:
        raise AlgorithmError("The primitive job failed!") from exc

    variance_and_shots = _prep_variance_and_shots(estimator_job, len(expectation_values))

    # Discard values below threshold
    observables_means = expectation_values * (np.abs(expectation_values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = list(zip(observables_means, variance_and_shots))

    return _prepare_result(observables_results, observables)


def _handle_zero_ops(
    observables_list: list[BaseOperator | PauliSumOp],
) -> list[BaseOperator | PauliSumOp]:
    """Replaces all occurrence of operators equal to 0 in the list with an equivalent ``PauliSumOp``
    operator."""
    if observables_list:
        zero_op = PauliSumOp.from_list([("I" * observables_list[0].num_qubits, 0)])
        for ind, observable in enumerate(observables_list):
            if observable == 0:
                observables_list[ind] = zero_op
    return observables_list


def _prepare_result(
    observables_results: list[tuple[complex, tuple[complex, int]]],
    observables: ListOrDict[BaseOperator | PauliSumOp],
) -> ListOrDict[tuple[complex, tuple[complex, int]]]:
    """
    Prepares a list of tuples of eigenvalues and (variance, shots) tuples from
    ``observables_results`` and ``observables``.

    Args:
        observables_results: A list of tuples (mean, (variance, shots)).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.

    Returns:
        A list or a dictionary of tuples (mean, (variance, shots)).
    """

    if isinstance(observables, list):
        # by construction, all None values will be overwritten
        observables_eigenvalues = [None] * len(observables)
        key_value_iterator = enumerate(observables_results)
    else:
        observables_eigenvalues = {}
        key_value_iterator = zip(observables.keys(), observables_results)

    for key, value in key_value_iterator:
        observables_eigenvalues[key] = value
    return observables_eigenvalues


def _prep_variance_and_shots(
    estimator_result: EstimatorResult,
    results_length: int,
) -> list[tuple[complex, int]]:
    """
    Prepares a list of tuples with variances and shots from results provided by expectation values
    calculations. If there is no variance or shots data available from a primitive, the values will
    be set to ``0``.

    Args:
        estimator_result: An estimator result.
        results_length: Number of expectation values calculated.

    Returns:
        A list of tuples of the form (variance, shots).
    """
    if not estimator_result.metadata:
        return [(0, 0)] * results_length

    results = []
    for metadata in estimator_result.metadata:
        variance, shots = 0.0, 0
        if metadata:
            if "variance" in metadata.keys():
                variance = metadata["variance"]
            if "shots" in metadata.keys():
                shots = metadata["shots"]

        results.append((variance, shots))

    return results
