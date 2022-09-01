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
"""Evaluator of auxiliary operators for algorithms."""

from typing import Tuple, Union, List, Sequence, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import (
    PauliSumOp,
)
from ..primitives import Estimator, EstimatorResult
from ..quantum_info.operators.base_operator import BaseOperator


def eval_observables(
    estimator: Estimator,
    quantum_state: QuantumCircuit,
    observables: Sequence[Union[BaseOperator, PauliSumOp]],
    threshold: float = 1e-12,
) -> List[Tuple[complex, complex]]:
    """
    Accepts a sequence of operators and calculates their expectation values - means
    and standard deviations. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: An unparametrized quantum circuit representing a quantum state that
            expectation values are computed against.
        observables: A sequence of operators whose expectation values are to be calculated.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list of tuples (mean, standard deviation).

    Raises:
        ValueError: If a ``quantum_state`` with free parameters is provided.
    """

    if (
        isinstance(quantum_state, QuantumCircuit)  # Statevector cannot be parametrized
        and len(quantum_state.parameters) > 0
    ):
        raise ValueError(
            "A parametrized representation of a quantum_state was provided. It is not "
            "allowed - it cannot have free parameters."
        )

    quantum_state = [quantum_state] * len(observables)
    estimator_job = estimator.run(quantum_state, observables)
    expectation_values = estimator_job.result().values

    # compute standard deviations
    std_devs = _compute_std_devs(estimator_job, len(expectation_values))

    # Discard values below threshold
    observables_means = expectation_values * (np.abs(expectation_values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = list(zip(observables_means, std_devs))

    # Return None eigenvalues for None operators if observables is a list.

    return _prepare_result(observables_results, observables)


def _prepare_result(
    observables_results: List[Tuple[complex, complex]],
    observables: Sequence[BaseOperator],
) -> List[Tuple[complex, complex]]:
    """
    Prepares a list of eigenvalues and standard deviations from ``observables_results`` and
    ``observables``.

    Args:
        observables_results: A list of of tuples (mean, standard deviation).
        observables: A list of operators whose expectation values are to be
            calculated.

    Returns:
        A list of tuples (mean, standard deviation).
    """

    observables_eigenvalues = [None] * len(observables)
    key_value_iterator = enumerate(observables_results)

    for key, value in key_value_iterator:
        if observables[key] is not None:
            observables_eigenvalues[key] = value
    return observables_eigenvalues


def _compute_std_devs(
    estimator_result: EstimatorResult,
    results_length: int,
) -> List[Optional[complex]]:
    """
    Calculates a list of standard deviations from expectation values of observables provided.

    Args:
        estimator_result: An estimator result.
        results_length: Number of expectation values calculated.

    Returns:
        A list of standard deviations.
    """
    if not estimator_result.metadata:
        return [0] * results_length

    std_devs = []
    for metadata in estimator_result.metadata:
        if metadata and "variance" in metadata.keys() and "shots" in metadata.keys():
            variance = metadata["variance"]
            shots = metadata["shots"]
            std_devs.append(np.sqrt(variance / shots))
        else:
            std_devs.append(0)

    return std_devs
