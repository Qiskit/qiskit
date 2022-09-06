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
from __future__ import annotations

from typing import Tuple, List

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import (
    PauliSumOp,
)
from . import AlgorithmError
from .list_or_dict import ListOrDict
from ..primitives import EstimatorResult, BaseEstimator
from ..quantum_info.operators.base_operator import BaseOperator


def eval_observables(
    estimator: BaseEstimator,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator | PauliSumOp],
    threshold: float = 1e-12,
) -> ListOrDict[Tuple[complex, complex]]:
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
        A list or a dictionary of tuples (mean, standard deviation).

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
    quantum_state = [quantum_state] * len(observables)
    try:
        estimator_job = estimator.run(quantum_state, observables_list)
        expectation_values = estimator_job.result().values
    except Exception as exc:
        raise AlgorithmError("The primitive job failed!") from exc

    std_devs = _compute_std_devs(estimator_job, len(expectation_values))

    # Discard values below threshold
    observables_means = expectation_values * (np.abs(expectation_values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = list(zip(observables_means, std_devs))

    # Return None eigenvalues for None operators if observables is a list.
    return _prepare_result(observables_results, observables)


def _prepare_result(
    observables_results: List[Tuple[complex, complex]],
    observables: ListOrDict[BaseOperator | PauliSumOp],
) -> ListOrDict[Tuple[complex, complex]]:
    """
    Prepares a list of eigenvalues and standard deviations from ``observables_results`` and
    ``observables``.

    Args:
        observables_results: A list of of tuples (mean, standard deviation).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.

    Returns:
        A list or a dictionary of tuples (mean, standard deviation).
    """

    if isinstance(observables, list):
        observables_eigenvalues = [None] * len(observables)
        key_value_iterator = enumerate(observables_results)
    else:
        observables_eigenvalues = {}
        key_value_iterator = zip(observables.keys(), observables_results)

    for key, value in key_value_iterator:
        if observables[key] is not None:
            observables_eigenvalues[key] = value
    return observables_eigenvalues


def _compute_std_devs(
    estimator_result: EstimatorResult,
    results_length: int,
) -> List[complex | None]:
    """
    Calculates a list of standard deviations from expectation values of observables provided. If
    there is no variance data available from a primitive, the standard deviation values will be set
    to ``None``.

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
            if variance is None or shots is None:
                std_devs.append(None)
            else:
                std_devs.append(np.sqrt(variance / shots))
        else:
            std_devs.append(0)

    return std_devs
