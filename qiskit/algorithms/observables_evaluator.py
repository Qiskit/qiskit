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

from typing import Tuple, Union, List, Iterable, Sequence, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import (
    CircuitSampler,
    ListOp,
    StateFn,
    OperatorBase,
    ExpectationBase,
    PauliSumOp,
)
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance

from .list_or_dict import ListOrDict
from ..primitives import Estimator, EstimatorResult
from ..quantum_info.operators.base_operator import BaseOperator


def eval_observables(
    estimator: Estimator,
    quantum_state: Sequence[QuantumCircuit],
    observables: Sequence[Union[BaseOperator, PauliSumOp]],
    threshold: float = 1e-12,
) -> ListOrDict[Tuple[complex, complex]]:
    """
    Accepts a list or a dictionary of operators and calculates their expectation values - means
    and standard deviations. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: An unparametrized quantum circuit representing a quantum state that
            expectation values are computed against.
        observables: A sequence of operators whose expectation values are to be
            calculated.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list or a dictionary of tuples (mean, standard deviation).

    Raises:
        ValueError: If a ``quantum_state`` with free parameters is provided.
    """

    if (
        isinstance(
            quantum_state, (QuantumCircuit, OperatorBase)
        )  # Statevector cannot be parametrized
        and len(quantum_state.parameters) > 0
    ):
        raise ValueError(
            "A parametrized representation of a quantum_state was provided. It is not "
            "allowed - it cannot have free parameters."
        )

    # if type(observables) != Sequence:
    #     observables = [observables]
    # if type(quantum_state) != Sequence:
    #     quantum_state = [quantum_state]
    estimator_job = estimator.run(quantum_state, observables)
    expectation_values = estimator_job.result().values

    # compute standard deviations
    std_devs = _compute_std_devs(estimator_job, len(expectation_values))

    # Discard values below threshold
    observables_means = expectation_values * (np.abs(expectation_values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = list(zip(observables_means, std_devs))

    # Return None eigenvalues for None operators if observables is a list.
    # None operators are already dropped in compute_minimum_eigenvalue if observables is a dict.

    return _prepare_result(observables_results, observables)


def _prepare_list_op(
    quantum_state: Union[
        Statevector,
        QuantumCircuit,
        OperatorBase,
    ],
    observables: ListOrDict[OperatorBase],
) -> ListOp:
    """
    Accepts a list or a dictionary of operators and converts them to a ``ListOp``.

    Args:
        quantum_state: An unparametrized quantum circuit representing a quantum state that
            expectation values are computed against.
        observables: A list or a dictionary of operators.

    Returns:
        A ``ListOp`` that includes all provided observables.
    """
    if isinstance(observables, dict):
        observables = list(observables.values())

    if not isinstance(quantum_state, StateFn):
        quantum_state = StateFn(quantum_state)

    return ListOp([StateFn(obs, is_measurement=True).compose(quantum_state) for obs in observables])


def _prepare_result(
    observables_results: List[Tuple[complex, complex]],
    observables: ListOrDict[OperatorBase],
) -> ListOrDict[Tuple[complex, complex]]:
    """
    Prepares a list or a dictionary of eigenvalues from ``observables_results`` and
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
