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

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow import (
    CircuitSampler,
    ListOp,
    StateFn,
    OperatorBase,
    ExpectationBase,
)
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_func

from .list_or_dict import ListOrDict


@deprecate_func(
    additional_msg=(
        "Instead, use the function "
        "``qiskit.algorithms.observables_evaluator.estimate_observables``. See "
        "https://qisk.it/algo_migration for a migration guide."
    ),
    since="0.24.0",
    package_name="qiskit-terra",
)
def eval_observables(
    quantum_instance: QuantumInstance | Backend,
    quantum_state: Statevector | QuantumCircuit | OperatorBase,
    observables: ListOrDict[OperatorBase],
    expectation: ExpectationBase,
    threshold: float = 1e-12,
) -> ListOrDict[tuple[complex, complex]]:
    """
    Deprecated: Accepts a list or a dictionary of operators and calculates
    their expectation values - means
    and standard deviations. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    This function has been superseded by the
    :func:`qiskit.algorithms.observables_evaluator.eval_observables` function.
    It will be deprecated in a future release and subsequently
    removed after that.

    Args:
        quantum_instance: A quantum instance used for calculations.
        quantum_state: An unparametrized quantum circuit representing a quantum state that
            expectation values are computed against.
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        expectation: An instance of ExpectationBase which defines a method for calculating
            expectation values.
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

    # Create new CircuitSampler to avoid breaking existing one's caches.
    sampler = CircuitSampler(quantum_instance)

    list_op = _prepare_list_op(quantum_state, observables)
    observables_expect = expectation.convert(list_op)
    observables_expect_sampled = sampler.convert(observables_expect)

    # compute means
    values = np.real(observables_expect_sampled.eval())

    # compute standard deviations
    # We use sampler.quantum_instance to take care of case in which quantum_instance is Backend
    std_devs = _compute_std_devs(
        observables_expect_sampled, observables, expectation, sampler.quantum_instance
    )

    # Discard values below threshold
    observables_means = values * (np.abs(values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = list(zip(observables_means, std_devs))

    # Return None eigenvalues for None operators if observables is a list.
    # None operators are already dropped in compute_minimum_eigenvalue if observables is a dict.

    return _prepare_result(observables_results, observables)


def _prepare_list_op(
    quantum_state: Statevector | QuantumCircuit | OperatorBase,
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
    observables_results: list[tuple[complex, complex]],
    observables: ListOrDict[OperatorBase],
) -> ListOrDict[tuple[complex, complex]]:
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
        observables_eigenvalues: ListOrDict[tuple[complex, complex]] = [None] * len(observables)
        key_value_iterator = enumerate(observables_results)
    else:
        observables_eigenvalues = {}
        key_value_iterator = zip(observables.keys(), observables_results)
    for key, value in key_value_iterator:
        if observables[key] is not None:
            observables_eigenvalues[key] = value
    return observables_eigenvalues


def _compute_std_devs(
    observables_expect_sampled: OperatorBase,
    observables: ListOrDict[OperatorBase],
    expectation: ExpectationBase,
    quantum_instance: QuantumInstance | Backend,
) -> list[complex]:
    """
    Calculates a list of standard deviations from expectation values of observables provided.

    Args:
        observables_expect_sampled: Expected values of observables.
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        expectation: An instance of ExpectationBase which defines a method for calculating
            expectation values.
        quantum_instance: A quantum instance used for calculations.

    Returns:
        A list of standard deviations.
    """
    variances = np.real(expectation.compute_variance(observables_expect_sampled))
    if not isinstance(variances, np.ndarray) and variances == 0.0:
        # when `variances` is a single value equal to 0., our expectation value is exact and we
        # manually ensure the variances to be a list of the correct length
        variances = np.zeros(len(observables), dtype=float)
    # TODO: this will crash if quantum_instance is a backend
    std_devs = np.sqrt(variances / quantum_instance.run_config.shots)
    return std_devs
