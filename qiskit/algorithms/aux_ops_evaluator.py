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
from typing import Tuple, Union, List

import numpy as np

from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import ListOrDict
from qiskit.opflow import (
    CircuitSampler,
    ListOp,
    StateFn,
    CircuitStateFn,
    OperatorBase,
    ExpectationBase,
)
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance


def _eval_observabless(
    quantum_instance: Union[QuantumInstance, BaseBackend, Backend],
    ansatz: OperatorBase,
    parameters: np.ndarray,
    observables: ListOrDict[OperatorBase],
    expectation: ExpectationBase,
    threshold: float = 1e-12,
) -> ListOrDict[Tuple[complex, complex]]:
    """
    Accepts a list or a dictionary of operators and calculates their expectation values - means
    and standard deviations. They are calculated with respect to a parametrized ansatz provided.
    An ansatz is bound using the list of parameter values provided. A user can optionally provide a
    threshold value which filters mean values falling below the threshold.

    Args:
        quantum_instance: A quantum instance used for calculations.
        ansatz: A parametrized ansatz that expectation values are computed against.
        parameters: An ordered list of parameters values used to bind an ansatz. They must follow an
            order of parameters in an ansatz (ansatz.ordered_parameters).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        expectation: An instance of ExpectationBase which defines a method for calculating
            expectation values.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list or a dictionary of tuples (mean, standard deviation).

    """

    # Create new CircuitSampler to avoid breaking existing one's caches.
    sampler = CircuitSampler(quantum_instance)

    list_op = _prepare_list_op(observables)

    param_dict = dict(zip(ansatz.ordered_parameters, parameters))

    bound_ansatz = ansatz.bind_parameters(param_dict)

    observables_expect = expectation.convert(
        StateFn(list_op, is_measurement=True).compose(CircuitStateFn(bound_ansatz))
    )
    observables_expect_sampled = sampler.convert(observables_expect)

    # compute means
    values = np.real(observables_expect_sampled.eval())

    # compute standard deviations
    std_devs = _compute_std_devs(
        observables_expect_sampled, observables, expectation, quantum_instance
    )

    # Discard values below threshold
    observables_means = values * (np.abs(values) > threshold)
    # zip means and standard deviations into tuples
    observables_results = zip(observables_means, std_devs)

    # Return None eigenvalues for None operators if observables is a list.
    # None operators are already dropped in compute_minimum_eigenvalue if observables is a dict.

    return _prepare_result(observables_results, observables)


def _prepare_list_op(observables: ListOrDict[OperatorBase]) -> ListOp[OperatorBase]:
    if isinstance(observables, dict):
        return ListOp(list(observables.values()))

    return ListOp(observables)


def _prepare_result(
    observables_results,
    observables: ListOrDict[OperatorBase],
) -> ListOrDict[Tuple[complex, complex]]:
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
    observables_expect_sampled,
    observables: ListOrDict[OperatorBase],
    expectation: ExpectationBase,
    quantum_instance: Union[QuantumInstance, BaseBackend, Backend],
) -> List[complex]:
    variances = np.real(expectation.compute_variance(observables_expect_sampled))
    if not isinstance(variances, np.ndarray) and variances == 0.0:
        # when `variances` is a single value equal to 0., our expectation value is exact and we
        # manually ensure the variances to be a list of the correct length
        variances = np.zeros(len(observables), dtype=float)
    std_devs = np.sqrt(variances / quantum_instance.run_config.shots)
    return std_devs
