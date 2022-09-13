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


"""Utilities for p-VQD."""

from typing import Union, List, Callable
import logging

import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter, ParameterExpression
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import ListOp, CircuitSampler, ExpectationBase, StateFn, OperatorBase
from qiskit.opflow.gradients.circuit_gradients import ParamShift

logger = logging.getLogger(__name__)


def _is_gradient_supported(ansatz: QuantumCircuit) -> bool:
    """Check whether we can apply a simple parameter shift rule to obtain gradients."""

    # check whether the circuit can be unrolled to supported gates
    try:
        unrolled = transpile(ansatz, basis_gates=ParamShift.SUPPORTED_GATES, optimization_level=0)
    except QiskitError:
        # failed to map to supported basis
        logger.log(
            logging.INFO,
            "No gradient support: Failed to unroll to gates supported by parameter-shift.",
        )
        return False

    # check whether all parameters are unique and we do not need to apply the chain rule
    # (since it's not implemented yet)
    total_num_parameters = 0
    for circuit_instruction in unrolled.data:
        for param in circuit_instruction.operation.params:
            if isinstance(param, ParameterExpression):
                if isinstance(param, Parameter):
                    total_num_parameters += 1
                else:
                    logger.log(
                        logging.INFO,
                        "No gradient support: Circuit is only allowed to have plain parameters, "
                        "as the chain rule is not yet implemented.",
                    )
                    return False

    if total_num_parameters != ansatz.num_parameters:
        logger.log(
            logging.INFO,
            "No gradient support: Circuit is only allowed to have unique parameters, "
            "as the product rule is not yet implemented.",
        )
        return False

    return True


def _get_observable_evaluator(
    ansatz: QuantumCircuit,
    observables: Union[OperatorBase, List[OperatorBase]],
    expectation: ExpectationBase,
    sampler: CircuitSampler,
) -> Callable[[np.ndarray], Union[float, List[float]]]:
    """Get a callable to evaluate a (list of) observable(s) for given circuit parameters."""

    if isinstance(observables, list):
        observables = ListOp(observables)

    expectation_value = StateFn(observables, is_measurement=True) @ StateFn(ansatz)
    converted = expectation.convert(expectation_value)

    ansatz_parameters = ansatz.parameters

    def evaluate_observables(theta: np.ndarray) -> Union[float, List[float]]:
        """Evaluate the observables for the ansatz parameters ``theta``.

        Args:
            theta: The ansatz parameters.

        Returns:
            The observables evaluated at the ansatz parameters.
        """
        value_dict = dict(zip(ansatz_parameters, theta))
        sampled = sampler.convert(converted, params=value_dict)
        return sampled.eval()

    return evaluate_observables
