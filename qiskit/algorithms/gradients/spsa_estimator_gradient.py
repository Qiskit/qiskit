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

"""Gradient of Sampler with Finite difference method."""

from __future__ import annotations

from typing import Sequence
import random

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import make_spsa_base_parameter_values


class SPSAEstimatorGradient(BaseEstimatorGradient):
    """
    Gradient of Estimator with the Simultaneous Perturbation Stochastic Approximation (SPSA).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        epsilon: float = 1e-6,
        seed: int | None = None,
        **run_options,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            seed: The seed for a random perturbation vector.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting."""
        self._epsilon = epsilon
        self._seed = random.seed(seed) if seed else None

        super().__init__(estimator, **run_options)

    def _evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:
        parameters = parameters or [None for _ in range(len(circuits))]
        gradients = []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):

            base_parameter_values_list = make_spsa_base_parameter_values(circuit, self._epsilon)
            circuit_parameters = circuit.parameters
            gradient_parameter_values = np.zeros(len(circuit_parameters))

            # a parameter set for the parameter option
            parameters = parameters_ or circuit_parameters
            param_set = set(parameters)

            gradient_parameter_values = np.array(parameter_values_)
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [
                gradient_parameter_values + base_parameter_values
                for base_parameter_values in base_parameter_values_list
            ]
            gradient_circuits = [circuit] * len(gradient_parameter_values_list)
            observable_list = [observable] * len(gradient_parameter_values_list)
            job = self._estimator.run(
                gradient_circuits, observable_list, gradient_parameter_values_list, **run_options
            )
            results = job.result()
            # Combines the results and coefficients to reconstruct the gradient
            # for the original circuit parameters
            values = np.zeros(len(parameter_values_))
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                # plus
                values[i] += results.values[0] / (2 * base_parameter_values_list[0][i])
                # minus
                values[i] -= results.values[1] / (2 * base_parameter_values_list[0][i])

            gradients.append(values)
        return EstimatorGradientResult(values=gradients, metadata=run_options)
