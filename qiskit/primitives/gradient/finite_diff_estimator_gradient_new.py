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

from collections import Counter, Iterable, defaultdict
from typing import Sequence, Type

import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import QuasiDistribution
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..base_estimator import BaseEstimator
from ..factories import SamplerFromCircuits
from ..sampler_result import SamplerResult
from .base_sampler_gradient import BaseSamplerGradient
from .estimator_gradient_result import EstimatorGradientResult
from ..utils import init_circuit
from .utils import make_base_parameter_values_fin_diff


class FiniteDiffEstimatorGradientNew:
    """
    Gradient of Sampler with Finite difference method.
    """

    def __init__(
        self,
        estimator: Type[BaseEstimator],
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: SparsePauliOp,
        epsilon: float = 1e-6,
    ):
        """
        TODO: Write
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._circuits = circuits
        self._epsilon = epsilon
        self._observables = observables

        self._base_parameter_values_dict = {}
        for i, circuit in enumerate(self._circuits):
            self._base_parameter_values_dict[i] = make_base_parameter_values_fin_diff(
                circuit, epsilon
            )

        # TODO: this should be modified to add new gradient circuits after new primitives change
        # call rebuild_circuits_with_unique_parameters when first time calculating the gradient for a circuit
        self._estimator = estimator(circuits=circuits, observables=observables)
        print(self._estimator.circuits[0])
        print(self._estimator.observables[0])
        print(self._estimator)


    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:

        partial = partial or [[] for _ in range(len(circuits))]
        gradients = []
        for circuit_index, observable, parameter_values_, partial_ in zip(circuits, observables, parameter_values, partial):

            circuit_parameters = self._circuits[circuit_index].parameters
            base_parameter_values_list = []
            gradient_parameter_values = np.zeros(len(circuit_parameters))
            print(gradient_parameter_values)

            # a parameter set for the partial option
            parameters = partial_ or circuit_parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = {}
            # parameter_value_map = {}
            # bring the base parameter values for parameters only in the partial parameter set.
            for i, param in enumerate(circuit_parameters):
                gradient_parameter_values[i] = parameter_values_[i]
                if param in param_set:
                    base_parameter_values_list.append(
                        self._base_parameter_values_dict[circuit_index][i * 2]
                    )
                    base_parameter_values_list.append(
                        self._base_parameter_values_dict[circuit_index][i * 2 + 1]
                    )
                    result_index_map[param] = result_index
                    result_index += 1
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [
                gradient_parameter_values + base_parameter_values
                for base_parameter_values in base_parameter_values_list
            ]
            circuit_indices = [circuit_index] * len(gradient_parameter_values_list)
            observables_indices = [observable] * len(gradient_parameter_values_list)

            print(self._estimator.circuits[0])
            print(observables_indices )
            print(gradient_parameter_values_list)



            results = self._estimator.__call__(circuit_indices, observables_indices,gradient_parameter_values_list)

            # Combines the results and coefficients to reconstruct the gradient for the original circuit parameters
            print(results)
            values = np.zeros(len(parameter_values_))
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                # plus
                values[i] += results.values[result_index_map[param] *2] /(2*self._epsilon)
                # minus
                values[i] -= results.values[result_index_map[param] *2 + 1] /(2*self._epsilon)

            gradients.append(values)
        return EstimatorGradientResult(values=gradients, metadata=[{}] * len(gradients))
