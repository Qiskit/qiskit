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
"""
Gradient of probabilities with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_job import EstimatorGradientJob
from .utils import make_lin_comb_gradient_circuit

Pauli_Z = Pauli("Z")


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values.
    This method employs a linear combination of unitaries,
    see e.g. https://arxiv.org/pdf/1811.11184.pdf
    """

    def __init__(self, estimator: BaseEstimator):
        """
        Args:
            estimator: The estimator used to compute the gradients.
        """
        self._gradient_circuit_data_dict = {}
        super().__init__(estimator)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientJob:
        partial = partial or [[] for _ in range(len(circuits))]
        gradients = []
        status = []

        for circuit, observable, parameter_values_, partial_ in zip(
            circuits, observables, parameter_values, partial
        ):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_index = index
            else:
                # if circuit is not passed in the constructor.
                circuit_index = len(self._circuits)
                self._circuit_ids[id(circuit)] = circuit_index
                self._gradient_circuit_data_dict[circuit_index] = make_lin_comb_gradient_circuit(
                    circuit
                )
                self._circuits.append(circuit)
            # Add an observable for the auxiliary qubit
            observable_ = observable.expand(Pauli_Z)

            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}

            # a parameter set for the partial option
            parameters = partial_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = defaultdict(list)
            gradient_circuits = []
            # gradient circuit indices and result indices
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                if not param in param_set:
                    continue
                for grad in gradient_circuit_data[param]:
                    gradient_circuits.append(grad.gradient_circuit)
                    result_index_map[param].append(result_index)
                    result_index += 1
            gradient_parameter_values_list = [
                parameter_values_ for i in range(len(gradient_circuits))
            ]
            observable_list = [observable_] * len(gradient_circuits)

            job = self._estimator.run(
                gradient_circuits, observable_list, gradient_parameter_values_list
            )
            results = job.result()

            values = np.zeros(len(circuit_parameters))
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for j, grad in enumerate(gradient_circuit_data[param]):
                    coeff = grad.coeff
                    result_index = result_index_map[param][j]
                    # if coeff has parameters, substitute them with the given parameter values
                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: parameter_value_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    values[i] += bound_coeff * results.values[result_index_map[param][j]]

            gradients.append(EstimatorResult(values, metadata=run_options))
            status.append(job.status())
        return EstimatorGradientJob(results=gradients, status=status)
