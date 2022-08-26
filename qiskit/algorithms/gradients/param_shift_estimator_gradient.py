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
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from copy import copy
from typing import Sequence

import numpy as np

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult

from .utils import (
    make_param_shift_gradient_circuit_data,
    make_param_shift_base_parameter_values,
)


class ParamShiftEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values by the parameter shift rule"""

    def __init__(self, estimator: BaseEstimator, **run_options):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._gradient_circuit_data_dict = {}
        self._base_parameter_values_dict = {}
        super().__init__(estimator, **run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        # if parameters is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]

        jobs, result_indices_all, coeffs_all = [], [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)

            gradient_circuit_data = self._gradient_circuit_data_dict.get(id(circuit))
            base_parameter_values_all = self._base_parameter_values_dict.get(id(circuit))
            if gradient_circuit_data is None and base_parameter_values_all is None:
                gradient_circuit_data = make_param_shift_gradient_circuit_data(circuit)
                self._gradient_circuit_data_dict[id(circuit)] = gradient_circuit_data
                base_parameter_values_all = make_param_shift_base_parameter_values(
                    gradient_circuit_data
                )
                self._base_parameter_values_dict[id(circuit)] = base_parameter_values_all

            plus_offsets, minus_offsets = [], []
            gradient_circuit = gradient_circuit_data.gradient_circuit
            gradient_parameter_values = np.zeros(
                len(gradient_circuit_data.gradient_circuit.parameters)
            )

            # only compute the gradients for parameters in the parameter set
            result_indices = []
            coeffs = []
            for i, param in enumerate(circuit.parameters):
                g_params = gradient_circuit_data.gradient_parameter_map[param]
                indices = [gradient_circuit.parameters.data.index(g_param) for g_param in g_params]
                gradient_parameter_values[indices] = parameter_values_[i]
                if param in param_set:
                    plus_offsets.extend(base_parameter_values_all[idx] for idx in indices)
                    minus_offsets.extend(
                        base_parameter_values_all[idx + len(gradient_circuit.parameters)]
                        for idx in indices
                    )
                    result_indices.extend(i for _ in range(len(indices)))
                    for g_param in g_params:
                        coeff = gradient_circuit_data.coeff_map[g_param]
                        # if coeff has parameters, we need to substitute
                        if isinstance(coeff, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff.parameters
                            }
                            bound_coeff = float(coeff.bind(local_map))
                        else:
                            bound_coeff = coeff
                        coeffs.append(bound_coeff)

            # add the base parameter values to the parameter values
            gradient_parameter_values_plus = [
                gradient_parameter_values + plus_offset for plus_offset in plus_offsets
            ]
            gradient_parameter_values_minus = [
                gradient_parameter_values + minus_offset for minus_offset in minus_offsets
            ]
            n = 2 * len(gradient_parameter_values_plus)

            job = self._estimator.run(
                [gradient_circuit] * n,
                [observable] * n,
                gradient_parameter_values_plus + gradient_parameter_values_minus,
                **run_options,
            )
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        results = [job.result() for job in jobs]
        gradients, metadata_ = [], []
        for i, result in enumerate(results):
            d = copy(run_options)
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = (result.values[:n] - result.values[n:]) / 2
            values = np.zeros(len(circuits[i].parameters))
            for grad_, idx, coeff in zip(gradient_, result_indices_all[i], coeffs_all[i]):
                values[idx] += coeff * grad_
            gradients.append(values)
            d['gradient_variance'] = np.var(gradient_)
            metadata_.append(result.metadata)

        return EstimatorGradientResult(values=gradients, metadata=metadata_)
