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

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import _make_param_shift_parameter_values, _param_shift_preprocessing


class ParamShiftEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values by the parameter shift rule"""

    def __init__(self, estimator: BaseEstimator, options: Options | None = None):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._gradient_circuits = {}
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, result_indices_all, coeffs_all, metadata_ = [], [], [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in param_set]})

            if self._gradient_circuits.get(id(circuit)):
                gradient_circuit, base_parameter_values_all = self._gradient_circuits[id(circuit)]
            else:
                gradient_circuit, base_parameter_values_all = _param_shift_preprocessing(circuit)
                self._gradient_circuits[id(circuit)] = (
                    gradient_circuit,
                    base_parameter_values_all,
                )

            (
                gradient_parameter_values_plus,
                gradient_parameter_values_minus,
                result_indices,
                coeffs,
            ) = _make_param_shift_parameter_values(
                gradient_circuit_data=gradient_circuit,
                base_parameter_values=base_parameter_values_all,
                parameter_values=parameter_values_,
                param_set=param_set,
            )
            n = 2 * len(gradient_parameter_values_plus)
            job = self._estimator.run(
                [gradient_circuit.gradient_circuit] * n,
                [observable] * n,
                gradient_parameter_values_plus + gradient_parameter_values_minus,
                **options,
            )
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = result.values[:n] - result.values[n:]
            values = np.zeros(len(metadata_[i]["parameters"]))
            for grad_, idx, coeff in zip(gradient_, result_indices_all[i], coeffs_all[i]):
                values[idx] += coeff * grad_
            gradients.append(values)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
