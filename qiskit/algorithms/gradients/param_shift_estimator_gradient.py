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
from qiskit.circuit import Parameter, QuantumCircuit, ParameterExpression
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import (
    _make_param_shift_parameter_values,
    _make_gradient_parameter_values,
    _make_gradient_parameters,
)


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
        self._parameter_value_offset_cache = {}
        self._SUPPORTED_GATES = [
            "x",
            "y",
            "z",
            "h",
            "rx",
            "ry",
            "rz",
            "p",
            "cx",
            "cy",
            "cz",
            "ryy",
            "rxx",
            "rzz",
            "rzx",
        ]
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the gradients of the expectation values by the parameter shift rule."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters, g_parameters)

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
    ):
        """Preprocess the gradient."""
        g_circuits, g_parameter_values, g_parameters = [], [], []
        for circuit, parameter_value_, parameters_ in zip(circuits, parameter_values, parameters):
            if not _circuit_key(circuit) in self._gradient_circuit_cache:
                self._gradient_circuit_cache[
                    _circuit_key(circuit)
                ] = self._assign_unique_parameters(circuit, supported_gates=self._SUPPORTED_GATES)
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                _make_gradient_parameter_values(circuit, gradient_circuit, parameter_value_)
            )
            g_parameters.append(_make_gradient_parameters(circuit, gradient_circuit, parameters_))
        return g_circuits, g_parameter_values, g_parameters

    def _postprocess(
        self,
        results: Sequence[EstimatorGradientResult],
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        g_parameters: Sequence[Sequence[Parameter] | None],
    ) -> EstimatorGradientResult:
        """Postprocess the gradient."""
        original_gradients, original_metadata = [], []
        for circuit, parameter_values_, parameters_, g_parameters_, gradient in zip(
            circuits, parameter_values, parameters, g_parameters, results.gradients
        ):
            parameter_set = self._make_parameter_set(circuit, parameters_)
            original_gradient = np.zeros(len(parameter_set))
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_parameter_set = self._make_parameter_set(
                gradient_circuit.gradient_circuit, g_parameters_
            )
            result_indices_ = [param for param in circuit.parameters if param in parameter_set]
            g_result_indices_ = [
                param
                for param in gradient_circuit.gradient_circuit.parameters
                if param in g_parameter_set
            ]
            g_result_indices = {param: i for i, param in enumerate(g_result_indices_)}

            for i, parameter in enumerate(result_indices_):
                for g_parameter, coeff in gradient_circuit.parameter_map[parameter]:
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values_[circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    original_gradient[i] += (
                        bound_coeff * gradient[g_result_indices[g_parameter]]
                    )
            original_gradients.append(original_gradient)
            original_metadata.append(
                [{"parameters": [p for p in circuit.parameters if p in parameter_set]}]
            )
        return EstimatorGradientResult(
            gradients=original_gradients, metadata=results.metadata, options=results.options
        )

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, metadata_ = [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # a set of parameters to be differentiated
            parameter_set = self._make_parameter_set(circuit, parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in parameter_set]})
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameter_set
            )
            n = len(param_shift_parameter_values)
            job = self._estimator.run(
                [circuit] * n,
                [observable] * n,
                param_shift_parameter_values,
                **options,
            )
            jobs.append(job)

        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc
        # compute the gradients
        gradients = []
        for result in results:
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = (result.values[:n] - result.values[n:]) / 2
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
