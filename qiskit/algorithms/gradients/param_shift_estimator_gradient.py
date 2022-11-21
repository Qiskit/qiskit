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
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import (
    _make_param_shift_parameter_values,
    _make_param_shift_parameter_value_offsets,
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
        self.parameter_value_offset_cache = {}
        SUPPORTED_GATES = [
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
        super().__init__(estimator, options, supported_gates=SUPPORTED_GATES, add_virtual_var=True)

    def _preprocess(self, circuits: Sequence[QuantumCircuit]):
        """Preprocess the gradient."""
        for circuit in circuits:
            if self._gradient_circuit_cache.get(_circuit_key(circuit)) is None:
                gradient_circuit = self._assign_unique_parameters(circuit)
                self._gradient_circuit_cache[_circuit_key(circuit)] = gradient_circuit
                self.parameter_value_offset_cache[
                    _circuit_key(circuit)
                ] = _make_param_shift_parameter_value_offsets(gradient_circuit)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, coeffs_all, metadata_ = [], [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # a set of parameters to be differentiated
            parameter_set = self._make_parameter_set(circuit, parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in parameter_set]})

            # make parameter shift parameter values
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            parameter_value_offsets = self.parameter_value_offset_cache[_circuit_key(circuit)]

            gradient_parameter_values, coeffs = _make_param_shift_parameter_values(
                gradient_circuit=gradient_circuit,
                parameter_value_offsets=parameter_value_offsets,
                parameter_values=parameter_values_,
                param_set=parameter_set,
            )
            n = len(gradient_parameter_values)
            job = self._estimator.run(
                [gradient_circuit.gradient_circuit] * n,
                [observable] * n,
                gradient_parameter_values,
                **options,
            )
            jobs.append(job)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = (result.values[:n] - result.values[n:]) * np.array(coeffs_all[i])
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
