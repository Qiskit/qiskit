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

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import _make_param_shift_parameter_values


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
            circuits, parameter_values, parameters, self._SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

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
