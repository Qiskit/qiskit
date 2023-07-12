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

from collections.abc import Sequence
from typing import Literal

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult

from ...exceptions import AlgorithmError


class FiniteDiffEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by finite difference method [1].

    **Reference:**
    [1] `Finite difference method <https://en.wikipedia.org/wiki/Finite_difference_method>`_
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        epsilon: float,
        options: Options | None = None,
        *,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            method: The computation method of the gradients.

                    - ``central`` computes :math:`\frac{f(x+e)-f(x-e)}{2e}`,
                    - ``forward`` computes :math:`\frac{f(x+e) - f(x)}{e}`,
                    - ``backward`` computes :math:`\frac{f(x)-f(x-e)}{e}`

                where :math:`e` is epsilon.

        Raises:
            ValueError: If ``epsilon`` is not positive.
            TypeError: If ``method`` is invalid.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        if method not in ("central", "forward", "backward"):
            raise TypeError(
                f"The argument method should be central, forward, or backward: {method} is given."
            )
        self._method = method
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        job_circuits, job_observables, job_param_values, metadata = [], [], [], []
        all_n = []

        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})

            # Combine inputs into a single job to reduce overhead.
            offset = np.identity(circuit.num_parameters)[indices, :]
            if self._method == "central":
                plus = parameter_values_ + self._epsilon * offset
                minus = parameter_values_ - self._epsilon * offset
                n = 2 * len(indices)
                job_circuits.extend([circuit] * n)
                job_observables.extend([observable] * n)
                job_param_values.extend(plus.tolist() + minus.tolist())
                all_n.append(n)
            elif self._method == "forward":
                plus = parameter_values_ + self._epsilon * offset
                n = len(indices) + 1
                job_circuits.extend([circuit] * n)
                job_observables.extend([observable] * n)
                job_param_values.extend([parameter_values_] + plus.tolist())
                all_n.append(n)
            elif self._method == "backward":
                minus = parameter_values_ - self._epsilon * offset
                n = len(indices) + 1
                job_circuits.extend([circuit] * n)
                job_observables.extend([observable] * n)
                job_param_values.extend([parameter_values_] + minus.tolist())
                all_n.append(n)

        # Run the single job with all circuits.
        job = self._estimator.run(job_circuits, job_observables, job_param_values, **options)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients
        gradients = []
        partial_sum_n = 0
        for n in all_n:
            if self._method == "central":
                result = results.values[partial_sum_n : partial_sum_n + n]
                gradient = (result[: n // 2] - result[n // 2 :]) / (2 * self._epsilon)
            elif self._method == "forward":
                result = results.values[partial_sum_n : partial_sum_n + n]
                gradient = (result[1:] - result[0]) / self._epsilon
            elif self._method == "backward":
                result = results.values[partial_sum_n : partial_sum_n + n]
                gradient = (result[0] - result[1:]) / self._epsilon
            partial_sum_n += n
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata, options=opt)
