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

from collections import defaultdict
from typing import Sequence

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, QuantumCircuit

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import _make_param_shift_parameter_values


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability by the parameter shift rule."""

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

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameter_sets = self._preprocess(
            circuits, parameter_values, parameter_sets, self.SUPPORTED_GATES
        )
        results = self._run_unique(g_circuits, g_parameter_values, g_parameter_sets, **options)
        return self._postprocess(results, circuits, parameter_values, parameter_sets)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, metadata = [], []
        for circuit, parameter_values_, parameter_set in zip(
            circuits, parameter_values, parameter_sets
        ):
            metadata.append({"parameters": [p for p in circuit.parameters if p in parameter_set]})
            # Make parameter values for the parameter shift rule.
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameter_set
            )
            n = len(param_shift_parameter_values)
            job = self._sampler.run(
                [circuit] * n,
                param_shift_parameter_values,
                **options,
            )
            jobs.append(job)

        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        # for i, result in enumerate(results):
        for result in results:
            n = len(result.quasi_dists) // 2
            gradient = []
            for dist_plus, dist_minus in zip(result.quasi_dists[:n], result.quasi_dists[n:]):
                grad_dist = defaultdict(float)
                for key, val in dist_plus.items():
                    grad_dist[key] += val / 2
                for key, val in dist_minus.items():
                    grad_dist[key] -= val / 2
                gradient.append(dict(grad_dist))
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
