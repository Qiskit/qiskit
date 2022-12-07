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
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import _param_shift_preprocessing, _make_param_shift_parameter_values


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the parameter shift rule [1].

    **Reference:**
    [1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. Evaluating analytic
    gradients on quantum hardware, `DOI <https://doi.org/10.1103/PhysRevA.99.032331>`_
    """

    def __init__(self, sampler: BaseSampler, options: Options | None = None):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._gradient_circuits = {}
        super().__init__(sampler, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, result_indices_all, coeffs_all, metadata_ = [], [], [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
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

            job = self._sampler.run(
                [gradient_circuit.gradient_circuit] * n,
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
            raise AlgorithmError("Sampler job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            n = len(result.quasi_dists) // 2
            grad_dists = np.zeros((len(metadata_[i]["parameters"]), 2 ** circuits[i].num_qubits))
            for idx, coeff, dist_plus, dist_minus in zip(
                result_indices_all[i], coeffs_all[i], result.quasi_dists[:n], result.quasi_dists[n:]
            ):
                grad_dists[idx][list(dist_plus.keys())] += (
                    np.array(list(dist_plus.values())) * coeff
                )
                grad_dists[idx][list(dist_minus.keys())] -= (
                    np.array(list(dist_minus.values())) * coeff
                )

            gradient_ = []
            for grad_dist in grad_dists:
                gradient_.append(dict(enumerate(grad_dist)))
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata_, options=opt)
