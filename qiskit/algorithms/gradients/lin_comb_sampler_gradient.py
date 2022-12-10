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

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import _make_lin_comb_gradient_circuit


class LinCombSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
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

            # TODO: support measurement in different basis (Y and Z+iY)
            gradient_circuits_ = self._gradient_circuits.get(id(circuit))
            if gradient_circuits_ is None:
                gradient_circuits_ = _make_lin_comb_gradient_circuit(circuit, add_measurement=True)
                self._gradient_circuits[id(circuit)] = gradient_circuits_

            # only compute the gradients for parameters in the parameter set
            gradient_circuits, result_indices, coeffs = [], [], []
            result_idx = 0
            for i, param in enumerate(circuit.parameters):
                if param in param_set:
                    gradient_circuits.extend(
                        grad.gradient_circuit for grad in gradient_circuits_[param]
                    )
                    result_indices.extend(result_idx for _ in gradient_circuits_[param])
                    result_idx += 1
                    for grad_data in gradient_circuits_[param]:
                        coeff = grad_data.coeff
                        # if the parameter is a parameter expression, we need to substitute
                        if isinstance(coeff, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff.parameters
                            }
                            bound_coeff = float(coeff.bind(local_map))
                        else:
                            bound_coeff = coeff
                        coeffs.append(bound_coeff)

            n = len(gradient_circuits)
            job = self._sampler.run(gradient_circuits, [parameter_values_] * n, **options)
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
            n = 2 ** circuits[i].num_qubits
            grad_dists = np.zeros((len(metadata_[i]["parameters"]), n))
            for idx, coeff, dist in zip(result_indices_all[i], coeffs_all[i], result.quasi_dists):
                plus = {key: val for key, val in dist.items() if key < n}
                minus = {key - n: val for key, val in dist.items() if key >= n}
                grad_dists[idx][list(plus.keys())] += (
                    np.fromiter(plus.values(), dtype=float) * coeff
                )
                grad_dists[idx][list(minus.keys())] -= (
                    np.fromiter(minus.values(), dtype=float) * coeff
                )

            gradient_ = []
            for grad_dist in grad_dists:
                gradient_.append(dict(enumerate(grad_dist)))
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata_, options=opt)
