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

from collections import Counter
from typing import Sequence

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import make_lin_comb_gradient_circuit


class LinCombSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    def __init__(self, sampler: BaseSampler, **run_options):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """

        self._gradient_circuit_data_dict = {}
        super().__init__(sampler, **run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **run_options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, result_indices_all, coeffs_all = [], [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)
            # TODO: support measurement in different basis (Y and Z+iY)
            gradient_circuit_data = self._gradient_circuit_data_dict.get(id(circuit))
            if gradient_circuit_data is None:
                gradient_circuit_data = make_lin_comb_gradient_circuit(
                    circuit, add_measurement=True
                )
                self._gradient_circuit_data_dict[id(circuit)] = gradient_circuit_data

            # only compute the gradients for parameters in the parameter set
            gradient_circuits = []
            result_indices = []
            coeffs = []
            for i, param in enumerate(circuit.parameters):
                if param in param_set:
                    gradient_circuits.extend(
                        grad_data.gradient_circuit for grad_data in gradient_circuit_data[param]
                    )
                    result_indices.extend(i for _ in gradient_circuit_data[param])
                    for grad_data in gradient_circuit_data[param]:
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
            job = self._sampler.run(gradient_circuits, [parameter_values_ for _ in range(n)])
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        results = [job.result() for job in jobs]
        gradients, metadata_ = [], []
        for i, result in enumerate(results):
            dists = [Counter() for _ in range(circuits[i].num_parameters)]
            num_bitstrings = 2 ** circuits[i].num_qubits
            for grad_quasi_, idx, coeff in zip(
                result.quasi_dists, result_indices_all[i], coeffs_all[i]
            ):
                for k_, v in grad_quasi_.items():
                    sign, k = divmod(k_, num_bitstrings)
                    dists[idx][k] += (-1) ** sign * coeff * v
            gradients.append([QuasiDistribution(dist) for dist in dists])

        # TODO: include primitive's run_options as well
        return SamplerGradientResult(
            gradients=gradients, metadata=metadata_, run_options=run_options
        )
