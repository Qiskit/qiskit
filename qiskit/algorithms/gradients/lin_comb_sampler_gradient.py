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

from collections import Counter, defaultdict
from typing import Sequence

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import make_lin_comb_gradient_circuit


class LinCombSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability.
    This method employs a linear combination of unitaries,
    see e.g. https://arxiv.org/pdf/1811.11184.pdf
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

    def _evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> SamplerGradientResult:
        parameters = parameters or [None for _ in range(len(circuits))]
        gradients = []

        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_index = index
            else:
                # if circuit is not passed in the constructor.
                circuit_index = len(self._circuits)
                self._circuit_ids[id(circuit)] = circuit_index
                self._gradient_circuit_data_dict[circuit_index] = make_lin_comb_gradient_circuit(
                    circuit, add_measurement=True
                )
                self._circuits.append(circuit)

            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}

            # a parameter set for the parameter option
            parameters = parameters_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = defaultdict(list)
            gradient_circuit = []
            # gradient circuit indices and result indices
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                if not param in param_set:
                    continue
                for grad in gradient_circuit_data[param]:
                    gradient_circuit.append(grad.gradient_circuit)
                    result_index_map[param].append(result_index)
                    result_index += 1
            gradient_parameter_values_list = [parameter_values_] * len(gradient_circuit)

            job = self._sampler.run(gradient_circuit, gradient_parameter_values_list, **run_options)
            results = job.result()

            param_set = set(parameters)
            dists = [Counter() for _ in range(len(parameter_values_))]
            num_bitstrings = 2 ** self._circuits[circuit_index].num_qubits
            i = 0
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for j, grad in enumerate(gradient_circuit_data[param]):
                    coeff = grad.coeff
                    result_index = result_index_map[param][j]
                    # if coeff has parameters, substitute them with the given parameter values
                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: parameter_value_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    for k, v in results.quasi_dists[result_index].items():
                        sign, k2 = divmod(k, num_bitstrings)
                        dists[i][k2] += (-1) ** sign * bound_coeff * v

            gradients.append([QuasiDistribution(dist) for dist in dists])
        return SamplerGradientResult(quasi_dists=gradients, metadata=run_options)
