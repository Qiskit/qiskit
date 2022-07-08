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

from copy import copy, deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.result import QuasiDistribution

from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult


@dataclass
class SubSampler:
    coeff: float | ParameterExpression
    circuit: QuantumCircuit
    index: int

#dataclass for g_circuit作って
#base_parameter_values_listとかcoeff_listとか全部入れた方がいいかも

class ParamShiftSamplerGradientUniqueParameters:
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    def __init__(self, sampler: Type[BaseSampler], circuit: QuantumCircuit):
        self._circuit = circuit
        self._g_circuit_dict = {}
        self._rebuild_circuit_with_unique_parameters(circuit)
        return

        self._same_param_dict = defaultdict(list)
        self._grad = self._preprocessing()
        circuits = [self._circuit]
        #print('self._glad: ',self._grad.items())
        for param, lst in self._grad.items():
            #print(param, lst)
            for arg in lst:
                circuits.append(arg.circuit)

        self._sampler = sampler(circuits=circuits)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def _rebuild_circuit_with_unique_parameters(self, circuit):
        g_circuit = circuit.copy_empty_like(f'g_{circuit.name}')
        param_inst_dict = defaultdict(list)
        parameter_map = defaultdict(list)
        coef_map = {}

        for inst in circuit.data:
            new_inst = deepcopy(inst)
            qubit_idx = circuit.qubits.index(inst[1][0])
            new_inst.qubits = (g_circuit.qubits[qubit_idx],)

            # Assign new unique parameters when the instruction is a parameterized.
            if inst.operation.is_parameterized():
                parameters = inst.operation.params
                new_inst_parameters = []
                # For a gate with multiple parameters e.g. a u gate
                for parameter in parameters:
                    subs_map = {}
                    # For a gate parameter with multiple parameter variables.
                    # e.g. ry(θ) with θ = (2x + y)
                    for parameter_variable in parameter.parameters:
                        if parameter_variable in param_inst_dict:
                            new_parameter_variable = Parameter(f'g{parameter_variable.name}_{len(param_inst_dict[parameter_variable])+1})')
                        else:
                            new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
                        subs_map[parameter_variable] = new_parameter_variable
                        param_inst_dict[parameter_variable].append(inst)
                        parameter_map[parameter_variable].append(new_parameter_variable)
                        # Coefficient to calculate derivative i.e. dt/dw in df/dt * dt/dw
                        coef_map[new_parameter_variable] = parameter.gradient(parameter_variable)
                    # Substitute the parameter variables with the corresponding new parameter variables in ``subs_map``.
                    new_parameter = parameter.subs(subs_map)
                    new_inst_parameters.append(new_parameter)
                new_inst.operation.params = new_inst_parameters
            g_circuit.append(new_inst)
        print(g_circuit.draw())
        self._g_circuit_dict[circuit] = {'g_circuit': g_circuit,'parameter_map': parameter_map, 'coef_map': coef_map}

    def _prepare_base_parameter_values(self, circuit):
        # construct internal parameter values for the parameter shift
        parameter_map = self._g_circuit_dict[circuit]['parameter_map']
        g_circuit = self._g_circuit_dict[circuit]['g_circuit']
        coef_map =  self._g_circuit_dict[circuit]['coef_map']
        g_param_index_map = {}

        for i, g_param in enumerate(g_circuit.parameters):
            g_param_index_map[g_param] = i

        base_parameter_values_list = []
        base_coef_list = []

        for i, param in enumerate(circuit.parameters):
            parameter_values_list = []
            coef_list = []
            for g_param in parameter_map[param]:
                g_param_idx = g_param_index_map[g_param]
                # for + pi/2 in the parameter shift rule
                parameter_values_plus = [0] * len(g_circuit.parameters)
                parameter_values_plus[g_param_idx] += np.pi/2
                parameter_values_list.append(parameter_values_plus)
                coef_list.append(coef_map[g_param]/2)
                # for - pi/2 in the parameter shift rule
                parameter_values_minus = [0] * len(g_circuit.parameters)
                parameter_values_minus[g_param_idx] -= np.pi/2
                parameter_values_list.append(parameter_values_minus)
                coef_list.append(-1*coef_map[g_param]/2)
            base_parameter_values_list.append(parameter_values_list)
            base_coef_list.append(coef_list)

        # returnじゃなくてsub classにしようかな。
        return base_parameter_values_list, coef_list









    def _preprocessing(self):

        grad = self._gradient_circuits(self._circuit)
        #print('grad: ', grad)
        ret = {}
        index = 1
        for param in self._circuit.parameters:
            lst = []

            for circ, coeff in grad[param]:
                lst.append(SubSampler(coeff=coeff, circuit=circ, index=index))
                index += 1
            ret[param] = lst
        return ret

    def __call__(self, parameter_values: Sequence[Sequence[float]], **run_options) -> SamplerResult:
        return self._sampler([0], parameter_values, **run_options)

    def gradient(
        self,
        parameter_values: Sequence[float],
        partial: Sequence[Parameter] | None = None,
        **run_options,
    ) -> SamplerResult:
        parameters = partial or self._circuit.parameters

        param_map = {}
        for j, param in enumerate(self._circuit.parameters):
            param_map[param] = parameter_values[j]

        circ_indices = []
        param_list = []
        for param in parameters:
            circ_indices.extend([f.index for f in self._grad[param]])
            if len(self._grad[param]) == 2:
                param_idx = param_map[param]
                parameter_value_plus = parameter_values.copy()
                parameter_value_plus[param_idx] = parameter_value_plus[param_idx] + np.pi / 2
                param_list.append(parameter_value_plus)
                parameter_value_minus = parameter_values.copy()
                parameter_value_minus[param_idx] = parameter_value_minus[param_idx] - np.pi / 2
                param_list.append(parameter_value_minus)
            else:
                for i in range(len(self._grad[param])):
                    param_list.append(parameter_values)

        #print('param_list',param_list)
        #results = self._sampler(circ_indices, [parameter_value] * size, **run_options)
        results = self._sampler(circ_indices, param_list, **run_options)

        param_set = set(parameters)
        dists = [Counter() for _ in range(len(parameter_values))]
        metadata = [{} for _ in range(len(parameters))]
        i = 0
        for j, (param, lst) in enumerate(self._grad.items()):
            #print('j',j)
            if param not in param_set:
                continue
            for subest in lst:
                coeff = subest.coeff
                if isinstance(coeff, ParameterExpression):
                    local_map = {param: param_map[param] for param in coeff.parameters}
                    bound_coeff = float(coeff.bind(local_map))
                else:
                    bound_coeff = coeff
                #print(bound_coeff)
                #print('results.quasi_dists',results.quasi_dists)
                dists[j].update(
                    Counter({k: bound_coeff * v for k, v in results.quasi_dists[i].items()})
                )
                #print('dists: ',dists)
                i += 1

        return SamplerResult(
            quasi_dists=[QuasiDistribution(dist) for dist in dists], metadata=metadata
        )
