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


class ParamShiftSamplerGradientOrigin:
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    def __init__(self, sampler: Type[BaseSampler], circuit: QuantumCircuit):
        self._circuit = circuit
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

    def _gradient_circuits(self, circuit: QuantumCircuit):
        circuit2 = transpile(circuit, basis_gates=self.SUPPORTED_GATES, optimization_level=0)
        print(circuit2.draw())
        ret = defaultdict(list)
        #self._same_param_dict = defaultdict(list)
        for inst in circuit2.data:

            print('inst: ',inst)
            if inst[0].is_parameterized():
                param = inst[0].params[0]
                print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                print('param: ', type(param),param, id(param))
                for p in param.parameters:
                    print('p: ',type(p),p, id(p))

                    # if same_param_dict[p]:
                        #print('p: ',type(p),p)
                        # TODO: Need to wait for an appropriate way to update parameters of
                        #   a particular instruction.
                        #   See https://github.com/Qiskit/qiskit-terra/issues/7894
                    #print('aaaaa',self._same_param_dict[p])
                    if self._same_param_dict[p]:
                        if len(self._same_param_dict[p]) == 1:
                            param_ = self._same_param_dict[p][0][0].params[0]

                            #print('same_param_dict[p][0][0].params[0]', type(self._same_param_dict[p][0][0].params[0]), self._same_param_dict[p][0][0].params[0], id(self._same_param_dict[p][0][0].params[0]))

                            self._same_param_dict[p][0][0].params[0] = param_ + np.pi / 2
                            #print(self._same_param_dict[p][0][0].params[0], id(self._same_param_dict[p][0][0].params[0]))
                            ret[p].append((circuit2.copy(), param_.gradient(p) / 2))
#                            print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
#                            print('param: ', type(param),param, id(param))
                            #print(circuit2.draw())
                            self._same_param_dict[p][0][0].params[0] = param_ - np.pi / 2
                            ret[p].append((circuit2.copy(), -param_.gradient(p) / 2))
                            #print(circuit2.draw())
#                            print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
#                            print('param: ', type(param),param, id(param))
                            self._same_param_dict[p][0][0].params[0] = param_
 #                           print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
  #                          print('param: ', type(param),param, id(param))
                        #print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                        inst[0].params[0] = param + np.pi / 2
                       # print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                        ret[p].append((circuit2.copy(), param.gradient(p) / 2))
                        #print(circuit2.draw())
                        inst[0].params[0] = param - np.pi / 2
                        ret[p].append((circuit2.copy(), -param.gradient(p) / 2))
                        #print(circuit2.draw())
                        inst[0].params[0] = param

                    self._same_param_dict[p].append(inst)

        #print('---------')
        for p, lst in self._same_param_dict.items():
            #print(p, lst)
            if len(lst) == 1:
                param_ = self._same_param_dict[p][0][0].params[0]

                #print('same_param_dict[p][0][0].params[0]', type(self._same_param_dict[p][0][0].params[0]), self._same_param_dict[p][0][0].params[0], id(self._same_param_dict[p][0][0].params[0]))

                self._same_param_dict[p][0][0].params[0] = param_ + np.pi / 2
                #print(self._same_param_dict[p][0][0].params[0], id(self._same_param_dict[p][0][0].params[0]))
                ret[p].append((circuit2, param_.gradient(p) / 2))
#                            print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
#                            print('param: ', type(param),param, id(param))
                #print(circuit2.draw())
                self._same_param_dict[p][0][0].params[0] = param_ - np.pi / 2
                ret[p].append((circuit2, -param_.gradient(p) / 2))
                #print(circuit2.draw())
#                            print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
#                            print('param: ', type(param),param, id(param))
                self._same_param_dict[p][0][0].params[0] = param_

        #print('aaaaaaaaa',ret)


                    # inst[0].params[0] = param + np.pi / 2
                    # ret[p].append((circuit2.copy(), param.gradient(p) / 2))
                    # print('aaaaaaaaaaaaaaaa')
                    # print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                    # print('param: ', type(param),param, id(param))

                    # inst[0].params[0] = param - np.pi / 2
                    # ret[p].append((circuit2.copy(), -param.gradient(p) / 2))
                    # print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                    # print('param: ', type(param),param, id(param))
                    # inst[0].params[0] = param
                    # print('inst[0].params[0]: ',type(inst[0].params[0]), inst[0].params[0], id(inst[0].params[0]))
                    # print('param: ', type(param),param, id(param))
                    # same_param_dict[p] = 1

        # ------------
        #print('num_same_param_dict: ', num_same_param_dict)
        print('ret', ret)
        #print('self._same_param_dict', self._same_param_dict)
        # grad_circ_index = {}
        # idx = 1
        # grad_circuits = []
        # grad_circuits.append(circuit2)
        # for param, num in num_same_param_dict.items():
        #     if 1 < num:
        #         grad_circ_index[param] = [idx, idx + num* 2]
        #         idx = idx + num* 2

        # print("grad_circ_index: ",grad_circ_index)

        # ------------

        return ret

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
