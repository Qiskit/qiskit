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
Gradient of expectation values with parameter shift
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..base_estimator import BaseEstimator
from ..estimator_result import EstimatorResult
from .base_estimator_gradient import BaseEstimatorGradient


@dataclass
class SubEstimator:
    coeff: float | ParameterExpression
    circuit: QuantumCircuit
    observable: SparsePauliOp
    index: int


class ParamShiftEstimatorGradient(BaseEstimatorGradient):
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    def __init__(
        self, estimator: Type[BaseEstimator], circuit: QuantumCircuit, observable: SparsePauliOp
    ):
        self._circuit = circuit
        self._observable = observable
        self._grad = self._preprocessing()
        circuits = [self._circuit]
        observables = [self._observable]
        for param, lst in self._grad.items():
            for arg in lst:
                circuits.append(arg.circuit)
        super().__init__(estimator(circuits=circuits, observables=observables))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    @classmethod
    def _gradient_circuits(cls, circuit: QuantumCircuit):
        circuit2 = transpile(circuit, basis_gates=cls.SUPPORTED_GATES, optimization_level=0)
        ret = defaultdict(list)
        for inst in circuit2.data:
            if inst[0].is_parameterized():
                param = inst[0].params[0]
                for p in param.parameters:
                    # TODO: Need to wait for an appropriate way to update parameters of
                    #   a particular instruction.
                    #   See https://github.com/Qiskit/qiskit-terra/issues/7894
                    inst[0].params[0] = param + np.pi / 2
                    ret[p].append((circuit2.copy(), param.gradient(p) / 2))
                    inst[0].params[0] = param - np.pi / 2
                    ret[p].append((circuit2.copy(), -param.gradient(p) / 2))
                    inst[0].params[0] = param
        return ret

    def _preprocessing(self):
        grad = self._gradient_circuits(self._circuit)
        ret = {}
        index = 1
        for param in self._circuit.parameters:
            lst = []
            for circ, coeff in grad[param]:
                lst.append(
                    SubEstimator(
                        coeff=coeff, circuit=circ, observable=self._observable, index=index
                    )
                )
                index += 1
            ret[param] = lst
        return ret

    def __call__(
        self, parameter_values: Sequence[Sequence[float]], **run_options
    ) -> EstimatorResult:
        return self._estimator([0], [0], parameter_values, **run_options)

    def gradient(
        self,
        parameter_value: Sequence[float],
        partial: Sequence[Parameter] | None = None,
        **run_options,
    ) -> EstimatorResult:
        parameters = partial or self._circuit.parameters

        param_map = {}
        for j, param in enumerate(self._circuit.parameters):
            param_map[param] = parameter_value[j]

        circ_indices = []
        for param in parameters:
            circ_indices.extend([f.index for f in self._grad[param]])
        size = len(circ_indices)
        results = self._estimator(circ_indices, [0] * size, [parameter_value] * size, **run_options)

        param_set = set(parameters)
        values = np.zeros(len(parameter_value))
        metadata = [{} for _ in range(len(parameters))]
        i = 0
        for j, (param, lst) in enumerate(self._grad.items()):
            if param not in param_set:
                continue
            for subest in lst:
                coeff = subest.coeff
                if isinstance(coeff, ParameterExpression):
                    local_map = {param: param_map[param] for param in coeff.parameters}
                    bound_coeff = coeff.bind(local_map)
                else:
                    bound_coeff = coeff
                values[j] += bound_coeff * results.values[i]
                i += 1

        return EstimatorResult(values=values, metadata=metadata)
