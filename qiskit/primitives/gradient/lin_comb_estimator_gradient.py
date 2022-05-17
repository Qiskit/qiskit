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
Gradient of expectation values with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np

from qiskit import transpile
from qiskit.circuit import (
    Gate,
    Instruction,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CYGate,
    CZGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
)
from qiskit.quantum_info import Pauli, SparsePauliOp

from ..base_estimator import BaseEstimator
from ..estimator_result import EstimatorResult
from .base_estimator_gradient import BaseEstimatorGradient

Pauli_Z = Pauli("Z")


@dataclass
class SubEstimator:
    coeff: float | ParameterExpression
    circuit: QuantumCircuit
    index: int


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """LCU estimator gradient"""

    SUPPORTED_GATES = [
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "h",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    ]

    def __init__(
        self, estimator: Type[BaseEstimator], circuit: QuantumCircuit, observable: SparsePauliOp
    ):
        self._circuit = circuit
        self._observable = observable
        self._grad, observable = self._preprocessing()
        circuits = [self._circuit]
        observables = [self._observable, observable]
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
        qr_superpos = QuantumRegister(1, "superpos")
        circuit2.add_register(qr_superpos)
        circuit2.h(qr_superpos)
        circuit2.data.insert(0, circuit2.data.pop())
        circuit2.sdg(qr_superpos)
        circuit2.data.insert(1, circuit2.data.pop())
        ret = defaultdict(list)
        for i, (inst, qregs, _) in enumerate(circuit2.data):
            if inst.is_parameterized():
                param = inst.params[0]
                for p in param.parameters:
                    gate = cls._gate_gradient(inst)
                    circuit3 = circuit2.copy()
                    # insert `gate` to i-th position
                    circuit3.append(gate, [qr_superpos[0]] + qregs, [])
                    circuit3.data.insert(i, circuit3.data.pop())
                    #
                    circuit3.h(qr_superpos)
                    ret[p].append((circuit3, param.gradient(p)))
        return ret

    def _preprocessing(self):
        grad = self._gradient_circuits(self._circuit)
        ret = {}
        observable = self._observable.expand(Pauli_Z)
        index = 1
        for param in self._circuit.parameters:
            lst = []
            for circ, coeff in grad[param]:
                lst.append(SubEstimator(coeff=coeff, circuit=circ, index=index))
                index += 1
            ret[param] = lst
        return ret, observable

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
        results = self._estimator(circ_indices, [1] * size, [parameter_value] * size, **run_options)

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

    @staticmethod
    def _gate_gradient(gate: Gate) -> Instruction:
        if isinstance(gate, RXGate):
            # theta
            return CXGate()
        if isinstance(gate, RYGate):
            # theta
            return CYGate()
        if isinstance(gate, RZGate):
            # theta
            return CZGate()
        if isinstance(gate, RXXGate):
            # theta
            cxx_circ = QuantumCircuit(3)
            cxx_circ.cx(0, 1)
            cxx_circ.cx(0, 2)
            cxx = cxx_circ.to_instruction()
            return cxx
        if isinstance(gate, RYYGate):
            # theta
            cyy_circ = QuantumCircuit(3)
            cyy_circ.cy(0, 1)
            cyy_circ.cy(0, 2)
            cyy = cyy_circ.to_instruction()
            return cyy
        if isinstance(gate, RZZGate):
            # theta
            czz_circ = QuantumCircuit(3)
            czz_circ.cz(0, 1)
            czz_circ.cz(0, 2)
            czz = czz_circ.to_instruction()
            return czz
        if isinstance(gate, RZXGate):
            # theta
            czx_circ = QuantumCircuit(3)
            czx_circ.cx(0, 2)
            czx_circ.cz(0, 1)
            czx = czx_circ.to_instruction()
            return czx
        raise TypeError(f"Unrecognized parameterized gate, {gate}")
