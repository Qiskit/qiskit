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
from dataclasses import dataclass
from typing import Sequence, Type

from qiskit import transpile
from qiskit.circuit import (
    ClassicalRegister,
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
from qiskit.result import QuasiDistribution

from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult


@dataclass
class SubEstimator:
    coeff: float | ParameterExpression
    circuit: QuantumCircuit
    index: int


class LinCombSamplerGradient:
    """LCU sampler gradient"""

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

    def __init__(self, sampler: Type[BaseSampler], circuit: QuantumCircuit):
        self._circuit = circuit
        self._grad = self._preprocessing()
        circuits = [self._circuit]
        for param, lst in self._grad.items():
            for arg in lst:
                circuits.append(arg.circuit)
        self._sampler = sampler(circuits=circuits)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    @classmethod
    def _gradient_circuits(cls, circuit: QuantumCircuit):
        circuit2 = transpile(circuit, basis_gates=cls.SUPPORTED_GATES, optimization_level=0)
        qr_superpos = QuantumRegister(1, "superpos")
        cr_superpos = ClassicalRegister(1, "superpos")
        circuit2.add_register(qr_superpos)
        circuit2.add_bits(cr_superpos)
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
                    circuit3.measure(qr_superpos, cr_superpos)
                    ret[p].append((circuit3, param.gradient(p)))
        return ret

    def _preprocessing(self):
        grad = self._gradient_circuits(self._circuit)
        ret = {}
        index = 1
        for param in self._circuit.parameters:
            lst = []
            for circ, coeff in grad[param]:
                lst.append(SubEstimator(coeff=coeff, circuit=circ, index=index))
                index += 1
            ret[param] = lst
        return ret

    def __call__(self, parameter_values: Sequence[Sequence[float]], **run_options) -> SamplerResult:
        return self._sampler([0], parameter_values, **run_options)

    def gradient(
        self,
        parameter_value: Sequence[float],
        partial: Sequence[Parameter] | None = None,
        **run_options,
    ) -> SamplerResult:
        parameters = partial or self._circuit.parameters

        param_map = {}
        for j, param in enumerate(self._circuit.parameters):
            param_map[param] = parameter_value[j]

        circ_indices = []
        for param in parameters:
            circ_indices.extend([f.index for f in self._grad[param]])
        size = len(circ_indices)
        results = self._sampler(circ_indices, [parameter_value] * size, **run_options)

        param_set = set(parameters)
        dists = [Counter() for _ in range(len(parameter_value))]
        metadata = [{} for _ in range(len(parameters))]
        num_bitstrings = 2**self._circuit.num_qubits
        i = 0
        for j, (param, lst) in enumerate(self._grad.items()):
            if param not in param_set:
                continue
            for subest in lst:
                coeff = subest.coeff
                if isinstance(coeff, ParameterExpression):
                    local_map = {param: param_map[param] for param in coeff.parameters}
                    bound_coeff = float(coeff.bind(local_map))
                else:
                    bound_coeff = coeff
                for k, v in results.quasi_dists[i].items():
                    sign, k2 = divmod(k, num_bitstrings)
                    dists[j][k2] += (-1) ** sign * bound_coeff * v
                i += 1

        return SamplerResult(
            quasi_dists=[QuasiDistribution(dist) for dist in dists], metadata=metadata
        )

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
