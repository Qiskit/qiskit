# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import itertools

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter


def build_circuit(width, gates):
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            qc.h(qr[k])
        for k in range(width - 1):
            qc.cx(qr[k], qr[k + 1])

    return qc


class CircuitConstructionBench:
    params = ([1, 2, 5, 8, 14, 20], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates"]
    timeout = 600

    def setup(self, width, gates):
        self.empty_circuit = build_circuit(width, 0)
        self.sample_circuit = build_circuit(width, gates)

    def time_circuit_construction(self, width, gates):
        build_circuit(width, gates)

    def time_circuit_extend(self, _, __):
        self.empty_circuit.extend(self.sample_circuit)

    def time_circuit_copy(self, _, __):
        self.sample_circuit.copy()


def build_parameterized_circuit(width, gates, param_count):
    params = [Parameter("param-%s" % x) for x in range(param_count)]
    param_iter = itertools.cycle(params)

    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            param = next(param_iter)
            qc.u2(0, param, qr[k])
        for k in range(width - 1):
            param = next(param_iter)
            qc.crx(param, qr[k], qr[k + 1])

    return qc, params


class ParameterizedCircuitConstructionBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, _, gates, params):
        if params > gates:
            raise NotImplementedError

    def time_build_parameterized_circuit(self, width, gates, params):
        build_parameterized_circuit(width, gates, params)


class ParameterizedCircuitBindBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, width, gates, params):
        if params > gates:
            raise NotImplementedError
        self.circuit, self.params = build_parameterized_circuit(width, gates, params)

    def time_bind_params(self, _, __, ___):
        # TODO: write more complete benchmarks of assign_parameters
        #  that test more of the input formats / combinations
        self.circuit.assign_parameters({x: 3.14 for x in self.params})
