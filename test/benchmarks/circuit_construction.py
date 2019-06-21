# -*- coding: utf-8 -*

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
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

from qiskit import QuantumRegister, QuantumCircuit


def build_circuit(width, gates):
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            qc.h(qr[k])
        for k in range(width-1):
            qc.cx(qr[k], qr[k+1])

    return qc


class CircuitConstructionBench:
    params = ([1, 2, 5, 8, 14, 20], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ['width', 'gates']
    timeout = 600

    def setup(self, width, gates):
        self.empty_circuit = build_circuit(width, 0)
        self.sample_circuit = build_circuit(width, gates)

    def time_circuit_construction(self, width, gates):
        build_circuit(width, gates)

    def time_circuit_extend(self, _, __):
        self.empty_circuit.extend(self.sample_circuit)
