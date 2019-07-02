# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
# pylint: disable=unused-argument

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info.random import random_unitary


class IsometryTranspileBench:
    params = ([0, 1, 2, 3], [3, 4, 5, 6])
    param_names = ['number of input qubits', 'number of output qubits']

    def setup(self, m, n):
        q = QuantumRegister(n)
        qc = QuantumCircuit(q)
        if not hasattr(qc, 'iso'):
            raise NotImplementedError
        iso = random_unitary(2 ** n, seed=0).data[:, 0:2 ** m]
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        qc.iso(iso, q[:m], q[m:])
        self.circuit = qc

    def time_simulator_transpile(self, *unused):
        transpile(self.circuit, basis_gates=['u1', 'u3', 'u2', 'cx'],
                  seed_transpiler=0)

    def track_cnot_counts(self, *unused):
        circuit = transpile(self.circuit, basis_gates=['u1', 'u3', 'u2', 'cx'],
                            seed_transpiler=0)
        counts = circuit.count_ops()
        cnot_count = counts.get('cx', 0)
        return cnot_count
