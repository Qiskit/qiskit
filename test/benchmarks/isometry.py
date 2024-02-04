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
# pylint: disable=unused-argument

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit.library.generalized_gates import Isometry


class IsometryTranspileBench:
    params = ([0, 1, 2, 3], [3, 4, 5, 6])
    param_names = ["number of input qubits", "number of output qubits"]

    def setup(self, m, n):
        q = QuantumRegister(n)
        qc = QuantumCircuit(q)
        if not hasattr(qc, "iso"):
            raise NotImplementedError
        iso = random_unitary(2**n, seed=0).data[:, 0 : 2**m]
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        iso_gate = Isometry(iso, 0, 0)
        qc.append(iso_gate, q)

        self.circuit = qc

    def track_cnot_counts_after_mapping_to_ibmq_16_melbourne(self, *unused):
        coupling = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]
        circuit = transpile(
            self.circuit,
            basis_gates=["u1", "u3", "u2", "cx"],
            coupling_map=coupling,
            seed_transpiler=0,
        )
        counts = circuit.count_ops()
        cnot_count = counts.get("cx", 0)
        return cnot_count
