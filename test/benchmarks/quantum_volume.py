# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=no-member,invalid-name,missing-docstring
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object

"""Module for estimating quantum volume.
See arXiv:1811.12926 [quant-ph]"""

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper import two_qubit_kak
from qiskit import BasicAer
from qiskit import transpiler


def build_model_circuit(qreg, depth=None, seed=None):
    """Create quantum volume model circuit on quantum register qreg of given
    depth (default depth is equal to width) and random seed.
    The model circuits consist of layers of Haar random
    elements of U(4) applied between corresponding pairs
    of qubits in a random bipartition.
    """
    width = len(qreg)
    depth = depth or width

    np.random.seed(seed)
    circuit = QuantumCircuit(
        qreg, name="Qvolume: %s by %s, seed: %s" % (width, depth, seed))

    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = np.random.permutation(width)

        # For each pair p in Pj, generate Haar random U(4)
        # Decompose each U(4) into CNOT + SU(2)
        for k in range(width // 2):
            U = random_unitary_matrix(4)
            for gate in two_qubit_kak(U):
                qs = [qreg[int(perm[2 * k + i])] for i in gate["args"]]
                pars = gate["params"]
                name = gate["name"]
                if name == "cx":
                    circuit.cx(qs[0], qs[1])
                elif name == "u1":
                    circuit.u1(pars[0], qs[0])
                elif name == "u2":
                    circuit.u2(*pars[:2], qs[0])
                elif name == "u3":
                    circuit.u3(*pars[:3], qs[0])
                elif name == "id":
                    pass  # do nothing
                else:
                    raise Exception("Unexpected gate name: %s" % name)
    return circuit


class QuantumVolumeBenchmark:
    params = ([1, 2, 3, 5, 8, 13, 14], [1, 2, 3, 5, 8, 13, 21, 34])
    param_names = ['qubits', 'depth']
    timeout = 600

    def setup(self, n, depth):
        random_seed = np.random.seed(10)
        qreg = QuantumRegister(n)
        self.circuit = build_model_circuit(qreg, depth=depth, seed=random_seed)
        self.sim_backend = BasicAer.get_backend('qasm_simulator')

    def time_simulator_transpile(self, _, __):
        transpiler.transpile(self.circuit, self.sim_backend)

    def time_ibmq_backend_transpile(self, _, __):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                        [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                        [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                        [13, 12]]
        transpiler.transpile(self.circuit,
                             basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                             coupling_map=coupling_map)
