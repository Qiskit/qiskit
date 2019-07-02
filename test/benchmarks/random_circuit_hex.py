# -*- coding: utf-8 -*-

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

import copy

try:
    from qiskit.quantum_info.synthesis import euler_angles_1q
except ImportError:
    try:
        from qiskit.mapper.compiling import euler_angles_1q
    except ImportError:
        from qiskit.mapper._compiling import euler_angles_1q
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer
try:
    from qiskit.compiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_transpiler'
except ImportError:
    from qiskit.transpiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_mapper'
try:
    from qiskit.quantum_info.random import random_unitary
    HAS_RANDOM_UNITARY = True
except ImportError:
    from qiskit.tools.qi.qi import random_unitary_matrix
    HAS_RANDOM_UNITARY = False


# Make a random circuit on a ring
def make_circuit_ring(nq, depth, seed):
    assert int(nq / 2) == nq / 2  # for now size of ring must be even
    # Create a Quantum Register
    q = QuantumRegister(nq)
    # Create a Classical Register
    c = ClassicalRegister(nq)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, c)
    offset = 1
    # initial round of random single-qubit unitaries
    for i in range(nq):
        qc.h(q[i])
    for j in range(depth):
        for i in range(int(nq / 2)):  # round of CNOTS
            k = i * 2 + offset + j % 2  # j%2 makes alternating rounds overlap
            qc.cx(q[k % nq], q[(k + 1) % nq])
        for i in range(nq):  # round of single-qubit unitaries
            if HAS_RANDOM_UNITARY:
                u = random_unitary(2, seed).data
            else:
                u = random_unitary_matrix(2)

            angles = euler_angles_1q(u)
            qc.u3(angles[0], angles[1], angles[2], q[i])

    # insert the final measurements
    qcm = copy.deepcopy(qc)
    for i in range(nq):
        qcm.measure(q[i], c[i])
    return [qc, qcm, nq]


class BenchRandomCircuitHex:
    params = [2 * i for i in range(2, 8)]
    param_names = ['n_qubits']
    version = 3

    def setup(self, n):
        depth = 2 * n
        self.seed = 0
        self.circuit = make_circuit_ring(n, depth, self.seed)[0]
        self.sim_backend = BasicAer.get_backend('qasm_simulator')

    def time_simulator_transpile(self, _):
        transpile(self.circuit, self.sim_backend,
                  **{TRANSPILER_SEED_KEYWORD: self.seed})

    def track_depth_simulator_transpile(self, _):
        return transpile(self.circuit, self.sim_backend,
                         **{TRANSPILER_SEED_KEYWORD: self.seed}).depth()

    def time_ibmq_backend_transpile(self, _):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                        [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                        [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                        [13, 12]]
        transpile(self.circuit,
                  basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                  coupling_map=coupling_map,
                  **{TRANSPILER_SEED_KEYWORD: self.seed})

    def track_depth_ibmq_backend_transpile(self, _):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                        [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                        [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                        [13, 12]]
        return transpile(self.circuit,
                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                         coupling_map=coupling_map,
                         **{TRANSPILER_SEED_KEYWORD: self.seed}).depth()
