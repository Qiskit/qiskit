# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,broad-except

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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

import random
import unittest

import numpy
from scipy.stats import chi2_contingency

import qiskit.backends.local.qasm_simulator_projectq as projectq_simulator
from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.wrapper import get_backend, execute
from ._random_circuit_generator import RandomCircuitGenerator
from .common import QiskitTestCase

try:
    pq_simulator = projectq_simulator.QasmSimulatorProjectQ()
except Exception as err:
    _skip_class = True
else:
    _skip_class = False


@unittest.skipIf(_skip_class, 'Project Q C++ simulator unavailable')
class TestQasmSimulatorProjectQ(QiskitTestCase):
    """
    Test projectq simulator.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Set up random circuits
        n_circuits = 1
        min_depth = 1
        max_depth = 10
        min_qubits = 1
        max_qubits = 4
        random_circuits = RandomCircuitGenerator(min_qubits=min_qubits,
                                                 max_qubits=max_qubits,
                                                 min_depth=min_depth,
                                                 max_depth=max_depth,
                                                 seed=None)
        for _ in range(n_circuits):
            basis = list(random.sample(random_circuits.op_signature.keys(),
                                       random.randint(2, 7)))
            if 'reset' in basis:
                basis.remove('reset')
            if 'u0' in basis:
                basis.remove('u0')
            random_circuits.add_circuits(1, basis=basis)
        cls.rqg = random_circuits

    def test_gate_x(self):
        shots = 100
        qp = QuantumProgram()
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.x(qr[0])
        qc.measure(qr, cr)
        qp.add_circuit("circuit_name", qc)
        result_pq = qp.execute('circuit_name',
                               backend='local_qasm_simulator_projectq',
                               seed=1, shots=shots)
        self.assertEqual(result_pq.get_counts(),
                         {'1': shots})

    def test_entangle(self):
        N = 5
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", N)
        cr = qp.create_classical_register("cr", N)
        qc = qp.create_circuit("circuit_name", [qr], [cr])
        qc.h(qr[0])
        for i in range(1, N):
            qc.cx(qr[0], qr[i])
        qc.measure(qr, cr)
        result = qp.execute(['circuit_name'],
                            backend='local_qasm_simulator_projectq',
                            seed=1, shots=100)
        counts = result.get_counts(result.get_names()[0])
        self.log.info(counts)
        for key, _ in counts.items():
            with self.subTest(key=key):
                self.assertTrue(key in ['0' * N, '1' * N])

    def test_random_circuits(self):
        qk_simulator = get_backend('local_qasm_simulator')
        for circuit in self.rqg.get_circuits(format_='QuantumCircuit'):
            self.log.info(circuit.qasm())
            shots = 1000
            min_cnts = int(shots / 10)
            result_pq = execute(circuit, pq_simulator.name)
            result_qk = execute(circuit, qk_simulator.name)
            counts_pq = result_pq.get_counts(result_pq.get_names()[0])
            counts_qk = result_qk.get_counts(result_qk.get_names()[0])
            # filter states with few counts
            counts_pq = {key: cnt for key, cnt in counts_pq.items()
                         if cnt > min_cnts}
            counts_qk = {key: cnt for key, cnt in counts_qk.items()
                         if cnt > min_cnts}
            self.log.info('local_qasm_simulator_projectq: %s', str(counts_pq))
            self.log.info('local_qasm_simulator: %s', str(counts_qk))
            threshold = 0.05 * shots
            self.assertDictAlmostEqual(counts_pq, counts_qk, threshold)
            states = counts_qk.keys()
            # contingency table
            ctable = numpy.array([[counts_pq[key] for key in states],
                                  [counts_qk[key] for key in states]])
            result = chi2_contingency(ctable)
            self.log.info('chi2_contingency: %s', str(result))
            with self.subTest(circuit=circuit):
                self.assertGreater(result[1], 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
