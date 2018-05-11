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
from qiskit import QuantumJob
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.wrapper import get_backend
import qiskit._compiler
from qiskit._compiler import compile_circuit
from ._random_circuit_generator import RandomCircuitGenerator
from .common import QiskitTestCase

try:
    pq_simulator = projectq_simulator.QasmSimulatorProjectQ()
except ImportError:
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
        n_circuits = 5
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
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr, name='test_gate_x')
        qc.x(qr[0])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile([qc], pq_simulator, shots=shots)
        q_job = QuantumJob(qobj, pq_simulator, preformatted=True,
                           resources={'max_credits': qobj['config']['max_credits']})
        job = pq_simulator.run(q_job)
        result_pq = job.result(timeout=30)
        self.assertEqual(result_pq.get_counts(result_pq.get_names()[0]),
                         {'1': shots})

    def test_entangle(self):
        shots = 100
        N = 5
        qr = QuantumRegister(N)
        cr = ClassicalRegister(N)
        qc = QuantumCircuit(qr, cr, name='test_entangle')

        qc.h(qr[0])
        for i in range(1, N):
            qc.cx(qr[0], qr[i])
        qc.measure(qr, cr)
        qobj = qiskit._compiler.compile([qc], pq_simulator, shots=shots)
        timeout = 30
        q_job = QuantumJob(qobj, pq_simulator, preformatted=True,
                           resources={'max_credits': qobj['config']['max_credits']})
        job = pq_simulator.run(q_job)
        result = job.result(timeout=timeout)
        counts = result.get_counts(result.get_names()[0])
        self.log.info(counts)
        for key, _ in counts.items():
            with self.subTest(key=key):
                self.assertTrue(key in ['0' * N, '1' * N])

    def test_random_circuits(self):
        qk_simulator = get_backend('local_qasm_simulator')
        for circuit in self.rqg.get_circuits(format_='QuantumCircuit'):
            self.log.info(circuit.qasm())
            compiled_circuit = compile_circuit(circuit)
            shots = 100
            job_pq = QuantumJob(compiled_circuit,
                                backend=pq_simulator,
                                seed=1, shots=shots)
            job_qk = QuantumJob(compiled_circuit,
                                backend=qk_simulator,
                                seed=1, shots=shots)
            result_pq = pq_simulator.run(job_pq).result()
            result_qk = qk_simulator.run(job_qk).result()
            counts_pq = result_pq.get_counts(result_pq.get_names()[0])
            counts_qk = result_qk.get_counts(result_qk.get_names()[0])
            self.log.info('local_qasm_simulator_projectq: %s', str(counts_pq))
            self.log.info('local_qasm_simulator: %s', str(counts_qk))
            states = counts_qk.keys() | counts_pq.keys()
            # contingency table
            ctable = numpy.array([[counts_pq.get(key, 0) for key in states],
                                  [counts_qk.get(key, 0) for key in states]])
            result = chi2_contingency(ctable)
            self.log.info('chi2_contingency: %s', str(result))
            with self.subTest(circuit=circuit):
                self.assertGreater(result[1], 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
