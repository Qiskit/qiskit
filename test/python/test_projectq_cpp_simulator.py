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

import qiskit.backends._projectq_simulator as projectq_simulator
import qiskit.backends._qasmsimulator as qasm_simulator
from qiskit import QuantumJob
from qiskit import QuantumProgram
from qiskit import _openquantumcompiler as openquantumcompiler
from ._random_circuit_generator import RandomCircuitGenerator
from .common import QiskitTestCase

try:
    pq_simulator = projectq_simulator.ProjectQSimulator()
except Exception as err:
    _skip_class = True
else:
    _skip_class = False


@unittest.skipIf(_skip_class, 'Project Q C++ simulator unavailable')
class TestProjectQCppSimulator(QiskitTestCase):
    """
    Test projectq simulator.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Set up random circuits
        n_circuits = 20
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
            random_circuits.add_circuits(1, basis=basis)
        cls.rqg = random_circuits

    def test_gate_x(self):
        shots = 100
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", 1)
        cr = qp.create_classical_register("cr", 1)
        qc = qp.create_circuit("circuitName", [qr], [cr])
        qc.x(qr[0])
        qc.measure(qr, cr)
        result_pq = qp.execute(['circuitName'],
                               backend='local_projectq_simulator',
                               seed=1, shots=shots)
        self.assertEqual(result_pq.get_counts(result_pq.get_names()[0]),
                         {'1': shots})

    def test_entangle(self):
        N = 5
        qp = QuantumProgram()
        qr = qp.create_quantum_register("qr", N)
        cr = qp.create_classical_register("cr", N)
        qc = qp.create_circuit("circuitName", [qr], [cr])
        qc.h(qr[0])
        for i in range(1, N):
            qc.cx(qr[0], qr[i])
        qc.measure(qr, cr)
        result = qp.execute(['circuitName'],
                            backend='local_projectq_simulator',
                            seed=1, shots=100)
        counts = result.get_counts(result.get_names()[0])
        self.log.info(counts)
        for key, _ in counts.items():
            with self.subTest(key=key):
                self.assertTrue(key in ['0' * N, '1' * N])

    def test_random_circuits(self):
        local_simulator = qasm_simulator.QasmSimulator()
        for circuit in self.rqg.get_circuits(format_='QuantumCircuit'):
            self.log.info(circuit.qasm())
            compiled_circuit = openquantumcompiler.compile(circuit)
            shots = 100
            min_cnts = int(shots / 10)
            job_pq = QuantumJob(compiled_circuit,
                                backend='local_projectq_simulator',
                                seed=1, shots=shots)
            job_py = QuantumJob(compiled_circuit,
                                backend='local_qasm_simulator',
                                seed=1, shots=shots)
            result_pq = pq_simulator.run(job_pq)
            result_py = local_simulator.run(job_py)
            counts_pq = result_pq.get_counts(result_pq.get_names()[0])
            counts_py = result_py.get_counts(result_py.get_names()[0])
            # filter states with few counts
            counts_pq = {key: cnt for key, cnt in counts_pq.items()
                         if cnt > min_cnts}
            counts_py = {key: cnt for key, cnt in counts_py.items()
                         if cnt > min_cnts}
            self.log.info('local_projectq_simulator: %s', str(counts_pq))
            self.log.info('local_qasm_simulator: %s', str(counts_py))
            self.assertTrue(counts_pq.keys() == counts_py.keys())
            states = counts_py.keys()
            # contingency table
            ctable = numpy.array([[counts_pq[key] for key in states],
                                  [counts_py[key] for key in states]])
            result = chi2_contingency(ctable)
            self.log.info('chi2_contingency: %s', str(result))
            with self.subTest(circuit=circuit):
                self.assertGreater(result[1], 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
