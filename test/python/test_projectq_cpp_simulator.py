# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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

import unittest
import os
import logging
from math import sqrt
import qiskit
import qiskit.backends._projectq_simulator as projectq_simulator
import qiskit.backends._qasmsimulator as qasm_simulator
from qiskit import QuantumProgram
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QuantumJob
from qiskit import _openquantumcompiler as openquantumcompiler
from .common import QiskitTestCase
from ._random_circuit_generator import RandomCircuitGenerator

try:
    pq_simulator = projectq_simulator.ProjectQSimulator()
except Exception as err:
    _skip_class = True
else:
    _skip_class = False

@unittest.skipIf(_skip_class, 'Project Q C++ simulator unavailable')
class TestProjectQCppSimulator(QiskitTestCase):
    """
    Test job_pocessor module.
    """

    @classmethod
    def setUpClass(cls):
        super(TestProjectQCppSimulator, cls).setUpClass()
        nCircuits = 10
        minDepth = 1
        maxDepth = 10
        minQubits = 1
        maxQubits = 4
        randomCircuits = RandomCircuitGenerator(minQubits=minQubits,
                                                maxQubits=maxQubits,
                                                minDepth=minDepth,
                                                maxDepth=maxDepth,
                                                seed=3)
        randomCircuits.add_circuits(nCircuits)
        cls.rqg = randomCircuits
        cls.seed = 88
        cls.qasmFileName = os.path.join(qiskit.__path__[0],
                                        '../test/python/qasm/example.qasm')
        with open(cls.qasmFileName, 'r') as qasm_file:
            cls.qasm_text = qasm_file.read()
            qr1 = QuantumRegister('q1', 2)
            qr2 = QuantumRegister('q2', 3)
            cr = ClassicalRegister('c', 2)
            qcs = QuantumCircuit(qr1, qr2, cr)
            qcs.h(qr1[0])
            qcs.h(qr2[2])
            qcs.measure(qr1[0], cr[0])
            cls.qcs = qcs
            # create qobj
        compiled_circuit1 = openquantumcompiler.compile(cls.qcs.qasm(),
                                                        format='json')
        compiled_circuit2 = openquantumcompiler.compile(cls.qasm_text,
                                                        format='json')
        cls.qobj = {'id': 'test_qobj',
                    'config': {
                        'max_credits': 3,
                        'shots': 100,
                        'backend': 'local_projectq_simulator',
                        'seed': 1111
                    },
                    'circuits': [
                        {
                            'name': 'test_circuit1',
                            'compiled_circuit': compiled_circuit1,
                            'basis_gates': 'u1,u2,u3,cx,id',
                            'layout': None,
                        },
                        {
                            'name': 'test_circuit2',
                            'compiled_circuit': compiled_circuit2,
                            'basis_gates': 'u1,u2,u3,cx,id',
                            'layout': None,
                        }
                    ]
        }
        cls.q_job = QuantumJob(cls.qobj,
                               backend='local_projectq_simulator',
                               preformatted=True)


    def test_gate_x(self):
        N = 1
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
                         {'1':shots})

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
        for key, value in counts.items():
            self.assertTrue(key in ['0' * N, '1' * N])
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
