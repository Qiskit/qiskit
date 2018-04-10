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

from sympy import sqrt

from qiskit import qasm, unroll, QuantumProgram, QuantumJob
from qiskit.backends.local.sympy_qasmsimulator import SympyQasmSimulator
from .common import QiskitTestCase


class LocalQasmSimulatorTest(QiskitTestCase):
    """Test local qasm simulator."""

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/simple.qasm')
        self.qp = QuantumProgram()
        self.qp.load_qasm_file(self.qasm_filename, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
            unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        circuit_config = {'coupling_map': None,
                          'basis_gates': 'u1,u2,u3,cx,id',
                          'layout': None,
                          'seed': self.seed}
        resources = {'max_credits': 3,
                     'wait': 5,
                     'timeout': 120}
        self.qobj = {'id': 'test_sim_single_shot',
                     'config': {
                         'max_credits': resources['max_credits'],
                         'shots': 1024,
                         'backend': 'local_sympy_qasm_simulator',
                     },
                     'circuits': [
                         {
                             'name': 'test',
                             'compiled_circuit': circuit,
                             'compiled_circuit_qasm': None,
                             'config': circuit_config
                         }
                     ]}
        self.q_job = QuantumJob(self.qobj,
                                backend='local_sympy_qasm_simulator',
                                circuit_config=circuit_config,
                                seed=self.seed,
                                resources=resources,
                                preformatted=True)

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = SympyQasmSimulator().run(self.q_job)
        actual = result.get_data('test')['quantum_state']
        self.assertEqual(result.get_status(), 'COMPLETED')
        self.assertEqual(actual[0], sqrt(2)/2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertEqual(actual[3], sqrt(2)/2)


if __name__ == '__main__':
    unittest.main()
