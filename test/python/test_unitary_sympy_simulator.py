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

from qiskit import (qasm, unroll, QuantumProgram, QuantumJob)
from qiskit.backends.local.sympy_unitarysimulator import SympyUnitarySimulator
from .common import QiskitTestCase


class LocalUnitarySimulatorTest(QiskitTestCase):
    """Test local unitary simulator."""

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/simple.qasm')
        self.qp = QuantumProgram()

    def test_unitary_simulator(self):
        """test generation of circuit unitary"""
        self.qp.load_qasm_file(self.qasm_filename, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
            unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        # strip measurements from circuit to avoid warnings
        circuit['operations'] = [op for op in circuit['operations']
                                 if op['name'] != 'measure']
        # the simulator is expecting a JSON format, so we need to convert it
        # back to JSON
        qobj = {
            'id': 'unitary',
            'config': {
                'max_credits': None,
                'shots': 1,
                'backend': 'local_sympy_unitary_simulator'
            },
            'circuits': [
                {
                    'name': 'test',
                    'compiled_circuit': circuit,
                    'compiled_circuit_qasm': self.qp.get_qasm('example'),
                    'config': {
                        'coupling_map': None,
                        'basis_gates': None,
                        'layout': None,
                        'seed': None
                    }
                }
            ]
        }

        q_job = QuantumJob(qobj,
                           backend='local_sympy_unitary_simulator',
                           preformatted=True)

        result = SympyUnitarySimulator().run(q_job)
        actual = result.get_data('test')['unitary']

        self.assertEqual(actual[0][0], sqrt(2)/2)
        self.assertEqual(actual[0][1], sqrt(2)/2)
        self.assertEqual(actual[0][2], 0)
        self.assertEqual(actual[0][3], 0)
        self.assertEqual(actual[1][0], 0)
        self.assertEqual(actual[1][1], 0)
        self.assertEqual(actual[1][2], sqrt(2)/2)
        self.assertEqual(actual[1][3], -sqrt(2)/2)
        self.assertEqual(actual[2][0], 0)
        self.assertEqual(actual[2][1], 0)
        self.assertEqual(actual[2][2], sqrt(2)/2)
        self.assertEqual(actual[2][3], sqrt(2)/2)
        self.assertEqual(actual[3][0], sqrt(2)/2)
        self.assertEqual(actual[3][1], -sqrt(2)/2)
        self.assertEqual(actual[3][2], 0)
        self.assertEqual(actual[3][3], 0)


if __name__ == '__main__':
    unittest.main()
