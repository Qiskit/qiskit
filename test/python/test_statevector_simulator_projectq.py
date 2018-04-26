# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

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


import unittest
from qiskit import qasm, unroll, QuantumProgram, QuantumJob
from qiskit.backends.local.statevector_simulator_projectq import StatevectorSimulatorProjectQ
from test.python.common import QiskitTestCase


class StatevectorSimulatorProjectQTest(QiskitTestCase):
    """Test ProjectQ C++ statevector simulator."""

    def setUp(self):
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
                          'layout': None}
        self.qobj = {'id': 'test_sim_single_shot',
                     'config': {
                         'shots': 1
                     },
                     'circuits': [
                         {
                             'name': 'test',
                             'compiled_circuit': circuit,
                             'config': circuit_config
                         }
                     ]}
        self.q_job = QuantumJob(self.qobj,
                                backend=StatevectorSimulatorProjectQ(),
                                circuit_config=circuit_config,
                                preformatted=True)

    def test_statevector_simulator_projectq(self):
        """Test final quantum state for single circuit run."""
        result = StatevectorSimulatorProjectQ().run(self.q_job)
        actual = result.get_data('test')['quantum_state']
        self.assertEqual(result.get_status(), 'COMPLETED')
        self.assertAlmostEqual((abs(actual[0]))**2, 1/2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1/2)


if __name__ == '__main__':
    unittest.main()
