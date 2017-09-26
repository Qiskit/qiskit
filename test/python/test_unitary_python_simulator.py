# -*- coding: utf-8 -*-

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

import cProfile
import io
import json
import pstats
import unittest

import numpy as np

from qiskit import qasm, unroll, QuantumProgram
from qiskit.simulators._unitarysimulator import UnitarySimulator

from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


class LocalUnitarySimulatorTest(QiskitTestCase):
    """Test local unitary simulator."""

    def setUp(self):
        self.seed = 88
        self.qasmFileName = self._get_resource_path('qasm/example.qasm')
        self.qp = QuantumProgram()

    def tearDown(self):
        pass

    def test_unitary_simulator(self):
        """test generation of circuit unitary"""
        shots = 1024
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        #strip measurements from circuit to avoid warnings
        circuit['operations'] = [op for op in circuit['operations']
                                 if op['name'] != 'measure']
        # the simulator is expecting a JSON format, so we need to convert it back to JSON
        qobj = {'id': 'unitary',
                'config': {
                    'max_credits': None,
                    'shots': 1,
                    'backend': 'local_unitary_simulator'
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
        # numpy.savetxt currently prints complex numbers in a way
        # loadtxt can't read. To save file do,
        # fmtstr=['% .4g%+.4gj' for i in range(numCols)]
        # np.savetxt('example_unitary_matrix.dat', numpyMatrix, fmt=fmtstr, delimiter=',')
        expected = np.loadtxt(os.path.join(self.modulePath,
                                           'example_unitary_matrix.dat'),
                              dtype='complex', delimiter=',')
        result = UnitarySimulator(qobj).run()
        self.assertTrue(np.allclose(result.get_data('test')['unitary'],
                                    expected,
                                    rtol=1e-3))

    def profile_unitary_simulator(self):
        """Profile randomly generated circuits.

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        nCircuits = 100
        maxDepth = 40
        maxQubits = 5
        pr = cProfile.Profile()
        randomCircuits = RandomQasmGenerator(seed=self.seed,
                                             maxDepth=maxDepth,
                                             maxQubits=maxQubits)
        randomCircuits.add_circuits(nCircuits, doMeasure=False)
        self.qp = randomCircuits.getProgram()
        pr.enable()
        self.qp.execute(self.qp.get_circuit_names(),
                        backend='local_unitary_simulator')
        pr.disable()
        sout = io.StringIO()
        ps = pstats.Stats(pr, stream=sout).sort_stats('cumulative')
        self.log.info('------- start profiling UnitarySimulator -----------')
        ps.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling UnitarySimulator -----------')
        sout.close()
        pr.dump_stats(self.moduleName + '.prof')

if __name__ == '__main__':
    unittest.main()
