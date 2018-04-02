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

import cProfile
import io
import pstats
import unittest

import numpy as np

import qiskit._jobprocessor as jobprocessor
from qiskit import (qasm, unroll, QuantumProgram, QuantumJob, QuantumCircuit,
                    QuantumRegister, ClassicalRegister)
from qiskit.backends._unitarysimulator import UnitarySimulator
from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


class LocalUnitarySimulatorTest(QiskitTestCase):
    """Test local unitary simulator."""

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/example.qasm')
        self.qp = QuantumProgram()

    def tearDown(self):
        pass

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
        # np.savetxt('example_unitary_matrix.dat', numpyMatrix, fmt=fmtstr,
        # delimiter=',')
        expected = np.loadtxt(self._get_resource_path('example_unitary_matrix.dat'),
                              dtype='complex', delimiter=',')
        q_job = QuantumJob(qobj,
                           backend='local_unitary_simulator',
                           preformatted=True)

        result = UnitarySimulator().run(q_job)
        self.assertTrue(np.allclose(result.get_data('test')['unitary'],
                                    expected,
                                    rtol=1e-3))

    def test_two_unitary_simulator(self):
        """test running two circuits

        This test is similar to one in test_quantumprogram but doesn't use
        multiprocessing.
        """
        qr = QuantumRegister('q', 2)
        cr = ClassicalRegister('c', 1)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr)
        qc2.cx(qr[0], qr[1])
        circuits = [qc1, qc2]
        quantum_job = QuantumJob(circuits, do_compile=True,
                                 backend='local_unitary_simulator')
        result = jobprocessor.run_backend(quantum_job)
        unitary1 = result[0]['data']['unitary']
        unitary2 = result[1]['data']['unitary']
        unitaryreal1 = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                                 [0.5, 0.5, -0.5, -0.5],
                                 [0.5, -0.5, -0.5, 0.5]])
        unitaryreal2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                                 [0., 0, 1, 0], [0, 1, 0, 0]])
        norm1 = np.trace(np.dot(np.transpose(np.conj(unitaryreal1)), unitary1))
        norm2 = np.trace(np.dot(np.transpose(np.conj(unitaryreal2)), unitary2))
        self.assertAlmostEqual(norm1, 4)
        self.assertAlmostEqual(norm2, 4)

    def profile_unitary_simulator(self):
        """Profile randomly generated circuits.

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        n_circuits = 100
        max_depth = 40
        max_qubits = 5
        pr = cProfile.Profile()
        random_circuits = RandomQasmGenerator(seed=self.seed,
                                              max_depth=max_depth,
                                              max_qubits=max_qubits)
        random_circuits.add_circuits(n_circuits, do_measure=False)
        self.qp = random_circuits.get_program()
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
