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

import os
import unittest

import numpy as np
from numpy.linalg import norm

import qiskit
import qiskit.backends._qiskit_cpp_simulator as qiskitsimulator
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QuantumJob
from qiskit import QuantumRegister
from qiskit import _openquantumcompiler as openquantumcompiler
from .common import QiskitTestCase


class TestLocalQiskitSimulator(QiskitTestCase):
    """
    Test job_processor module.
    """

    def setUp(self):
        self.seed = 88
        self.qasm_filename = os.path.join(qiskit.__path__[0],
                                          '../test/python/qasm/example.qasm')
        with open(self.qasm_filename, 'r') as qasm_file:
            self.qasm_text = qasm_file.read()
            self.qasm_ast = qiskit.qasm.Qasm(data=self.qasm_text).parse()
            self.qasm_be = qiskit.unroll.CircuitBackend(['u1', 'u2', 'u3', 'id', 'cx'])
            self.qasm_circ = qiskit.unroll.Unroller(self.qasm_ast, self.qasm_be).execute()
        qr = QuantumRegister('q', 2)
        cr = ClassicalRegister('c', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qc = qc
        # create qobj
        compiled_circuit1 = openquantumcompiler.compile(self.qc,
                                                        format='json')
        compiled_circuit2 = openquantumcompiler.compile(self.qasm_circ,
                                                        format='json')
        self.qobj = {'id': 'test_qobj',
                     'config': {
                         'max_credits': 3,
                         'shots': 100,
                         'backend': 'local_qiskit_simulator',
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
                     ]}
        self.q_job = QuantumJob(self.qobj,
                                backend='local_qiskit_simulator',
                                preformatted=True)

    def test_x90_coherent_error_matrix(self):
        X90 = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        U = qiskitsimulator.x90_error_matrix(0., 0.).dot(X90)
        target = X90
        self.assertAlmostEqual(norm(U - target), 0.0, places=10,
                               msg="identity error matrix")
        U = qiskitsimulator.x90_error_matrix(np.pi / 2., 0.).dot(X90)
        target = -1j * np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = qiskitsimulator.x90_error_matrix(0., np.pi / 2.).dot(X90)
        target = np.array([[1., -1], [1, 1.]]) / np.sqrt(2.)
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = qiskitsimulator.x90_error_matrix(np.pi / 2, np.pi / 2.).dot(X90)
        target = np.array([[0., -1], [1, 0.]])
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = qiskitsimulator.x90_error_matrix(0.02, -0.03)
        self.assertAlmostEqual(norm(U.dot(U.conj().T) - np.eye(2)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_cx_coherent_error_matrix(self):
        CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        U = qiskitsimulator.cx_error_matrix(0., 0.).dot(CX)
        target = CX
        self.assertAlmostEqual(norm(U - target), 0.0, places=10,
                               msg="identity error matrix")
        U = qiskitsimulator.cx_error_matrix(np.pi / 2., 0.).dot(CX)
        target = np.array([[1, 0, 1j, 0],
                           [0, -1j, 0, 1],
                           [1j, 0, 1, 0],
                           [0, 1, 0, -1j]]) / np.sqrt(2)
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = qiskitsimulator.cx_error_matrix(0.03, -0.04)
        self.assertAlmostEqual(norm(U.dot(U.conj().T) - np.eye(4)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_run_qobj(self):
        try:
            simulator = qiskitsimulator.QISKitCppSimulator()
        except FileNotFoundError as fnferr:
            raise unittest.SkipTest(
                'cannot find {} in path'.format(fnferr))
        result = simulator.run(self.q_job)

        # TODO: reenable the assertEqual with the exact counts once the issue
        # with the randomness is solved, as different compiler/oses seem to
        # provide different values even for the same seed.
        expected2 = {'000 000': 18,
                     '001 001': 15,
                     '010 010': 13,
                     '011 011': 11,
                     '100 100': 10,
                     '101 101': 10,
                     '110 110': 12,
                     '111 111': 11}
        # self.assertEqual(result.get_counts('test_circuit2'), expected2)
        self.assertEqual(set(result.get_counts('test_circuit2').keys()),
                         set(expected2.keys()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
