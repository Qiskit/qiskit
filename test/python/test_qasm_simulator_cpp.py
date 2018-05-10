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

import json
import unittest
import numpy as np
from numpy.linalg import norm

import qiskit
import qiskit._compiler
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import QuantumJob
from qiskit import QuantumRegister
from qiskit.backends.local.qasm_simulator_cpp import (QasmSimulatorCpp,
                                                      cx_error_matrix,
                                                      x90_error_matrix)
from .common import QiskitTestCase


class TestLocalQasmSimulatorCpp(QiskitTestCase):
    """
    Test job_processor module.
    """

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/example.qasm')
        with open(self.qasm_filename, 'r') as qasm_file:
            self.qasm_text = qasm_file.read()
            self.qasm_ast = qiskit.qasm.Qasm(data=self.qasm_text).parse()
            self.qasm_be = qiskit.unroll.CircuitBackend(['u1', 'u2', 'u3', 'id', 'cx'])
            self.qasm_circ = qiskit.unroll.Unroller(self.qasm_ast, self.qasm_be).execute()
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qc = qc
        # create qobj
        compiled_circuit1 = qiskit._compiler.compile_circuit(self.qc, format='json')
        compiled_circuit2 = qiskit._compiler.compile_circuit(self.qasm_circ, format='json')
        self.qobj = {'id': 'test_qobj',
                     'config': {
                         'max_credits': 3,
                         'shots': 2000,
                         'backend_name': 'local_qasm_simulator_cpp',
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
        # Simulator backend
        try:
            self.backend = QasmSimulatorCpp()
        except FileNotFoundError as fnferr:
            raise unittest.SkipTest(
                'cannot find {} in path'.format(fnferr))

        self.q_job = QuantumJob(self.qobj,
                                backend=self.backend,
                                preformatted=True)

    def test_x90_coherent_error_matrix(self):
        X90 = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        U = x90_error_matrix(0., 0.).dot(X90)
        target = X90
        self.assertAlmostEqual(norm(U - target), 0.0, places=10,
                               msg="identity error matrix")
        U = x90_error_matrix(np.pi / 2., 0.).dot(X90)
        target = -1j * np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = x90_error_matrix(0., np.pi / 2.).dot(X90)
        target = np.array([[1., -1], [1, 1.]]) / np.sqrt(2.)
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = x90_error_matrix(np.pi / 2, np.pi / 2.).dot(X90)
        target = np.array([[0., -1], [1, 0.]])
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = x90_error_matrix(0.02, -0.03)
        self.assertAlmostEqual(norm(U.dot(U.conj().T) - np.eye(2)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_cx_coherent_error_matrix(self):
        CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        U = cx_error_matrix(0., 0.).dot(CX)
        target = CX
        self.assertAlmostEqual(norm(U - target), 0.0, places=10,
                               msg="identity error matrix")
        U = cx_error_matrix(np.pi / 2., 0.).dot(CX)
        target = np.array([[1, 0, 1j, 0],
                           [0, -1j, 0, 1],
                           [1j, 0, 1, 0],
                           [0, 1, 0, -1j]]) / np.sqrt(2)
        self.assertAlmostEqual(norm(U - target), 0.0, places=10)
        U = cx_error_matrix(0.03, -0.04)
        self.assertAlmostEqual(norm(U.dot(U.conj().T) - np.eye(4)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_run_qobj(self):
        result = self.backend.run(self.q_job).result()
        shots = self.qobj['config']['shots']
        threshold = 0.04 * shots
        counts = result.get_counts('test_circuit2')
        target = {'100 100': shots / 8, '011 011': shots / 8,
                  '101 101': shots / 8, '111 111': shots / 8,
                  '000 000': shots / 8, '010 010': shots / 8,
                  '110 110': shots / 8, '001 001': shots / 8}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_qobj_measure_opt(self):
        filename = self._get_resource_path('qobj/cpp_measure_opt.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        shots = q_job.qobj['config']['shots']
        expected_data = {
            'measure (opt)': {
                'deterministic': True,
                'counts': {'00': shots},
                'statevector': np.array([1, 0, 0, 0])},
            'x0 measure (opt)': {
                'deterministic': True,
                'counts': {'01': shots},
                'statevector': np.array([0, 1, 0, 0])},
            'x1 measure (opt)': {
                'deterministic': True,
                'counts': {'10': shots},
                'statevector': np.array([0, 0, 1, 0])},
            'x0 x1 measure (opt)': {
                'deterministic': True,
                'counts': {'11': shots},
                'statevector': np.array([0, 0, 0, 1])},
            'y0 measure (opt)': {
                'deterministic': True,
                'counts': {'01': shots},
                'statevector': np.array([0, 1j, 0, 0])},
            'y1 measure (opt)': {
                'deterministic': True,
                'counts': {'10': shots},
                'statevector': np.array([0, 0, 1j, 0])},
            'y0 y1 measure (opt)': {
                'deterministic': True,
                'counts': {'11': shots},
                'statevector': np.array([0, 0, 0, -1j])},
            'h0 measure (opt)': {
                'deterministic': False,
                'counts': {'00': shots / 2, '01': shots / 2},
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])},
            'h1 measure (opt)': {
                'deterministic': False,
                'counts': {'00': shots / 2, '10': shots / 2},
                'statevector': np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])},
            'h0 h1 measure (opt)': {
                'deterministic': False,
                'counts': {'00': shots / 4, '01': shots / 4,
                           '10': shots / 4, '11': shots / 4},
                'statevector': np.array([0.5, 0.5, 0.5, 0.5])},
            'bell measure (opt)': {
                'deterministic': False,
                'counts': {'00': shots / 2, '11': shots / 2},
                'statevector': np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])}
        }

        for name in expected_data:
            # Check counts:
            counts = result.get_counts(name)
            expected_counts = expected_data[name]['counts']
            if expected_data[name].get('deterministic', False):
                self.assertEqual(counts, expected_counts,
                                 msg=name + ' counts')
            else:
                threshold = 0.04 * shots
                self.assertDictAlmostEqual(counts, expected_counts,
                                           threshold, msg=name + 'counts')
            # Check snapshot
            snapshots = result.get_snapshots(name)
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = snapshots['0']['statevector'][0]
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_measure_opt_flag(self):
        filename = self._get_resource_path('qobj/cpp_measure_opt_flag.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        shots = q_job.qobj['config']['shots']
        sampled_measurements = {
            'measure (sampled)': True,
            'trivial (sampled)': True,
            'reset1 (shots)': False,
            'reset2 (shots)': False,
            'reset3 (shots)': False,
            'gate1 (shots)': False,
            'gate2 (shots)': False,
            'gate3 (shots)': False,
            'gate4 (shots)': False
        }

        for name in sampled_measurements:
            snapshots = result.get_snapshots(name)
            # Check snapshot keys
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            # Check number of snapshots
            # there should be 1 for measurement sampling optimization
            # and there should be >1 for each shot beign simulated.
            num_snapshots = len(snapshots['0'].get('statevector', []))
            if sampled_measurements[name] is True:
                self.assertEqual(num_snapshots, 1,
                                 msg=name + ' snapshot length')
            else:
                self.assertEqual(num_snapshots, shots,
                                 msg=name + ' snapshot length')

    def test_qobj_reset(self):
        filename = self._get_resource_path('qobj/cpp_reset.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        expected_data = {
            'reset': {'statevector': np.array([1, 0])},
            'x reset': {'statevector': np.array([1, 0])},
            'y reset': {'statevector': np.array([1, 0])},
            'h reset': {'statevector': np.array([1, 0])}
        }
        for name in expected_data:
            # Check snapshot is |0> state
            snapshots = result.get_snapshots(name)
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = snapshots['0']['statevector'][0]
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_save_load(self):
        filename = self._get_resource_path('qobj/cpp_save_load.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()

        snapshots = result.get_snapshots('save_command')
        self.assertEqual(set(snapshots), {'0', '1', '10', '11'},
                         msg='snapshot keys')
        state0 = snapshots['0']['statevector'][0]
        state10 = snapshots['10']['statevector'][0]
        state1 = snapshots['1']['statevector'][0]
        state11 = snapshots['11']['statevector'][0]

        expected_state0 = np.array([1, 0])
        expected_state10 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

        fidelity0 = np.abs(expected_state0.dot(state0.conj())) ** 2
        fidelity1 = np.abs(expected_state0.dot(state1.conj())) ** 2
        fidelity10 = np.abs(expected_state10.dot(state10.conj())) ** 2
        fidelity11 = np.abs(expected_state10.dot(state11.conj())) ** 2
        self.assertAlmostEqual(fidelity0, 1.0, places=10, msg='snapshot 0')
        self.assertAlmostEqual(fidelity10, 1.0, places=10, msg='snapshot 0')
        self.assertAlmostEqual(fidelity1, 1.0, places=10, msg='snapshot 0')
        self.assertAlmostEqual(fidelity11, 1.0, places=10, msg='snapshot 0')

    def test_qobj_single_qubit_gates(self):
        filename = self._get_resource_path('qobj/cpp_single_qubit_gates.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        expected_data = {
            'snapshot': {
                'statevector': np.array([1, 0])},
            'id(U)': {
                'statevector': np.array([1, 0])},
            'id(u3)': {
                'statevector': np.array([1, 0])},
            'id(u1)': {
                'statevector': np.array([1, 0])},
            'id(direct)': {
                'statevector': np.array([1, 0])},
            'x(U)': {
                'statevector': np.array([0, 1])},
            'x(u3)': {
                'statevector': np.array([0, 1])},
            'x(direct)': {
                'statevector': np.array([0, 1])},
            'y(U)': {
                'statevector': np.array([0, 1j])},
            'y(u3)': {
                'statevector': np.array([0, 1j])},
            'y(direct)': {
                'statevector': np.array([0, 1j])},
            'h(U)': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])},
            'h(u3)': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])},
            'h(u2)': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])},
            'h(direct)': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])},
            'h(direct) z(U)': {
                'statevector': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])},
            'h(direct) z(u3)': {
                'statevector': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])},
            'h(direct) z(u1)': {
                'statevector': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])},
            'h(direct) z(direct)': {
                'statevector': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])},
            'h(direct) s(U)': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])},
            'h(direct) s(u3)': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])},
            'h(direct) s(u1)': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])},
            'h(direct) s(direct)': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])},
            'h(direct) sdg(U)': {
                'statevector': np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])},
            'h(direct) sdg(u3)': {
                'statevector': np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])},
            'h(direct) sdg(u1)': {
                'statevector': np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])},
            'h(direct) sdg(direct)': {
                'statevector': np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])},
            'h(direct) t(U)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 + 0.5j])},
            'h(direct) t(u3)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 + 0.5j])},
            'h(direct) t(u1)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 + 0.5j])},
            'h(direct) t(direct)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 + 0.5j])},
            'h(direct) tdg(U)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 - 0.5j])},
            'h(direct) tdg(u3)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 - 0.5j])},
            'h(direct) tdg(u1)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 - 0.5j])},
            'h(direct) tdg(direct)': {
                'statevector': np.array([1 / np.sqrt(2), 0.5 - 0.5j])}
        }

        for name in expected_data:
            # Check snapshot
            snapshots = result.get_snapshots(name)
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = snapshots['0']['statevector'][0]
            expected_state = expected_data[name]['statevector']
            inner_product = expected_state.dot(state.conj())
            self.assertAlmostEqual(inner_product, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_two_qubit_gates(self):
        filename = self._get_resource_path('qobj/cpp_two_qubit_gates.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        expected_data = {
            'h0 CX01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])},
            'h0 CX10': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])},
            'h1 CX01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])},
            'h1 CX10': {
                'statevector': np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])},
            'h0 cx01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])},
            'h0 cx10': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])},
            'h1 cx01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])},
            'h1 cx10': {
                'statevector': np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])},
            'h0 cz01': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])},
            'h0 cz10': {
                'statevector': np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])},
            'h1 cz01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])},
            'h1 cz10': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])},
            'h0 h1 cz01': {'statevector': np.array([0.5, 0.5, 0.5, -0.5])},
            'h0 h1 cz10': {'statevector': np.array([0.5, 0.5, 0.5, -0.5])},
            'h0 rzz01': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0])},
            'h0 rzz10': {
                'statevector': np.array([1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0])},
            'h1 rzz01': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1j / np.sqrt(2), 0])},
            'h1 rzz10': {
                'statevector': np.array([1 / np.sqrt(2), 0, 1j / np.sqrt(2), 0])},
            'h0 h1 rzz01': {'statevector': np.array([0.5, 0.5j, 0.5j, 0.5])},
            'h0 h1 rzz10': {'statevector': np.array([0.5, 0.5j, 0.5j, 0.5])}
        }

        for name in expected_data:
            # Check snapshot
            snapshots = result.get_snapshots(name)
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = snapshots['0']['statevector'][0]
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_conditionals(self):
        filename = self._get_resource_path('qobj/cpp_conditionals.json')
        with open(filename, 'r') as file:
            q_job = QuantumJob(json.load(file),
                               backend=self.backend,
                               preformatted=True)
        result = self.backend.run(q_job).result()
        expected_data = {
            'single creg (c0=0)': {
                'statevector': np.array([1, 0, 0, 0])},
            'single creg (c0=1)': {
                'statevector': np.array([0, 0, 0, 1])},
            'two creg (c1=0)': {
                'statevector': np.array([1, 0, 0, 0])},
            'two creg (c1=1)': {
                'statevector': np.array([0, 0, 0, 1])}
        }

        for name in expected_data:
            # Check snapshot
            snapshots = result.get_snapshots(name)
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = snapshots['0']['statevector'][0]
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')


if __name__ == '__main__':
    unittest.main(verbosity=2)
