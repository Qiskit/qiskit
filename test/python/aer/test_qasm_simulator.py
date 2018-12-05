# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,redefined-builtin

import json
import unittest

import numpy as np
from numpy.linalg import norm

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.backends.aer.qasm_simulator import (QasmSimulator,
                                                cx_error_matrix,
                                                x90_error_matrix)
from qiskit.qobj import Qobj
from qiskit.result.postprocess import format_statevector
from qiskit.quantum_info import state_fidelity
from qiskit import compile
from ..common import QiskitTestCase, Path
from ..common import requires_cpp_simulator


class TestAerQasmSimulator(QiskitTestCase):
    """
    Test job_processor module.
    """

    @requires_cpp_simulator
    def setUp(self):
        self.backend = QasmSimulator()

        qasm_file_name = 'example.qasm'
        qasm_file_path = self._get_resource_path(
            'qasm/' + qasm_file_name, Path.TEST)
        self.qc1 = QuantumCircuit.from_qasm_file(qasm_file_path)

        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        self.qc2 = QuantumCircuit(qr, cr)
        self.qc2.h(qr[0])
        self.qc2.measure(qr[0], cr[0])

        self.qobj = compile([self.qc1, self.qc2], backend=self.backend,
                            shots=2000, seed=1111)

    def test_x90_coherent_error_matrix(self):
        x90 = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        u_matrix = x90_error_matrix(0., 0.).dot(x90)
        target = x90
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10,
                               msg="identity error matrix")
        u_matrix = x90_error_matrix(np.pi / 2., 0.).dot(x90)
        target = -1j * np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10)
        u_matrix = x90_error_matrix(0., np.pi / 2.).dot(x90)
        target = np.array([[1., -1], [1, 1.]]) / np.sqrt(2.)
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10)
        u_matrix = x90_error_matrix(np.pi / 2, np.pi / 2.).dot(x90)
        target = np.array([[0., -1], [1, 0.]])
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10)
        u_matrix = x90_error_matrix(0.02, -0.03)
        self.assertAlmostEqual(norm(u_matrix.dot(u_matrix.conj().T) - np.eye(2)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_cx_coherent_error_matrix(self):
        cx_matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        u_matrix = cx_error_matrix(0., 0.).dot(cx_matrix)
        target = cx_matrix
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10,
                               msg="identity error matrix")
        u_matrix = cx_error_matrix(np.pi / 2., 0.).dot(cx_matrix)
        target = np.array([[1, 0, 1j, 0],
                           [0, -1j, 0, 1],
                           [1j, 0, 1, 0],
                           [0, 1, 0, -1j]]) / np.sqrt(2)
        self.assertAlmostEqual(norm(u_matrix - target), 0.0, places=10)
        u_matrix = cx_error_matrix(0.03, -0.04)
        self.assertAlmostEqual(norm(u_matrix.dot(u_matrix.conj().T) - np.eye(4)), 0.0,
                               places=10, msg="Test error matrix is unitary")

    def test_run_qobj(self):
        result = self.backend.run(self.qobj).result()
        shots = self.qobj.config.shots
        threshold = 0.04 * shots
        counts = result.get_counts(self.qc1)
        target = {'100 100': shots / 8, '011 011': shots / 8,
                  '101 101': shots / 8, '111 111': shots / 8,
                  '000 000': shots / 8, '010 010': shots / 8,
                  '110 110': shots / 8, '001 001': shots / 8}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_qobj_measure_opt(self):
        filename = self._get_resource_path('qobj/cpp_measure_opt.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
        shots = qobj.config.shots
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
            snapshots = result.data(name)['snapshots']['statevector']
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = format_statevector(snapshots['0'][0])
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_measure_opt_flag(self):
        filename = self._get_resource_path('qobj/cpp_measure_opt_flag.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
        shots = qobj.config.shots
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
            snapshots = result.data(name)['snapshots']['statevector']
            # Check snapshot keys
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            # Check number of snapshots
            # there should be 1 for measurement sampling optimization
            # and there should be >1 for each shot being simulated.
            num_snapshots = len(snapshots['0'])
            if sampled_measurements[name] is True:
                self.assertEqual(num_snapshots, 1,
                                 msg=name + ' snapshot length')
            else:
                self.assertEqual(num_snapshots, shots,
                                 msg=name + ' snapshot length')

    def test_qobj_reset(self):
        filename = self._get_resource_path('qobj/cpp_reset.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
        expected_data = {
            'reset': {'statevector': np.array([1, 0])},
            'x reset': {'statevector': np.array([1, 0])},
            'y reset': {'statevector': np.array([1, 0])},
            'h reset': {'statevector': np.array([1, 0])}
        }
        for name in expected_data:
            # Check snapshot is |0> state
            snapshots = result.data(name)['snapshots']['statevector']
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = format_statevector(snapshots['0'][0])
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_save_load(self):
        filename = self._get_resource_path('qobj/cpp_save_load.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()

        snapshots = result.data('save_command')['snapshots']['statevector']
        self.assertEqual(set(snapshots), {'0', '1', '10', '11'},
                         msg='snapshot keys')
        state0 = format_statevector(snapshots['0'][0])
        state10 = format_statevector(snapshots['10'][0])
        state1 = format_statevector(snapshots['1'][0])
        state11 = format_statevector(snapshots['11'][0])

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
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
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
            snapshots = result.data(name)['snapshots']['statevector']
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = format_statevector(snapshots['0'][0])
            expected_state = expected_data[name]['statevector']
            inner_product = expected_state.dot(state.conj())
            self.assertAlmostEqual(inner_product, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_qobj_two_qubit_gates(self):
        filename = self._get_resource_path('qobj/cpp_two_qubit_gates.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
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
            snapshots = result.data(name)['snapshots']['statevector']
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = format_statevector(snapshots['0'][0])
            expected_state = expected_data[name]['statevector']
            fidelity = state_fidelity(expected_state, state)
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_conditionals(self):
        filename = self._get_resource_path('qobj/cpp_conditionals.json')
        with open(filename, 'r') as file:
            qobj = Qobj.from_dict(json.load(file))
        result = self.backend.run(qobj).result()
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
            snapshots = result.data(name)['snapshots']['statevector']
            self.assertEqual(set(snapshots), {'0'},
                             msg=name + ' snapshot keys')
            self.assertEqual(len(snapshots['0']), 1,
                             msg=name + ' snapshot length')
            state = format_statevector(snapshots['0'][0])
            expected_state = expected_data[name]['statevector']
            fidelity = np.abs(expected_state.dot(state.conj())) ** 2
            self.assertAlmostEqual(fidelity, 1.0, places=10,
                                   msg=name + ' snapshot fidelity')

    def test_memory(self):
        qr = QuantumRegister(4, 'qr')
        cr0 = ClassicalRegister(2, 'cr0')
        cr1 = ClassicalRegister(2, 'cr1')
        circ = QuantumCircuit(qr, cr0, cr1)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[3])
        circ.measure(qr[0], cr0[0])
        circ.measure(qr[1], cr0[1])
        circ.measure(qr[2], cr1[0])
        circ.measure(qr[3], cr1[1])

        shots = 50
        qobj = compile(circ, backend=self.backend, shots=shots, memory=True)
        result = self.backend.run(qobj).result()
        memory = result.get_memory()
        self.assertEqual(len(memory), shots)
        for mem in memory:
            self.assertIn(mem, ['10 00', '10 11'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
