# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Test of QCVV/tomography module."""

import unittest
import numpy as np

from qiskit import execute, BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.qcvv import tomography as tomo
from qiskit.test import QiskitTestCase


class TestTomography(QiskitTestCase):
    """Tests for tomography.py"""

    def test_marginal_counts(self):
        counts = {"00": 13, "01": 36, "10": 28, "11": 23}
        self.assertEqual(tomo.marginal_counts(counts, [0]), {"0": 41, "1": 59})
        self.assertEqual(tomo.marginal_counts(counts, [1]), {"0": 49, "1": 51})

    def test_state_tomography_set_default(self):
        pauli_set = tomo.state_tomography_set([0], meas_basis='Pauli')
        default_set = tomo.state_tomography_set([0])
        self.assertEqual(pauli_set['circuits'], default_set['circuits'])

    def test_process_tomography_set_default(self):
        tomo_set = tomo.process_tomography_set([0])
        default_set = tomo.process_tomography_set(
            [0], meas_basis='Pauli', prep_basis='SIC')
        self.assertEqual(tomo_set['circuits'], default_set['circuits'])

    def test_state_tomography_1qubit(self):
        # Tomography set
        tomo_set = tomo.state_tomography_set([0])
        # Get test circuits
        circuits, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        rho = _tomography_test_data(circuits['Zp'], qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [1, 0], threshold))
        rho = _tomography_test_data(circuits['Zm'], qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 1], threshold))
        rho = _tomography_test_data(circuits['Xp'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(circuits['Xm'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(circuits['Yp'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1j / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(circuits['Ym'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1j / np.sqrt(2)],
                                 threshold))

    def test_state_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.state_tomography_set([0, 1])
        # Get test circuits
        circuits, qr, cr = _test_circuits_2qubit()
        shots = 2000
        threshold = 1e-2
        # Test simulation and fitting
        rho = _tomography_test_data(circuits['Bell'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(circuits['X1Id0'], qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 0, 1, 0], threshold))

    def test_process_tomography_1qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0])
        # Get test circuits
        circuits, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        choi = _tomography_test_data(circuits['Zp'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        choi = _tomography_test_data(circuits['Xp'], qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [0.5, 0.5, 0.5, -0.5], threshold))

    def test_process_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0, 1])
        # Get test circuits
        circuits, qr, cr = _test_circuits_2qubit()
        # Test simulation and fitting
        shots = 1000
        threshold = 0.015
        ref_x1_id0 = np.array(
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]) / 2
        choi = _tomography_test_data(circuits['X1Id0'], qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(choi, ref_x1_id0, threshold))


def _tomography_test_data(circuit, qr, cr, tomoset, shots):
    tomo_circs = tomo.create_tomography_circuits(circuit, qr, cr, tomoset)
    result = execute(tomo_circs,
                     BasicAer.get_backend('qasm_simulator'),
                     shots=shots,
                     seed=42).result()
    data = tomo.tomography_data(result, circuit.name, tomoset)
    return tomo.fit_tomography_data(data)


def _tomography_test_fit(fitted_rho, ref_state_vector, threshold=1e-2):
    fidelity = fitted_rho.dot(ref_state_vector).conj().dot(ref_state_vector)
    fidelity = np.sqrt(fidelity)
    return np.abs(1.0 - fidelity) < threshold


def _test_circuits_1qubit():
    circuits = {}
    qr = QuantumRegister(1, name='qr')
    cr = ClassicalRegister(1, name='cr')

    # Test Circuits Z eigenstate
    tmp = QuantumCircuit(qr, cr, name='Zp')
    circuits['Zp'] = tmp
    tmp = QuantumCircuit(qr, cr, name='Zm')
    tmp.x(qr[0])
    circuits['Zm'] = tmp
    # Test Circuits X eigenstate
    tmp = QuantumCircuit(qr, cr, name='Xp')
    tmp.h(qr[0])
    circuits['Xp'] = tmp
    tmp = QuantumCircuit(qr, cr, name='Xm')
    tmp.h(qr[0])
    tmp.z(qr[0])
    circuits['Xm'] = tmp
    # Test Circuits Y eigenstate
    tmp = QuantumCircuit(qr, cr, name='Yp')
    tmp.h(qr[0])
    tmp.s(qr[0])
    circuits['Yp'] = tmp
    tmp = QuantumCircuit(qr, cr, name='Ym')
    tmp.h(qr[0])
    tmp.s(qr[0])
    tmp.z(qr[0])
    circuits['Ym'] = tmp
    return circuits, qr, cr


def _test_circuits_2qubit():
    circuits = {}
    qr = QuantumRegister(2, name='qr')
    cr = ClassicalRegister(2, name='cr')

    # Test Circuits Bell state
    tmp = QuantumCircuit(qr, cr, name='Bell')
    tmp.h(qr[0])
    tmp.cx(qr[0], qr[1])
    circuits['Bell'] = tmp
    tmp = QuantumCircuit(qr, cr, name='X1Id0')
    tmp.x(qr[1])
    circuits['X1Id0'] = tmp
    return circuits, qr, cr


if __name__ == '__main__':
    unittest.main()
