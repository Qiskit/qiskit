# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Test of QCVV/tomography module."""

import unittest

import numpy as np

from qiskit import QuantumProgram
from qiskit.tools.qcvv import tomography as tomo
from .common import QiskitTestCase


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
        qprogram, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        rho = _tomography_test_data(qprogram, 'Zp', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [1, 0], threshold))
        rho = _tomography_test_data(qprogram, 'Zm', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 1], threshold))
        rho = _tomography_test_data(qprogram, 'Xp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qprogram, 'Xm', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qprogram, 'Yp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1j / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qprogram, 'Ym', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1j / np.sqrt(2)],
                                 threshold))

    def test_state_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.state_tomography_set([0, 1])
        # Get test circuits
        qprogram, qr, cr = _test_circuits_2qubit()
        shots = 2000
        threshold = 1e-2
        # Test simulation and fitting
        rho = _tomography_test_data(qprogram, 'Bell', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qprogram, 'X1Id0', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 0, 1, 0], threshold))

    def test_process_tomography_1qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0])
        # Get test circuits
        qprogram, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        choi = _tomography_test_data(qprogram, 'Zp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        choi = _tomography_test_data(qprogram, 'Xp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [0.5, 0.5, 0.5, -0.5], threshold))

    def test_process_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0, 1])
        # Get test circuits
        qprogram, qr, cr = _test_circuits_2qubit()
        # Test simulation and fitting
        shots = 1000
        threshold = 0.015
        ref_x1_id0 = np.array(
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]) / 2
        choi = _tomography_test_data(qprogram, 'X1Id0', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(choi, ref_x1_id0, threshold))


def _tomography_test_data(qprogram, name, qr, cr, tomoset, shots):
    tomo.create_tomography_circuits(qprogram, name, qr, cr, tomoset)
    result = qprogram.execute(tomo.tomography_circuit_names(tomoset, name),
                              shots=shots, seed=42, timeout=180)
    data = tomo.tomography_data(result, name, tomoset)
    return tomo.fit_tomography_data(data)


def _tomography_test_fit(fitted_rho, ref_state_vector, threshold=1e-2):
    fidelity = fitted_rho.dot(ref_state_vector).conj().dot(ref_state_vector)
    fidelity = np.sqrt(fidelity)
    return np.abs(1.0 - fidelity) < threshold


def _test_circuits_1qubit():
    qprogram = QuantumProgram()
    qr = qprogram.create_quantum_register('qr', 1)
    cr = qprogram.create_classical_register('cr', 1)

    # Test Circuits Z eigenstate
    circ = qprogram.create_circuit('Zp', [qr], [cr])
    circ = qprogram.create_circuit('Zm', [qr], [cr])
    circ.x(qr[0])
    # Test Circuits X eigenstate
    circ = qprogram.create_circuit('Xp', [qr], [cr])
    circ.h(qr[0])
    circ = qprogram.create_circuit('Xm', [qr], [cr])
    circ.h(qr[0])
    circ.z(qr[0])
    # Test Circuits Y eigenstate
    circ = qprogram.create_circuit('Yp', [qr], [cr])
    circ.h(qr[0])
    circ.s(qr[0])
    circ = qprogram.create_circuit('Ym', [qr], [cr])
    circ.h(qr[0])
    circ.s(qr[0])
    circ.z(qr[0])
    return qprogram, qr, cr


def _test_circuits_2qubit():
    qprogram = QuantumProgram()
    qr = qprogram.create_quantum_register('qr', 2)
    cr = qprogram.create_classical_register('cr', 2)

    # Test Circuits Bell state
    circ = qprogram.create_circuit('Bell', [qr], [cr])
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])
    circ = qprogram.create_circuit('X1Id0', [qr], [cr])
    circ.x(qr[1])
    return qprogram, qr, cr


if __name__ == '__main__':
    unittest.main()
