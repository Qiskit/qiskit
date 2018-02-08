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
        qp, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        rho = _tomography_test_data(qp, 'Zp', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [1, 0], threshold))
        rho = _tomography_test_data(qp, 'Zm', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 1], threshold))
        rho = _tomography_test_data(qp, 'Xp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qp, 'Xm', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qp, 'Yp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 1j / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qp, 'Ym', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), -1j / np.sqrt(2)],
                                 threshold))

    def test_state_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.state_tomography_set([0, 1])
        # Get test circuits
        qp, qr, cr = _test_circuits_2qubit()
        shots = 2000
        threshold = 1e-2
        # Test simulation and fitting
        rho = _tomography_test_data(qp, 'Bell', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(rho, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        rho = _tomography_test_data(qp, 'X1Id0', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 0, 1, 0], threshold))

    def test_process_tomography_1qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0])
        # Get test circuits
        qp, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 2000
        threshold = 1e-2
        choi = _tomography_test_data(qp, 'Zp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                                 threshold))
        choi = _tomography_test_data(qp, 'Xp', qr, cr, tomo_set, shots)
        self.assertTrue(
            _tomography_test_fit(choi, [0.5, 0.5, 0.5, -0.5], threshold))

    def test_process_tomography_2qubit(self):
        # Tomography set
        tomo_set = tomo.process_tomography_set([0, 1])
        # Get test circuits
        qp, qr, cr = _test_circuits_2qubit()
        # Test simulation and fitting
        shots = 1000
        threshold = 1e-2
        ref_X1Id0 = np.array(
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]) / 2
        choi = _tomography_test_data(qp, 'X1Id0', qr, cr, tomo_set, shots)
        self.assertTrue(_tomography_test_fit(choi, ref_X1Id0, threshold))


def _tomography_test_data(qp, name, qr, cr, tomoset, shots):
    tomo.create_tomography_circuits(qp, name, qr, cr, tomoset)
    result = qp.execute(tomo.tomography_circuit_names(tomoset, name),
                        shots=shots, seed=42, timeout=180)
    data = tomo.tomography_data(result, name, tomoset)
    return tomo.fit_tomography_data(data)


def _tomography_test_fit(fitted_rho, ref_state_vector, threshold=1e-2):
    fidelity = fitted_rho.dot(ref_state_vector).conj().dot(ref_state_vector)
    fidelity = np.sqrt(fidelity)
    return np.abs(1.0 - fidelity) < threshold


def _test_circuits_1qubit():
    qp = QuantumProgram()
    qr = qp.create_quantum_register('qr', 1)
    cr = qp.create_classical_register('cr', 1)

    # Test Circuits Z eigenstate
    circ = qp.create_circuit('Zp', [qr], [cr])
    circ = qp.create_circuit('Zm', [qr], [cr])
    circ.x(qr[0])
    # Test Circuits X eigenstate
    circ = qp.create_circuit('Xp', [qr], [cr])
    circ.h(qr[0])
    circ = qp.create_circuit('Xm', [qr], [cr])
    circ.h(qr[0])
    circ.z(qr[0])
    # Test Circuits Y eigenstate
    circ = qp.create_circuit('Yp', [qr], [cr])
    circ.h(qr[0])
    circ.s(qr[0])
    circ = qp.create_circuit('Ym', [qr], [cr])
    circ.h(qr[0])
    circ.s(qr[0])
    circ.z(qr[0])
    return qp, qr, cr


def _test_circuits_2qubit():
    qp = QuantumProgram()
    qr = qp.create_quantum_register('qr', 2)
    cr = qp.create_classical_register('cr', 2)

    # Test Circuits Bell state
    circ = qp.create_circuit('Bell', [qr], [cr])
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])
    circ = qp.create_circuit('X1Id0', [qr], [cr])
    circ.x(qr[1])
    return qp, qr, cr


if __name__ == '__main__':
    unittest.main()
