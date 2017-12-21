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
from .common import QiskitTestCase
from qiskit import QuantumProgram
from qiskit.tools.qcvv import tomography as tomo
import numpy as np


class TestTomography(QiskitTestCase):
    """Tests for tomography.py"""

    def test_marginal_counts(self):
        counts = {"00": 13, "01": 36, "10": 28, "11": 23}
        self.assertEqual(tomo.marginal_counts(counts, [0]), {"0": 41, "1": 59})
        self.assertEqual(tomo.marginal_counts(counts, [1]), {"0": 49, "1": 51})

    def test_state_tomography_sets_1qubit(self):
        tomoset1 = tomo.state_tomography_set([0], meas_basis='Pauli')
        tomoset1_default = tomo.state_tomography_set([0])
        self.assertEqual(self.state_refset1['qubits'], tomoset1['qubits'])
        self.assertEqual(self.state_refset1['circuits'], tomoset1['circuits'])
        self.assertEqual(self.state_refset1['qubits'],
                         tomoset1_default['qubits'])
        self.assertEqual(self.state_refset1['circuits'],
                         tomoset1_default['circuits'])

    def test_state_tomography_sets_2qubit(self):
        tomoset2 = tomo.state_tomography_set([0, 1], meas_basis='Pauli')
        self.assertEqual(self.state_refset2['qubits'], tomoset2['qubits'])
        self.assertEqual(self.state_refset2['circuits'], tomoset2['circuits'])

    def test_state_tomography_circuit_names_1qubit(self):
        self.assertEqual(
            tomo.tomography_circuit_names(self.state_refset1, 'test'),
            ['test_meas_X(0)', 'test_meas_Y(0)', 'test_meas_Z(0)'])

    def test_state_tomography_circuit_names_2qubit(self):
        self.assertEqual(
            tomo.tomography_circuit_names(self.state_refset2, 'test'),
            ['test_meas_X(0)X(1)', 'test_meas_X(0)Y(1)',
             'test_meas_X(0)Z(1)', 'test_meas_Y(0)X(1)',
             'test_meas_Y(0)Y(1)', 'test_meas_Y(0)Z(1)',
             'test_meas_Z(0)X(1)', 'test_meas_Z(0)Y(1)',
             'test_meas_Z(0)Z(1)'])

    def test_state_tomography_1qubit(self):
        # Get test circuits
        qp, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 1000
        threshold = 1e-2
        rho = _tomography_test_data(
            qp, 'Zp', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(rho, [1, 0], threshold))
        rho = _tomography_test_data(
            qp, 'Zm', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(rho, [0, 1], threshold))
        rho = _tomography_test_data(
            qp, 'Xp', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(
            rho, [1 / np.sqrt(2), 1 / np.sqrt(2)], threshold))
        rho = _tomography_test_data(
            qp, 'Xm', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(
            rho, [1 / np.sqrt(2), -1 / np.sqrt(2)], threshold))
        rho = _tomography_test_data(
            qp, 'Yp', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(
            rho, [1 / np.sqrt(2), 1j / np.sqrt(2)], threshold))
        rho = _tomography_test_data(
            qp, 'Ym', qr, cr, self.state_refset1, shots)
        self.assertTrue(_tomography_test_fit(
            rho, [1 / np.sqrt(2), -1j / np.sqrt(2)], threshold))

    def test_state_tomography_2qubit(self):
        # Get test circuits
        qp, qr, cr = _test_circuits_2qubit()
        shots = 1000
        threshold = 1e-2
        # Test simulation and fitting
        rho = _tomography_test_data(
            qp, 'Bell', qr, cr, self.state_refset2, shots)
        self.assertTrue(_tomography_test_fit(
            rho, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], threshold))

    def test_process_tomography_sets_1qubit(self):
        tomoset1_default = tomo.process_tomography_set([0])
        tomoset1_sic = tomo.process_tomography_set(
            [0], meas_basis='Pauli', prep_basis='SIC')
        tomoset1_pauli = tomo.process_tomography_set(
            [0], meas_basis='Pauli', prep_basis='Pauli')
        self.assertEqual(self.process_refset1_sic['qubits'],
                         tomoset1_default['qubits'])
        self.assertEqual(self.process_refset1_sic['circuits'],
                         tomoset1_default['circuits'])
        self.assertEqual(self.process_refset1_sic['qubits'],
                         tomoset1_sic['qubits'])
        self.assertEqual(self.process_refset1_sic['circuits'],
                         tomoset1_sic['circuits'])
        self.assertEqual(self.process_refset1_pauli['qubits'],
                         tomoset1_pauli['qubits'])
        self.assertEqual(self.process_refset1_pauli['circuits'],
                         tomoset1_pauli['circuits'])

    def test_process_tomography_circuit_names_1qubit(self):
        self.assertEqual(
            tomo.tomography_circuit_names(self.process_refset1_sic, 'test'),
            ['test_prep_S0(0)_meas_X(0)', 'test_prep_S0(0)_meas_Y(0)',
             'test_prep_S0(0)_meas_Z(0)', 'test_prep_S1(0)_meas_X(0)',
             'test_prep_S1(0)_meas_Y(0)', 'test_prep_S1(0)_meas_Z(0)',
             'test_prep_S2(0)_meas_X(0)', 'test_prep_S2(0)_meas_Y(0)',
             'test_prep_S2(0)_meas_Z(0)', 'test_prep_S3(0)_meas_X(0)',
             'test_prep_S3(0)_meas_Y(0)', 'test_prep_S3(0)_meas_Z(0)'])
        self.assertEqual(
            tomo.tomography_circuit_names(self.process_refset1_pauli, 'test'),
            ['test_prep_X0(0)_meas_X(0)', 'test_prep_X0(0)_meas_Y(0)',
             'test_prep_X0(0)_meas_Z(0)', 'test_prep_X1(0)_meas_X(0)',
             'test_prep_X1(0)_meas_Y(0)', 'test_prep_X1(0)_meas_Z(0)',
             'test_prep_Y0(0)_meas_X(0)', 'test_prep_Y0(0)_meas_Y(0)',
             'test_prep_Y0(0)_meas_Z(0)', 'test_prep_Y1(0)_meas_X(0)',
             'test_prep_Y1(0)_meas_Y(0)', 'test_prep_Y1(0)_meas_Z(0)',
             'test_prep_Z0(0)_meas_X(0)', 'test_prep_Z0(0)_meas_Y(0)',
             'test_prep_Z0(0)_meas_Z(0)', 'test_prep_Z1(0)_meas_X(0)',
             'test_prep_Z1(0)_meas_Y(0)', 'test_prep_Z1(0)_meas_Z(0)'])

    def test_process_tomography_1qubit(self):
        # Get test circuits
        qp, qr, cr = _test_circuits_1qubit()
        # Test simulation and fitting
        shots = 1000
        threshold = 1e-2
        choi = _tomography_test_data(
            qp, 'Zp', qr, cr, self.process_refset1_sic, shots)
        self.assertTrue(_tomography_test_fit(
            choi, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], threshold))
        choi = _tomography_test_data(
            qp, 'Zm', qr, cr, self.process_refset1_sic, shots)
        self.assertTrue(_tomography_test_fit(
            choi, [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], threshold))
        choi = _tomography_test_data(
            qp, 'Xp', qr, cr, self.process_refset1_sic, shots)
        self.assertTrue(_tomography_test_fit(
            choi, [0.5, 0.5, 0.5, -0.5], threshold))

    # Class member variables
    state_refset1 = {
        'qubits': [0],
        'circuits': [
            {'meas': {0: 'X'}},
            {'meas': {0: 'Y'}},
            {'meas': {0: 'Z'}}
        ],
        'meas_basis': tomo.PAULI_BASIS
    }

    state_refset2 = {
        'qubits': [0, 1],
        'circuits': [
            {'meas': {0: 'X', 1: 'X'}},
            {'meas': {0: 'X', 1: 'Y'}},
            {'meas': {0: 'X', 1: 'Z'}},
            {'meas': {0: 'Y', 1: 'X'}},
            {'meas': {0: 'Y', 1: 'Y'}},
            {'meas': {0: 'Y', 1: 'Z'}},
            {'meas': {0: 'Z', 1: 'X'}},
            {'meas': {0: 'Z', 1: 'Y'}},
            {'meas': {0: 'Z', 1: 'Z'}}
        ],
        'meas_basis': tomo.PAULI_BASIS
    }

    process_refset1_pauli = {
        'qubits': [0],
        'meas_basis': tomo.PAULI_BASIS,
        'prep_basis': tomo.PAULI_BASIS,
        'circuits': [
            {'meas': {0: 'X'}, 'prep': {0: ('X', 0)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('X', 0)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('X', 0)}},
            {'meas': {0: 'X'}, 'prep': {0: ('X', 1)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('X', 1)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('X', 1)}},
            {'meas': {0: 'X'}, 'prep': {0: ('Y', 0)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('Y', 0)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('Y', 0)}},
            {'meas': {0: 'X'}, 'prep': {0: ('Y', 1)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('Y', 1)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('Y', 1)}},
            {'meas': {0: 'X'}, 'prep': {0: ('Z', 0)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('Z', 0)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('Z', 0)}},
            {'meas': {0: 'X'}, 'prep': {0: ('Z', 1)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('Z', 1)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('Z', 1)}}
        ]
    }

    process_refset1_sic = {
        'qubits': [0],
        'meas_basis': tomo.PAULI_BASIS,
        'prep_basis': tomo.SIC_BASIS,
        'circuits': [
            {'meas': {0: 'X'}, 'prep': {0: ('S', 0)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('S', 0)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('S', 0)}},
            {'meas': {0: 'X'}, 'prep': {0: ('S', 1)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('S', 1)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('S', 1)}},
            {'meas': {0: 'X'}, 'prep': {0: ('S', 2)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('S', 2)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('S', 2)}},
            {'meas': {0: 'X'}, 'prep': {0: ('S', 3)}},
            {'meas': {0: 'Y'}, 'prep': {0: ('S', 3)}},
            {'meas': {0: 'Z'}, 'prep': {0: ('S', 3)}}
        ]
    }


def _tomography_test_data(qp, name, qr, cr, tomoset, shots):
    tomo.create_tomography_circuits(qp, name, qr, cr, tomoset)
    result = qp.execute(tomo.tomography_circuit_names(
        tomoset, name), shots=shots, seed=42)
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
    return qp, qr, cr


if __name__ == '__main__':
    unittest.main()
