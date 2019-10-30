# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

"""Quick program to test the qi tools modules."""

import unittest
from unittest.mock import Mock, call, patch
import math
from io import StringIO
import numpy as np

from qiskit.tools.qi.qi import partial_trace, vectorize, devectorize, outer
from qiskit.tools.qi.qi import concurrence, qft, chop
from qiskit.tools.qi.qi import shannon_entropy, entropy, mutual_information
from qiskit.tools.qi.qi import choi_to_rauli
from qiskit.tools.qi.qi import entanglement_of_formation, is_pos_def
from qiskit.tools.qi.qi import __eof_qubit as eof_qubit
from qiskit.quantum_info import purity
from qiskit.quantum_info.random import random_density_matrix
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase


class TestQI(QiskitTestCase):
    """Tests for qi.py"""

    def test_partial_trace(self):
        # reference
        rho0 = [[0.5, 0.5], [0.5, 0.5]]
        rho1 = [[1, 0], [0, 0]]
        rho2 = [[0, 0], [0, 1]]
        rho10 = np.kron(rho1, rho0)
        rho20 = np.kron(rho2, rho0)
        rho21 = np.kron(rho2, rho1)
        rho210 = np.kron(rho21, rho0)
        rhos = [rho0, rho1, rho2, rho10, rho20, rho21]

        # test partial trace
        tau0 = partial_trace(rho210, [1, 2])
        tau1 = partial_trace(rho210, [0, 2])
        tau2 = partial_trace(rho210, [0, 1])

        # test different dimensions
        tau10 = partial_trace(rho210, [1], dimensions=[4, 2])
        tau20 = partial_trace(rho210, [1], dimensions=[2, 2, 2])
        tau21 = partial_trace(rho210, [0], dimensions=[2, 4])
        taus = [tau0, tau1, tau2, tau10, tau20, tau21]

        all_pass = True
        for i, j in zip(rhos, taus):
            all_pass &= (np.linalg.norm(i - j) == 0)
        self.assertTrue(all_pass)

    def test_vectorize(self):
        mat = [[1, 2], [3, 4]]
        col = [1, 3, 2, 4]
        row = [1, 2, 3, 4]
        paul = [5, 5, -1j, -3]
        test_pass = (np.linalg.norm(vectorize(mat) - col) == 0 and
                     np.linalg.norm(vectorize(mat, method='col') - col) == 0 and
                     np.linalg.norm(vectorize(mat, method='row') - row) == 0 and
                     np.linalg.norm(vectorize(mat, method='pauli') - paul) == 0)
        self.assertTrue(test_pass)

    def test_devectorize(self):
        mat = [[1, 2], [3, 4]]
        col = [1, 3, 2, 4]
        row = [1, 2, 3, 4]
        paul = [5, 5, -1j, -3]
        test_pass = (np.linalg.norm(devectorize(col) - mat) == 0 and
                     np.linalg.norm(devectorize(col, method='col') - mat) == 0 and
                     np.linalg.norm(devectorize(row, method='row') - mat) == 0 and
                     np.linalg.norm(devectorize(paul, method='pauli') - mat) == 0)
        self.assertTrue(test_pass)

    def test_outer(self):
        v_z = [1, 0]
        v_y = [1, 1j]
        rho_z = [[1, 0], [0, 0]]
        rho_y = [[1, -1j], [1j, 1]]
        op_zy = [[1, -1j], [0, 0]]
        op_yz = [[1, 0], [1j, 0]]
        test_pass = (np.linalg.norm(outer(v_z) - rho_z) == 0 and
                     np.linalg.norm(outer(v_y) - rho_y) == 0 and
                     np.linalg.norm(outer(v_y, v_z) - op_yz) == 0 and
                     np.linalg.norm(outer(v_z, v_y) - op_zy) == 0)
        self.assertTrue(test_pass)

    def test_purity(self):
        rho1 = [[1, 0], [0, 0]]
        rho2 = [[0.5, 0], [0, 0.5]]
        rho3 = 0.7 * np.array(rho1) + 0.3 * np.array(rho2)
        test_pass = (purity(rho1) == 1.0 and
                     purity(rho2) == 0.5 and
                     round(purity(rho3), 10) == 0.745)
        self.assertTrue(test_pass)

    def test_purity_1d_input(self):
        input_state = [1, 0]
        res = purity(input_state)
        self.assertEqual(1, res)

    def test_concurrence(self):
        psi1 = [1, 0, 0, 0]
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        rho2 = [[0, 0, 0, 0], [0, 0.5, -0.5j, 0],
                [0, 0.5j, 0.5, 0], [0, 0, 0, 0]]
        rho3 = 0.5 * np.array(rho1) + 0.5 * np.array(rho2)
        rho4 = 0.75 * np.array(rho1) + 0.25 * np.array(rho2)
        test_pass = (concurrence(psi1) == 0.0 and
                     concurrence(rho1) == 1.0 and
                     concurrence(rho2) == 1.0 and
                     concurrence(rho3) == 0.0 and
                     concurrence(rho4) == 0.5)
        self.assertTrue(test_pass)

    def test_concurrence_not_two_qubits(self):
        input_state = np.array([[0, 1], [1, 0]])
        self.assertRaises(Exception, concurrence, input_state)

    def test_qft(self):
        num_qbits = 3
        circuit = Mock()
        q = list(range(num_qbits))
        qft(circuit, q, num_qbits)
        self.assertEqual([call(0), call(1), call(2)], circuit.h.mock_calls)
        expected_calls = [call(math.pi / 2.0, 1, 0),
                          call(math.pi / 4.0, 2, 0),
                          call(math.pi / 2.0, 2, 1)]
        self.assertEqual(expected_calls, circuit.cu1.mock_calls)

    def test_chop(self):
        array_in = [1.023, 1.0456789, 0.0000001, 0.1]
        res = chop(array_in, epsilon=1e-3)
        for i, expected in enumerate([1.023, 1.0456789, 0.0, 0.1]):
            self.assertEqual(expected, res[i])

    def test_chop_imaginary(self):
        array_in = np.array([0.000456789+0.0004j, 1.0456789, 4+0.00004j,
                             0.0000742+3j, 0.000002, 2+6j])
        res = chop(array_in, epsilon=1e-3)
        for i, expected in enumerate([0.0+0.0j, 1.0456789, 4+0.0j, 0+3j,
                                      0.0, 2+6j]):
            self.assertEqual(expected, res[i])

    def test_shannon_entropy(self):
        input_pvec = np.array([0.5, 0.3, 0.07, 0.1, 0.03])
        # Base 2
        self.assertAlmostEqual(1.7736043871504037,
                               shannon_entropy(input_pvec))
        # Base e
        self.assertAlmostEqual(1.229368880382052,
                               shannon_entropy(input_pvec, np.e))
        # Base 10
        self.assertAlmostEqual(0.533908120973504,
                               shannon_entropy(input_pvec, 10))

    def test_entropy(self):
        input_density_matrix = np.array([[0.5, 0.0], [0.0, 0.5]])
        res = entropy(input_density_matrix)
        self.assertAlmostEqual(0.6931471805599453, res)

    def test_entropy_1d(self):
        input_vector = np.array([0.5, 1, 0])
        res = entropy(input_vector)
        self.assertEqual(0, res)

    def test_mutual_information(self):
        input_state = np.array([[0.5, 0.25, 0.75, 1],
                                [1, 0, 1, 0],
                                [0.5, 0.5, 0.5, 0.5],
                                [0, 1, 0, 1]])
        res = mutual_information(input_state, 2)
        self.assertAlmostEqual(-0.15821825498448047, res)

    def test_entanglement_of_formation(self):
        input_state = np.array([[0.5, 0.25, 0.75, 1],
                                [1, 0, 1, 0],
                                [0.5, 0.5, 0.5, 0.5],
                                [0, 1, 0, 1]])
        res = entanglement_of_formation(input_state, 2)
        self.assertAlmostEqual(0.6985340217364572, res)

    def test_entanglement_of_formation_1d_input(self):
        input_state = np.array([0.5, 0.25, 0.75, 1])
        res = entanglement_of_formation(input_state, 2)
        self.assertAlmostEqual(0.15687647805861626, res)

    def test_entanglement_of_formation_invalid_input(self):
        input_state = np.array([[0, 1], [1, 0]])
        expected = "Input must be a state-vector or 2-qubit density matrix."
        with patch('sys.stdout', new=StringIO()) as fake_stout:
            res = entanglement_of_formation(input_state, 1)
        self.assertEqual(fake_stout.getvalue().strip(), expected)
        self.assertIsNone(res)

    def test__eof_qubit(self):
        input_rho = np.array([[0.5, 0.25, 0.75, 1],
                              [1, 0, 1, 0],
                              [0.5, 0.5, 0.5, 0.5],
                              [0, 1, 0, 1]])
        res = eof_qubit(input_rho)
        self.assertAlmostEqual(0.6985340217364572, res)

    def test_is_pos_def(self):
        input_x = np.array([[1, 0],
                            [0, 1]])
        res = is_pos_def(input_x)
        self.assertTrue(res)

    def test_choi_to_rauli(self):
        input_matrix = np.array([[0.5, 0.25, 0.75, 1],
                                 [1, 0, 1, 0],
                                 [0.5, 0.5, 0.5, 0.5],
                                 [0, 1, 0, 1]])
        res = choi_to_rauli(input_matrix)
        expected = np.array([[2.0+0.j, 2.25+0.0j, 0.0+0.75j, -1.0+0.0j],
                             [1.75+0.j, 2.5+0.j, 0.-1.5j, 0.75+0.0j],
                             [0.0-0.25j, 0.0+0.5j, -0.5+0.0j, 0.0-1.25j],
                             [0.0+0.j, 0.25+0.0j, 0.0-1.25j, 1.0+0.0j]])
        self.assertTrue(np.array_equal(expected, res))

    def test_random_density_matrix(self):
        random_hs_matrix = random_density_matrix(2, seed=42)
        self.assertEqual((2, 2), random_hs_matrix.shape)
        random_bures_matrix = random_density_matrix(2, method='Bures', seed=40)
        self.assertEqual((2, 2), random_bures_matrix.shape)

    def test_random_density_matrix_invalid_method(self):
        self.assertRaises(QiskitError, random_density_matrix, 2,
                          method='Special')


if __name__ == '__main__':
    unittest.main()
