# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test the qi tools modules."""

import unittest
from copy import deepcopy

import numpy as np

from qiskit.tools.qi.pauli import Pauli, random_pauli, inverse_pauli, \
    pauli_group, sgn_prod
from qiskit.tools.qi.qi import partial_trace, vectorize, devectorize, outer
from qiskit.tools.qi.qi import state_fidelity, purity, concurrence
from .common import QiskitTestCase


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

    def test_state_fidelity(self):
        psi1 = [0.70710678118654746, 0, 0, 0.70710678118654746]
        psi2 = [0., 0.70710678118654746, 0.70710678118654746, 0.]
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        mix = [[0.25, 0, 0, 0], [0, 0.25, 0, 0],
               [0, 0, 0.25, 0], [0, 0, 0, 0.25]]
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, rho1), 1.0, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi1, mix), 0.25, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi2, rho1), 0.0, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi2, mix), 0.25, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(rho1, psi1), 1.0, places=7,
                               msg='matrix-vector input')
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7,
                               msg='matrix-matrix input')

    def test_purity(self):
        rho1 = [[1, 0], [0, 0]]
        rho2 = [[0.5, 0], [0, 0.5]]
        rho3 = 0.7 * np.array(rho1) + 0.3 * np.array(rho2)
        test_pass = (purity(rho1) == 1.0 and
                     purity(rho2) == 0.5 and
                     round(purity(rho3), 10) == 0.745)
        self.assertTrue(test_pass)

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


class TestPauli(QiskitTestCase):
    """Tests for Pauli class"""

    def setUp(self):
        v = np.zeros(3)
        w = np.zeros(3)
        v[0] = 1
        w[1] = 1
        v[2] = 1
        w[2] = 1

        self.p3 = Pauli(v, w)

    def test_random_pauli5(self):
        length = 2
        q = random_pauli(length)
        self.log.info(q)
        self.assertEqual(q.numberofqubits, length)
        self.assertEqual(len(q.v), length)
        self.assertEqual(len(q.w), length)
        self.assertEqual(len(q.to_label()), length)
        self.assertEqual(len(q.to_matrix()), 2 ** length)

    def test_pauli_invert(self):
        self.log.info("===== p3 =====")
        self.log.info(self.p3)
        self.assertEqual(str(self.p3), 'v = 1.0\t0.0\t1.0\t\nw = 0.0\t1.0\t1.0\t')

        self.log.info("\tIn label form:")
        self.log.info(self.p3.to_label())
        self.assertEqual(self.p3.to_label(), 'ZXY')

        self.log.info("\tIn matrix form:")
        self.log.info(self.p3.to_matrix())
        m = np.array([
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. - 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. - 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]])
        self.assertTrue((self.p3.to_matrix() == m).all())

        self.log.info("===== r =====")
        r = inverse_pauli(self.p3)
        self.assertEqual(str(r), 'v = 1.0\t0.0\t1.0\t\nw = 0.0\t1.0\t1.0\t')

        self.log.info("In label form:")
        self.log.info(r.to_label())
        self.assertEqual(r.to_label(), 'ZXY')

        self.log.info("\tIn matrix form:")
        self.assertTrue((r.to_matrix() == m).all())

    def test_pauli_group(self):
        self.log.info("Group in tensor order:")
        expected = ['III', 'XII', 'YII', 'ZII', 'IXI', 'XXI', 'YXI', 'ZXI', 'IYI', 'XYI', 'YYI',
                    'ZYI', 'IZI', 'XZI', 'YZI', 'ZZI', 'IIX', 'XIX', 'YIX', 'ZIX', 'IXX', 'XXX',
                    'YXX', 'ZXX', 'IYX', 'XYX', 'YYX', 'ZYX', 'IZX', 'XZX', 'YZX', 'ZZX', 'IIY',
                    'XIY', 'YIY', 'ZIY', 'IXY', 'XXY', 'YXY', 'ZXY', 'IYY', 'XYY', 'YYY', 'ZYY',
                    'IZY', 'XZY', 'YZY', 'ZZY', 'IIZ', 'XIZ', 'YIZ', 'ZIZ', 'IXZ', 'XXZ', 'YXZ',
                    'ZXZ', 'IYZ', 'XYZ', 'YYZ', 'ZYZ', 'IZZ', 'XZZ', 'YZZ', 'ZZZ']
        grp = pauli_group(3, case=1)
        for j in grp:
            self.log.info('==== j (tensor order) ====')
            self.log.info(j.to_label())
            self.assertEqual(expected.pop(0), j.to_label())

        self.log.info("Group in weight order:")
        expected = ['III', 'XII', 'YII', 'ZII', 'IXI', 'IYI', 'IZI', 'IIX', 'IIY', 'IIZ', 'XXI',
                    'YXI', 'ZXI', 'XYI', 'YYI', 'ZYI', 'XZI', 'YZI', 'ZZI', 'XIX', 'YIX', 'ZIX',
                    'IXX', 'IYX', 'IZX', 'XIY', 'YIY', 'ZIY', 'IXY', 'IYY', 'IZY', 'XIZ', 'YIZ',
                    'ZIZ', 'IXZ', 'IYZ', 'IZZ', 'XXX', 'YXX', 'ZXX', 'XYX', 'YYX', 'ZYX', 'XZX',
                    'YZX', 'ZZX', 'XXY', 'YXY', 'ZXY', 'XYY', 'YYY', 'ZYY', 'XZY', 'YZY', 'ZZY',
                    'XXZ', 'YXZ', 'ZXZ', 'XYZ', 'YYZ', 'ZYZ', 'XZZ', 'YZZ', 'ZZZ']
        grp = pauli_group(3)
        for j in grp:
            self.log.info('==== j (weight order) ====')
            self.log.info(j.to_label())
            self.assertEqual(expected.pop(0), j.to_label())

    def test_pauli_sgn_prod(self):
        p1 = Pauli(np.array([0]), np.array([1]))
        p2 = Pauli(np.array([1]), np.array([1]))

        self.log.info("sign product:")
        p3, sgn = sgn_prod(p1, p2)
        self.log.info("p1: %s", p1.to_label())
        self.log.info("p2: %s", p2.to_label())
        self.log.info("p3: %s", p3.to_label())
        self.log.info("sgn_prod(p1, p2): %s", str(sgn))
        self.assertEqual(p1.to_label(), 'X')
        self.assertEqual(p2.to_label(), 'Y')
        self.assertEqual(p3.to_label(), 'Z')
        self.assertEqual(sgn, 1j)

        self.log.info("sign product reverse:")
        p3, sgn = sgn_prod(p2, p1)
        self.log.info("p2: %s", p2.to_label())
        self.log.info("p1: %s", p1.to_label())
        self.log.info("p3: %s", p3.to_label())
        self.log.info("sgn_prod(p2, p1): %s", str(sgn))
        self.assertEqual(p1.to_label(), 'X')
        self.assertEqual(p2.to_label(), 'Y')
        self.assertEqual(p3.to_label(), 'Z')
        self.assertEqual(sgn, -1j)

    def test_equality_equal(self):
        """Test equality operator: equal Paulis"""
        p1 = self.p3
        p2 = deepcopy(p1)

        self.log.info(p1 == p2)
        self.assertTrue(p1 == p2)

        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.assertEqual(p1.to_label(), 'ZXY')
        self.assertEqual(p2.to_label(), 'ZXY')

    def test_equality_different(self):
        """Test equality operator: different Paulis"""
        p1 = self.p3
        p2 = deepcopy(p1)

        p2.v[0] = (p1.v[0] + 1) % 2
        self.log.info(p1 == p2)
        self.assertFalse(p1 == p2)

        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.assertEqual(p1.to_label(), 'ZXY')
        self.assertEqual(p2.to_label(), 'IXY')

    def test_inequality_equal(self):
        """Test inequality operator: equal Paulis"""
        p1 = self.p3
        p2 = deepcopy(p1)

        self.log.info(p1 != p2)
        self.assertFalse(p1 != p2)

        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.assertEqual(p1.to_label(), 'ZXY')
        self.assertEqual(p2.to_label(), 'ZXY')

    def test_inequality_different(self):
        """Test inequality operator: different Paulis"""
        p1 = self.p3
        p2 = deepcopy(p1)

        p2.v[0] = (p1.v[0] + 1) % 2
        self.log.info(p1 != p2)
        self.assertTrue(p1 != p2)

        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.assertEqual(p1.to_label(), 'ZXY')
        self.assertEqual(p2.to_label(), 'IXY')


if __name__ == '__main__':
    unittest.main()
