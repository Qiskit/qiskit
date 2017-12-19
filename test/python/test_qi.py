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

"""Quick program to test the Pauli class."""
from copy import deepcopy
import unittest

import numpy as np
from qiskit.tools.qi.pauli import Pauli, random_pauli, inverse_pauli, pauli_group, sgn_prod
from qiskit.tools.qi.qi import (partial_trace, vectorize, devectorize, outer,
                                state_fidelity, purity, concurrence)

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
        tau0 = partial_trace(rho210, sys=[1, 2])
        tau1 = partial_trace(rho210, sys=[0, 2])
        tau2 = partial_trace(rho210, sys=[0, 1])

        # test different dimensions
        tau10 = partial_trace(rho210, dims=[4, 2], sys=[1])
        tau20 = partial_trace(rho210, dims=[2, 2, 2], sys=[1])
        tau21 = partial_trace(rho210, dims=[2, 4], sys=[0])
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
        test_pass = np.linalg.norm(vectorize(mat) - col) == 0 and \
            np.linalg.norm(vectorize(mat, method='col') - col) == 0 and \
            np.linalg.norm(vectorize(mat, method='row') - row) == 0 and \
            np.linalg.norm(vectorize(mat, method='pauli') - paul) == 0
        self.assertTrue(test_pass)

    def test_devectorize(self):
        mat = [[1, 2], [3, 4]]
        col = [1, 3, 2, 4]
        row = [1, 2, 3, 4]
        paul = [5, 5, -1j, -3]
        test_pass = np.linalg.norm(devectorize(col) - mat) == 0 and \
            np.linalg.norm(devectorize(col, method='col') - mat) == 0 and \
            np.linalg.norm(devectorize(row, method='row') - mat) == 0 and \
            np.linalg.norm(devectorize(paul, method='pauli') - mat) == 0
        self.assertTrue(test_pass)

    def test_outer(self):
        v_z = [1, 0]
        v_y = [1, 1j]
        rho_z = [[1, 0], [0, 0]]
        rho_y = [[1, -1j], [1j, 1]]
        op_zy = [[1, -1j], [0, 0]]
        op_yz = [[1, 0], [1j, 0]]
        test_pass = np.linalg.norm(outer(v_z) - rho_z) == 0 and \
            np.linalg.norm(outer(v_y) - rho_y) == 0 and \
            np.linalg.norm(outer(v_y, v_z) - op_yz) == 0 and \
            np.linalg.norm(outer(v_z, v_y) - op_zy) == 0
        self.assertTrue(test_pass)

    def test_state_fidelity(self):
        psi1 = [0.70710678118654746, 0, 0, 0.70710678118654746]
        psi2 = [0., 0.70710678118654746, 0.70710678118654746, 0.]
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        mix = [[0.25, 0, 0, 0], [0, 0.25, 0, 0],
               [0, 0, 0.25, 0], [0, 0, 0, 0.25]]
        test_pass = round(state_fidelity(psi1, psi1), 7) == 1.0 and \
            round(state_fidelity(psi1, psi2), 8) == 0.0 and \
            round(state_fidelity(psi1, rho1), 8) == 1.0 and \
            round(state_fidelity(psi1, mix), 8) == 0.5 and \
            round(state_fidelity(psi2, rho1), 8) == 0.0 and \
            round(state_fidelity(psi2, mix), 8) == 0.5 and \
            round(state_fidelity(rho1, rho1), 8) == 1.0 and \
            round(state_fidelity(rho1, mix), 8) == 0.5 and \
            round(state_fidelity(mix, mix), 8) == 1.0
        self.assertTrue(test_pass)

    def test_purity(self):
        rho1 = [[1, 0], [0, 0]]
        rho2 = [[0.5, 0], [0, 0.5]]
        rho3 = 0.7 * np.array(rho1) + 0.3 * np.array(rho2)
        test_pass = purity(rho1) == 1.0 and \
            purity(rho2) == 0.5 and \
            round(purity(rho3), 10) == 0.745
        self.assertTrue(test_pass)

    def test_concurrence(self):
        psi1 = [1, 0, 0, 0]
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        rho2 = [[0, 0, 0, 0], [0, 0.5, -0.5j, 0],
                [0, 0.5j, 0.5, 0], [0, 0, 0, 0]]
        rho3 = 0.5 * np.array(rho1) + 0.5 * np.array(rho2)
        rho4 = 0.75 * np.array(rho1) + 0.25 * np.array(rho2)
        test_pass = concurrence(psi1) == 0.0 and \
            concurrence(rho1) == 1.0 and \
            concurrence(rho2) == 1.0 and \
            concurrence(rho3) == 0.0 and \
            concurrence(rho4) == 0.5
        self.assertTrue(test_pass)


class TestPauli(QiskitTestCase):
    """Tests for Pauli class"""

    def test_pauli(self):
        v = np.zeros(3)
        w = np.zeros(3)
        v[0] = 1
        w[1] = 1
        v[2] = 1
        w[2] = 1

        p = Pauli(v, w)
        self.log.info(p)
        self.log.info("In label form:")
        self.log.info(p.to_label())
        self.log.info("In matrix form:")
        self.log.info(p.to_matrix())

        q = random_pauli(2)
        self.log.info(q)

        r = inverse_pauli(p)
        self.log.info("In label form:")
        self.log.info(r.to_label())

        self.log.info("Group in tensor order:")
        grp = pauli_group(3, case=1)
        for j in grp:
            self.log.info(j.to_label())

        self.log.info("Group in weight order:")
        grp = pauli_group(3)
        for j in grp:
            self.log.info(j.to_label())

        self.log.info("sign product:")
        p1 = Pauli(np.array([0]), np.array([1]))
        p2 = Pauli(np.array([1]), np.array([1]))
        p3, sgn = sgn_prod(p1, p2)
        self.log.info(p1.to_label())
        self.log.info(p2.to_label())
        self.log.info(p3.to_label())
        self.log.info(sgn)

        self.log.info("sign product reverse:")
        p3, sgn = sgn_prod(p2, p1)
        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.log.info(p3.to_label())
        self.log.info(sgn)

    def test_equality_equal(self):
        """Test equality operator: equal Paulis"""
        p1 = random_pauli(5)
        p2 = deepcopy(p1)
        self.assertTrue(p1 == p2)
        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.log.info(p1 == p2)

    def test_equality_different(self):
        """Test equality operator: different Paulis"""
        p1 = random_pauli(5)
        p2 = deepcopy(p1)
        p2.v[0] = (p1.v[0] + 1) % 2
        self.assertFalse(p1 == p2)
        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.log.info(p1 == p2)

    def test_inequality_equal(self):
        """Test inequality operator: equal Paulis"""
        p1 = random_pauli(5)
        p2 = deepcopy(p1)
        self.assertFalse(p1 != p2)
        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.log.info(p1 != p2)

    def test_inequality_different(self):
        """Test inequality operator: different Paulis"""
        p1 = random_pauli(5)
        p2 = deepcopy(p1)
        p2.v[0] = (p1.v[0] + 1) % 2
        self.assertTrue(p1 != p2)
        self.log.info(p2.to_label())
        self.log.info(p1.to_label())
        self.log.info(p1 != p2)


if __name__ == '__main__':
    unittest.main()
