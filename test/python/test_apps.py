# -*- coding: utf-8 -*-

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

"""Quick program to test the apps tools modules."""

import unittest

from qiskit.tools.apps.optimization import make_Hamiltonian
from qiskit.tools.apps.fermion import parity_set, update_set, flip_set, fermionic_maps, \
    two_qubit_reduction
from qiskit.tools.qi.pauli import Pauli

import numpy as np

from .common import QiskitTestCase


class TestAppsFermion(QiskitTestCase):
    """Tests for apps"""

    def setUp(self):
        self.j1 = 20
        self.j2 = 3
        self.j3 = 5
        self.n = 10

        self.a2 = np.arange(4).reshape(2, 2)
        self.a5 = np.arange(16).reshape(2, 2, 2, 2)
        self.e0 = [np.complex(-0.75, 0),
                   np.complex(0.75, 0),
                   np.complex(0, -0.25),
                   np.complex(0, 0.25),
                   np.complex(-1.5, 0),
                   np.complex(1.5, 0)]
        self.zz = np.array([0, 0])
        self.oz = np.array([1, 0])
        self.zo = np.array([0, 1])
        self.oo = np.array([1, 1])

    def test_parity_set(self):
        r = parity_set(self.j1, self.n)
        self.assertEqual(r, [4.])
        r = parity_set(self.j2, self.n)
        self.assertFalse(r)
        self.assertEqual(len(r), 0)

    def test_update_set(self):
        r = update_set(self.j1, self.n)
        self.assertFalse(r)
        r = update_set(self.j2, self.n)
        self.assertEqual(r, [9.])

    def test_flip_set(self):
        r = flip_set(self.j1, self.n)
        self.assertEqual(r, [4.])
        r = flip_set(self.j2, self.n)
        self.assertFalse(r)
        r = flip_set(self.j3, self.n)
        self.assertFalse(r)

    def test_fermionic_maps_jordan_wigner(self):
        self.e0[0] = np.complex(0.75, 0)
        r = fermionic_maps(self.a2, self.a5, "JORDAN_WIGNER")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.oo, self.oo),
             Pauli(self.zz, self.oo),
             Pauli(self.zo, self.oo),
             Pauli(self.oz, self.oo),
             Pauli(self.zo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_fermionic_maps_parity(self):
        r = fermionic_maps(self.a2, self.a5, "PARITY")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.zo, self.oz),
             Pauli(self.zz, self.oz),
             Pauli(self.oo, self.oz),
             Pauli(self.oz, self.oz),
             Pauli(self.oo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_fermionic_maps_binary_tree(self):
        r = fermionic_maps(self.a2, self.a5, "BINARY_TREE")
        self.assertEqual(len(r), 6)
        r0 = [i[0] for i in r]
        self.assertEqual(self.e0, r0)
        r1 = [i[1] for i in r]
        e = [Pauli(self.zo, self.oz),
             Pauli(self.zz, self.oz),
             Pauli(self.oo, self.oz),
             Pauli(self.oz, self.oz),
             Pauli(self.oo, self.zz),
             Pauli(self.zz, self.zz)]
        self.assertEqual(r1, e)

    def test_two_qubit_reduction(self):
        pauli_list = []
        n = 4
        w = np.arange(n ** 2).reshape(n, n)

        for i in range(n):
            for j in range(i):
                if w[i, j] != 0:
                    wp = np.zeros(n)
                    vp = np.zeros(n)
                    vp[n - i - 1] = 1
                    vp[n - j - 1] = 1
                    pauli_list.append((w[i, j], Pauli(vp, wp)))
        r = two_qubit_reduction(pauli_list, 10)
        r0 = [i[0] for i in r]
        self.assertEqual([-5, -8, -2, 13], r0)
        r1 = [i[1] for i in r]
        e  = [Pauli(self.zo, self.zz),
              Pauli(self.zz, self.zz),
              Pauli(self.oz, self.zz),
              Pauli(self.oo, self.zz)]
        self.assertEqual(e,r1)


if __name__ == '__main__':
    unittest.main()
