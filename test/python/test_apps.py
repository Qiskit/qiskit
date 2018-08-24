# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Quick program to test the apps tools modules."""

import unittest
from unittest.mock import patch
from io import StringIO
from scipy import linalg as la
import numpy as np

from qiskit.tools.apps.optimization import make_Hamiltonian, Hamiltonian_from_file, group_paulis, \
        trial_circuit_ry, print_pauli_list_grouped, Energy_Estimate
from qiskit.tools.apps.fermion import parity_set, update_set, flip_set, fermionic_maps, \
        two_qubit_reduction
from qiskit.tools.qi.pauli import Pauli

from .common import QiskitTestCase


class TestQuantumOptimization(QiskitTestCase):
    """Tests for quantum optimization"""

    def test_trial_functions(self):
        """trial_circuit_ry function"""
        entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

        m = 1
        n = 6
        theta = np.zeros(m * n)

        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 456)

        self.log.info("With No measurement:\n")
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, None, None)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 324)

        self.log.info("With Y measurement:\n")
        meas_sting = ['Y' for x in range(n)]
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, meas_sting)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 564)

    def test_group_paulis(self):
        """ qiskit.tools.apps.optimization.group_paulis function"""
        ham_name = self._get_resource_path("../performance/H2/H2Equilibrium.txt")
        zz = np.array([0, 0])
        oo = np.array([1, 1])

        pauli_list = Hamiltonian_from_file(ham_name)
        pauli_list_grouped = group_paulis(pauli_list)

        self.assertEqual(len(pauli_list_grouped), 2)
        r0 = [i[0] for i in pauli_list_grouped]
        r1 = [i[1] for i in pauli_list_grouped]

        self.assertEqual(len(r0), 2)
        r00 = [i[0] for i in r0]
        r01 = [i[1] for i in r0]
        e01 = [Pauli(oo, zz), Pauli(zz, oo)]
        self.assertEqual([0, 0], r00)
        self.assertEqual(e01, r01)

        self.assertEqual(len(r1), 2)
        r10 = [i[0] for i in r1]
        r11 = [i[1] for i in r1]
        e11 = [Pauli(oo, zz), Pauli(zz, oo)]
        self.assertEqual([0.011279956224107712, 0.18093133934472627], r10)
        self.assertEqual(e11, r11)

        expected_stout = ("Post Rotations of TPB set 0:\nZZ\n0\n\nZZ\n0.0112800\n"
                          "II\n-1.0523761\nZI\n0.3979357\nIZ\n"
                          "0.3979357\n\n\nPost Rotations of TPB set 1:\nXX\n0\n\n"
                          "XX\n0.1809313")
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            print_pauli_list_grouped(pauli_list_grouped)

        self.assertMultiLineEqual(fakeOutput.getvalue().strip(), expected_stout)


class TestHamiltonian(QiskitTestCase):
    """ Energy_Estimate and make_Hamiltonian functions """

    def test_hamiltonian(self):
        # pylint: disable=missing-docstring,unexpected-keyword-arg
        # printing an example from a H2 file
        hfile = self._get_resource_path("../performance/H2/H2Equilibrium.txt")
        hamiltonian = make_Hamiltonian(Hamiltonian_from_file(hfile))
        self.log.info(hamiltonian)
        # [[-0.24522469381221926 0 0 0.18093133934472627 ]
        # [0 -1.0636560168497590 0.18093133934472627 0]
        # [0 0.18093133934472627 -1.0636560168497592 0]
        # [0.18093133934472627 0 0 -1.8369675149908681]]

        expected_result = [
            [(-0.245224693812 + 0j), 0j, 0j, (0.180931339345 + 0j)],
            [0j, (-1.06365601685 + 0j), (0.180931339345 + 0j), 0j],
            [0j, (0.180931339345 + 0j), (-1.06365601685 + 0j), 0j],
            [(0.180931339345 + 0j), 0j, 0j, (-1.83696751499 + 0j)]
        ]

        for i in range(4):
            with self.subTest(i=i):
                for result, expected in zip(hamiltonian[i], expected_result[i]):
                    np.testing.assert_almost_equal(result, expected)

        # printing an example from a graph input
        n = 3
        v0 = np.zeros(n)
        v0[2] = 1
        v1 = np.zeros(n)
        v1[0] = 1
        v1[1] = 1
        v2 = np.zeros(n)
        v2[0] = 1
        v2[2] = 1
        v3 = np.zeros(n)
        v3[1] = 1
        v3[2] = 1

        pauli_list = [(1, Pauli(v0, np.zeros(n))), (1, Pauli(v1, np.zeros(n))),
                      (1, Pauli(v2, np.zeros(n))), (1, Pauli(v3, np.zeros(n)))]
        a = make_Hamiltonian(pauli_list)
        self.log.info(a)

        w, v = la.eigh(a, eigvals=(0, 0))
        self.log.info(w)
        self.log.info(v)

        data = {'000': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'001': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'010': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'011': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'100': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'101': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'110': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'111': 10}
        self.log.info(Energy_Estimate(data, pauli_list))


class TestAppsFermion(QiskitTestCase):
    """Tests for apps"""

    def setUp(self):
        self.j1 = 8
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
        """ qiskit.tools.apps.fermion.parity_set function"""
        r = parity_set(self.j1, self.n)
        self.assertEqual(r, [4.])
        r = parity_set(self.j2, self.n)
        self.assertEqual(r.size, 0)

    def test_update_set(self):
        """ qiskit.tools.apps.fermion.update_set function"""
        r = update_set(self.j1, self.n)
        self.assertEqual(r.size, 0)
        r = update_set(self.j2, self.n)
        self.assertEqual(r, [9.])

    def test_flip_set(self):
        """ qiskit.tools.apps.fermion.flip_set function"""
        r = flip_set(self.j1, self.n)
        self.assertEqual(r.size, 0)
        r = flip_set(self.j2, self.n)
        self.assertEqual(r.size, 0)
        r = flip_set(self.j3, self.n)
        self.assertEqual(r.size, 0)

    def test_fermionic_maps_jordan_wigner(self):
        """ qiskit.tools.apps.fermion.fermionic_maps with JORDAN_WIGNER map type"""
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
        """ qiskit.tools.apps.fermion.fermionic_maps with PARITY map type"""
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
        """ qiskit.tools.apps.fermion.fermionic_maps with BINARY_TREE map type"""
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
        """ qiskit.tools.apps.fermion.two_qubit_reduction"""
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
        e = [Pauli(self.zo, self.zz),
             Pauli(self.zz, self.zz),
             Pauli(self.oz, self.zz),
             Pauli(self.oo, self.zz)]
        self.assertEqual(e, r1)


if __name__ == '__main__':
    unittest.main()
